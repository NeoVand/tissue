/* eslint-disable @typescript-eslint/no-explicit-any */
import { init, defaultDevice, jit, tree, numpy as np } from '@jax-js/jax';
import {
	MODEL_CONFIG as cfg,
	NEURON_COUNT,
	type Atlas,
	type Backend,
	type Checkpoint,
	type EngineEvent,
	type Initialization,
	type Metrics,
	type Probe,
	type BindingExample,
	type RepairOptions,
	type RepairResult,
	type RepairObservation,
	type RepairArm
} from '../protocol';
import {
	assignmentSplits,
	CALIBRATION_EXAMPLES,
	TEST_EXAMPLES,
	RandomStream,
	sampleExample
} from './dataset';
import {
	answerLoss,
	deserializeTree,
	forward,
	initializeParameters,
	lesionMask,
	oneHotBatch,
	serializeTree,
	validateSerialized,
	type Params
} from './transformer';
import { fusedAdam, type AdamState } from './adam';
import { chooseRepairArms, freezeUnselected, trainableMask } from './repair';
import type { QueryMeasurement, QuerySplitMeasurement } from '../query-protocol';
import { QUERY_BATCH_SIZE, pairedQueryGroups } from './query-dataset';

const yieldTask = () => new Promise<void>((resolve) => setTimeout(resolve, 0));
const argmax = (values: ArrayLike<number>) => {
	let best = 0;
	for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
	return best;
};

/** The worker is the sole owner of this object and all its device arrays. */
export class ModelRuntime {
	private params: Params | null = null;
	private optimizer: ReturnType<typeof fusedAdam<any>> | null = null;
	private state: AdamState | null = null;
	private inference: any = null;
	private capture: any = null;
	private random = new RandomStream(42);
	private seed = 42;
	private step = 0;
	private elapsedMs = 0;
	private trainLoss: number | null = null;
	private backend: Backend = 'cpu';
	private stopRequested = false;
	private latestMetrics: Metrics | null = null;

	async initialize(
		seed: number,
		preference: Backend | 'auto' = 'auto',
		emit: (event: EngineEvent) => void = () => {}
	): Promise<Initialization> {
		this.dispose();
		if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff)
			throw new Error('Seed must be an unsigned 32-bit integer');
		const candidates: Backend[] = preference === 'auto' ? ['webgpu', 'wasm', 'cpu'] : [preference];
		let selected: Backend | undefined;
		for (const candidate of candidates) {
			emit({
				type: 'status',
				message: `Initializing ${candidate === 'webgpu' ? 'WebGPU' : candidate.toUpperCase()}…`
			});
			try {
				const available = await init(candidate);
				if (available.includes(candidate)) {
					defaultDevice(candidate);
					const sanity = await np
						.array(new Float32Array([1, 2, 3]))
						.add(2)
						.data();
					if (sanity[0] !== 3 || sanity[1] !== 4 || sanity[2] !== 5)
						throw new Error(`${candidate} failed the numerical self-check`);
					selected = candidate;
					break;
				}
			} catch (error) {
				if (preference !== 'auto') throw error;
			}
		}
		if (!selected) throw new Error('No numerical backend could initialize');
		this.backend = selected;
		defaultDevice(selected);
		this.seed = seed;
		this.random = new RandomStream(seed ^ 0x74726169);
		this.params = initializeParameters(seed);
		this.optimizer = fusedAdam(answerLoss, { lr: cfg.learningRate, b1: 0.9, b2: 0.99 });
		this.state = this.optimizer.init(this.params);
		this.inference = jit(
			(p: Params, tokens: any, positions: any, mask: any) =>
				forward(p, tokens, positions, mask).logprobs
		);
		this.capture = jit((p: Params, tokens: any, positions: any, mask: any) => {
			const result = forward(p, tokens, positions, mask, true);
			return [result.logprobs, ...result.activations];
		});
		this.step = 0;
		this.elapsedMs = 0;
		this.trainLoss = null;
		this.stopRequested = false;
		emit({ type: 'status', message: 'Compiling the fixed held-out evaluation…' });
		const metrics = await this.evaluate();
		return {
			seed,
			backend: selected,
			parameterCount: (tree.leaves(this.params) as any[]).reduce(
				(total, leaf) => total + leaf.size,
				0
			),
			metrics,
			trainAssignments: assignmentSplits.train.length,
			calibrationAssignments: assignmentSplits.calibration.length,
			testAssignments: assignmentSplits.test.length
		};
	}
	private async predict(
		examples: BindingExample[],
		neuron?: number | null,
		parameters = this.params
	): Promise<number[][]> {
		if (!parameters) throw new Error('Initialize the model first');
		const batch = oneHotBatch(examples);
		batch.targets.dispose();
		const flat = (await this.inference(
			tree.ref(parameters),
			batch.tokens,
			batch.positions,
			lesionMask(neuron)
		).data()) as Float32Array;
		return examples.map((_, index) =>
			Array.from(flat.subarray(index * cfg.vocabulary.length, (index + 1) * cfg.vocabulary.length))
		);
	}
	async evaluate(): Promise<Metrics> {
		if (!this.params) throw new Error('Initialize the model first');
		let loss = 0,
			correct = 0;
		for (let start = 0; start < TEST_EXAMPLES.length; start += cfg.batchSize) {
			const examples = TEST_EXAMPLES.slice(start, start + cfg.batchSize);
			const predictions = await this.predict(examples);
			predictions.forEach((logprobs, index) => {
				const mass = logprobs.reduce((sum, value) => sum + Math.exp(value), 0);
				if (!Number.isFinite(mass) || Math.abs(mass - 1) > 1e-3)
					throw new Error(`Numerical backend produced invalid probabilities (mass ${mass})`);
				loss -= logprobs[examples[index].answerId];
				correct += +(argmax(logprobs) === examples[index].answerId);
			});
		}
		const metrics: Metrics = {
			step: this.step,
			trainLoss: this.trainLoss,
			validationLoss: loss / TEST_EXAMPLES.length,
			accuracy: correct / TEST_EXAMPLES.length,
			elapsedMs: this.elapsedMs,
			backend: this.backend,
			validationExamples: TEST_EXAMPLES.length,
			chanceAccuracy: 1 / 8
		};
		this.latestMetrics = metrics;
		return metrics;
	}
	async train(steps: number, emit: (event: EngineEvent) => void = () => {}): Promise<Metrics> {
		if (!this.params || !this.optimizer || !this.state)
			throw new Error('Initialize the model first');
		if (!Number.isInteger(steps) || steps < 1 || steps > 10000)
			throw new Error('Training steps must be an integer from 1 to 10000');
		this.stopRequested = false;
		emit({
			type: 'status',
			message:
				this.step === 0
					? 'Compiling gradients and Adam; the first update takes longer…'
					: 'Training answer-token prediction…'
		});
		for (let i = 0; i < steps && !this.stopRequested; i++) {
			const start = performance.now();
			const examples = Array.from({ length: cfg.batchSize }, () =>
				sampleExample('train', this.random)
			);
			const batch = oneHotBatch(examples);
			const [loss, nextParams, nextState] = this.optimizer.step(
				this.params,
				this.state,
				batch.tokens,
				batch.positions,
				batch.targets,
				lesionMask()
			);
			this.params = nextParams;
			this.state = nextState;
			this.trainLoss = (await loss.data())[0];
			this.step++;
			this.elapsedMs += performance.now() - start;
			if (!Number.isFinite(this.trainLoss))
				throw new Error(
					`Training diverged at step ${this.step}; reset to restore a finite checkpoint`
				);
			if (this.step % 25 === 0 || i === steps - 1)
				emit({ type: 'metrics', metrics: await this.evaluate() });
			// GPU readback completes each step, then this lets pause/reset messages in.
			await yieldTask();
		}
		if (this.latestMetrics?.step !== this.step)
			emit({ type: 'metrics', metrics: await this.evaluate() });
		return this.latestMetrics!;
	}
	pause() {
		this.stopRequested = true;
	}
	async probe(exampleIndex = 0, neuron?: number | null): Promise<Probe> {
		if (!this.params) throw new Error('Initialize the model first');
		if (
			!Number.isInteger(exampleIndex) ||
			exampleIndex < 0 ||
			exampleIndex >= CALIBRATION_EXAMPLES.length
		)
			throw new Error('Probe example is outside the fixed calibration set');
		const example = CALIBRATION_EXAMPLES[exampleIndex];
		const batch = oneHotBatch([example]);
		batch.targets.dispose();
		const [logprobs, ...layers] = this.capture(
			tree.ref(this.params),
			batch.tokens,
			batch.positions,
			lesionMask(neuron)
		) as any[];
		const lp = (await logprobs.data()) as Float32Array;
		const captured = await Promise.all(
			layers.map(async (layer) => (await layer.data()) as Float32Array)
		);
		const activations = Array.from({ length: cfg.sequenceLength }, (_, token) =>
			Array.from(
				{ length: NEURON_COUNT },
				(_, node) =>
					captured[Math.floor(node / cfg.hidden)][token * cfg.hidden + (node % cfg.hidden)]
			)
		);
		const probabilities = Array.from(lp.slice(6), (value) => Math.exp(value));
		return {
			step: this.step,
			exampleIndex,
			example,
			probabilities,
			otherProbability: Math.max(0, 1 - probabilities.reduce((a, b) => a + b, 0)),
			predictedAnswer: cfg.vocabulary[argmax(lp)],
			activations,
			lesionNeuron: neuron ?? null
		};
	}
	async captureAtlas(
		includeEffects = false,
		emit: (event: EngineEvent) => void = () => {}
	): Promise<Atlas> {
		if (!this.params) throw new Error('Initialize the model first');
		this.stopRequested = false;
		emit({ type: 'status', message: 'Measuring the fixed calibration responses…' });
		const examples = CALIBRATION_EXAMPLES;
		const batch = oneHotBatch(examples);
		batch.targets.dispose();
		const [logprobs, ...layers] = this.capture(
			tree.ref(this.params),
			batch.tokens,
			batch.positions,
			lesionMask()
		) as any[];
		const lp = (await logprobs.data()) as Float32Array;
		const captured = await Promise.all(
			layers.map(async (layer) => (await layer.data()) as Float32Array)
		);
		const activationFingerprints = Array.from({ length: NEURON_COUNT }, (_, node) =>
			Array.from(
				{ length: examples.length * cfg.sequenceLength },
				(_, context) =>
					captured[Math.floor(node / cfg.hidden)][context * cfg.hidden + (node % cfg.hidden)]
			)
		);
		const outgoingWeights = (
			await Promise.all(
				this.params.layers.map(async (layer) => {
					const flat = (await layer.mlpFc2.ref.data()) as Float32Array;
					return Array.from({ length: cfg.hidden }, (_, neuron) =>
						Array.from(flat.subarray(neuron * cfg.width, (neuron + 1) * cfg.width))
					);
				})
			)
		).flat();
		let correct = 0;
		examples.forEach((example, index) => {
			correct += +(
				argmax(lp.subarray(index * cfg.vocabulary.length, (index + 1) * cfg.vocabulary.length)) ===
				example.answerId
			);
		});
		let effectFingerprints: number[][] | undefined;
		if (includeEffects) {
			effectFingerprints = [];
			for (let neuron = 0; neuron < NEURON_COUNT; neuron++) {
				if (this.stopRequested) throw new Error('Intervention measurement cancelled');
				const lesioned = await this.predict(examples, neuron);
				effectFingerprints.push(
					lesioned.flatMap((row, index) =>
						row
							.slice(6)
							.map(
								(value, digit) =>
									Math.exp(value) - Math.exp(lp[index * cfg.vocabulary.length + 6 + digit])
							)
					)
				);
				if (neuron % 8 === 7 || neuron === NEURON_COUNT - 1) {
					emit({
						type: 'measurement',
						completed: neuron + 1,
						total: NEURON_COUNT,
						step: this.step
					});
					await yieldTask();
				}
			}
		}
		return {
			step: this.step,
			seed: this.seed,
			capturedAt: new Date().toISOString(),
			neurons: Array.from({ length: NEURON_COUNT }, (_, id) => ({
				id,
				layer: Math.floor(id / cfg.hidden),
				channel: id % cfg.hidden
			})),
			activationFingerprints,
			effectFingerprints,
			outgoingWeights,
			examples,
			calibrationAccuracy: correct / examples.length,
			intervention: includeEffects ? 'zero-all-token-positions' : 'not-measured'
		};
	}
	/** Fixed matched-query groups. Measurement never touches parameters, Adam, or the training RNG. */
	async measureQueryShifts(
		emit: (event: EngineEvent) => void = () => {}
	): Promise<QueryMeasurement> {
		if (!this.params || !this.state) throw new Error('Initialize the model first');
		this.stopRequested = false;
		const started = performance.now();
		const capturedAt = new Date().toISOString();
		const before = JSON.stringify(await this.exportCheckpoint());
		const hash = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(before));
		const checkpointHash = Array.from(new Uint8Array(hash), (byte) =>
			byte.toString(16).padStart(2, '0')
		).join('');
		const groups = {
			calibration: pairedQueryGroups('calibration'),
			test: pairedQueryGroups('test')
		};
		const splits = {} as QueryMeasurement['splits'];
		let completed = 0;
		const checkCancellation = () => {
			if (this.stopRequested) throw new Error('Paired-query measurement cancelled');
		};
		emit({ type: 'measurement', completed: 0, total: NEURON_COUNT * 2, step: this.step });
		for (const split of ['calibration', 'test'] as const) {
			checkCancellation();
			emit({ type: 'status', message: `Measuring paired queries on fixed ${split} assignments…` });
			const examples = groups[split].flatMap((group) => group.examples);
			const measured: QuerySplitMeasurement = {
				intactProbabilities: [],
				otherProbabilities: [],
				activations: Array.from({ length: NEURON_COUNT }, () => []),
				effects: Array.from({ length: NEURON_COUNT }, () => []),
				accuracy: 0,
				prefixMaxDifference: 0,
				capturePredictionMaxDifference: 0,
				probabilityMassMaxError: 0
			};
			const validate = (rows: number[][]) => {
				for (const row of rows) {
					if (row.length !== cfg.vocabulary.length || row.some((value) => !Number.isFinite(value)))
						throw new Error('Paired-query measurement received invalid log probabilities');
					const mass = row.reduce((sum, value) => sum + Math.exp(value), 0);
					measured.probabilityMassMaxError = Math.max(
						measured.probabilityMassMaxError,
						Math.abs(mass - 1)
					);
					if (Math.abs(mass - 1) > 1e-3)
						throw new Error(`Paired-query measurement produced invalid probability mass ${mass}`);
				}
			};
			let correct = 0;
			// Both splits contain 48 examples. Keeping whole triples together uses exactly two B=24 batches.
			for (let start = 0; start < examples.length; start += QUERY_BATCH_SIZE) {
				checkCancellation();
				const batchExamples = examples.slice(start, start + QUERY_BATCH_SIZE);
				const batch = oneHotBatch(batchExamples);
				batch.targets.dispose();
				const outputs = this.capture(
					tree.ref(this.params),
					batch.tokens,
					batch.positions,
					lesionMask()
				) as any[];
				// Read all outputs before checking them; each .data() consumes its owned array.
				const [flat, ...layers] = await Promise.all(
					outputs.map(async (output) => (await output.data()) as Float32Array)
				);
				const rows = batchExamples.map((_, i) =>
					Array.from(flat.subarray(i * cfg.vocabulary.length, (i + 1) * cfg.vocabulary.length))
				);
				validate(rows);
				const ordinary = await this.predict(batchExamples);
				validate(ordinary);
				for (let i = 0; i < rows.length; i++) {
					correct += +(argmax(rows[i]) === batchExamples[i].answerId);
					measured.intactProbabilities.push(rows[i].slice(6).map(Math.exp));
					measured.otherProbabilities.push(
						rows[i].slice(0, 6).reduce((sum, value) => sum + Math.exp(value), 0)
					);
					for (let output = 0; output < cfg.vocabulary.length; output++)
						measured.capturePredictionMaxDifference = Math.max(
							measured.capturePredictionMaxDifference,
							Math.abs(Math.exp(rows[i][output]) - Math.exp(ordinary[i][output]))
						);
					for (let neuron = 0; neuron < NEURON_COUNT; neuron++) {
						const values = layers[Math.floor(neuron / cfg.hidden)];
						const channel = neuron % cfg.hidden;
						const offset = i * cfg.sequenceLength * cfg.hidden;
						measured.activations[neuron].push(
							values[offset + (cfg.sequenceLength - 1) * cfg.hidden + channel]
						);
						const firstQueryOffset = Math.floor(i / 3) * 3 * cfg.sequenceLength * cfg.hidden;
						for (let token = 0; token < cfg.sequenceLength - 1; token++)
							measured.prefixMaxDifference = Math.max(
								measured.prefixMaxDifference,
								Math.abs(
									values[offset + token * cfg.hidden + channel] -
										values[firstQueryOffset + token * cfg.hidden + channel]
								)
							);
					}
				}
				if (layers.some((values) => values.some((value) => !Number.isFinite(value) || value < 0)))
					throw new Error('Paired-query capture produced invalid post-ReLU activations');
				if (measured.capturePredictionMaxDifference > 1e-6)
					throw new Error('Paired-query capture disagrees with ordinary prediction');
				if (measured.prefixMaxDifference > 1e-6)
					throw new Error('A future query changed an earlier causal-prefix activation');
				await yieldTask();
			}
			measured.accuracy = correct / examples.length;
			for (let neuron = 0; neuron < NEURON_COUNT; neuron++) {
				for (let start = 0; start < examples.length; start += QUERY_BATCH_SIZE) {
					checkCancellation();
					const rows = await this.predict(examples.slice(start, start + QUERY_BATCH_SIZE), neuron);
					validate(rows);
					for (let i = 0; i < rows.length; i++)
						for (let digit = 0; digit < 8; digit++)
							measured.effects[neuron].push(
								Math.exp(rows[i][digit + 6]) - measured.intactProbabilities[start + i][digit]
							);
				}
				completed++;
				if (completed % 8 === 0) {
					emit({ type: 'measurement', completed, total: NEURON_COUNT * 2, step: this.step });
					await yieldTask();
				}
			}
			splits[split] = measured;
		}
		checkCancellation();
		const outgoingWeights = (
			await Promise.all(
				this.params.layers.map(async (layer) => {
					const flat = (await layer.mlpFc2.ref.data()) as Float32Array;
					return Array.from({ length: cfg.hidden }, (_, channel) =>
						Array.from(flat.subarray(channel * cfg.width, (channel + 1) * cfg.width))
					);
				})
			)
		).flat();
		if (JSON.stringify(await this.exportCheckpoint()) !== before)
			throw new Error('Paired-query measurement modified the source checkpoint');
		checkCancellation();
		return {
			version: 1,
			design: 'paired-query-v1',
			seed: this.seed,
			step: this.step,
			backend: this.backend,
			capturedAt,
			elapsedMs: performance.now() - started,
			groups,
			splits,
			outgoingWeights,
			checkpointHash,
			checkpointHashAlgorithm: 'sha256-full-checkpoint-json-v1',
			checkpointPreserved: true,
			intervention: 'zero-all-token-positions'
		};
	}
	async exportCheckpoint(): Promise<Checkpoint> {
		if (!this.params || !this.state) throw new Error('Initialize the model first');
		return {
			version: 1,
			architecture: 'binding-transformer-v1',
			seed: this.seed,
			step: this.step,
			elapsedMs: this.elapsedMs,
			trainLoss: this.trainLoss,
			parameters: await serializeTree(this.params),
			optimizer: {
				m: await serializeTree(this.state.m),
				v: await serializeTree(this.state.v),
				t: this.state.t
			},
			randomState: this.random.state
		};
	}
	async loadCheckpoint(checkpoint: Checkpoint): Promise<Metrics> {
		if (!this.params || !this.state) throw new Error('Initialize the model first');
		if (
			!checkpoint ||
			checkpoint.version !== 1 ||
			checkpoint.architecture !== 'binding-transformer-v1' ||
			!Number.isInteger(checkpoint.step) ||
			checkpoint.step < 0 ||
			!Number.isInteger(checkpoint.randomState) ||
			checkpoint.randomState < 0 ||
			checkpoint.randomState > 0xffffffff ||
			!Number.isInteger(checkpoint.seed) ||
			checkpoint.seed < 0 ||
			checkpoint.seed > 0xffffffff ||
			!Number.isFinite(checkpoint.elapsedMs) ||
			checkpoint.elapsedMs < 0 ||
			(checkpoint.trainLoss !== null && !Number.isFinite(checkpoint.trainLoss)) ||
			checkpoint.optimizer?.t !== checkpoint.step
		)
			throw new Error('Unsupported or invalid checkpoint');
		validateSerialized(this.params, checkpoint.parameters);
		validateSerialized(this.params, checkpoint.optimizer.m);
		validateSerialized(this.params, checkpoint.optimizer.v);
		if (checkpoint.optimizer.v.some((leaf) => leaf.values.some((value) => value < 0)))
			throw new Error('Invalid Adam second moment');
		const nextParams = deserializeTree(this.params, checkpoint.parameters);
		const m = deserializeTree(this.params, checkpoint.optimizer.m);
		const v = deserializeTree(this.params, checkpoint.optimizer.v);
		tree.dispose(this.params);
		tree.dispose(this.state.m);
		tree.dispose(this.state.v);
		this.params = nextParams;
		this.state = { m, v, t: checkpoint.optimizer.t };
		this.step = checkpoint.step;
		this.seed = checkpoint.seed;
		this.elapsedMs = checkpoint.elapsedMs;
		this.trainLoss = checkpoint.trainLoss;
		this.random = new RandomStream(checkpoint.randomState);
		return this.evaluate();
	}
	/** A matched-budget pilot, keeping the resident training trajectory untouched. */
	async controlledRepair(
		options: RepairOptions,
		emit: (event: EngineEvent) => void = () => {}
	): Promise<RepairResult> {
		if (!this.params || !this.state) throw new Error('Initialize the model first');
		const { atlas, lesionNeuron } = options;
		const steps = options.steps ?? 50,
			count = options.neighborCount ?? 8;
		if (atlas.step !== this.step || atlas.seed !== this.seed)
			throw new Error('Recapture this checkpoint before comparing repair');
		if (
			!Number.isInteger(lesionNeuron) ||
			lesionNeuron < 0 ||
			lesionNeuron >= NEURON_COUNT ||
			!Number.isInteger(steps) ||
			steps < 1 ||
			steps > 250 ||
			!Number.isInteger(count) ||
			count < 1 ||
			count > 32
		)
			throw new Error('Invalid repair budget or neuron');
		const selections = chooseRepairArms(atlas, lesionNeuron, count, this.seed);
		const serialized = await serializeTree(this.params);
		const evaluate = async (
			parameters: Params,
			lesion: number | null,
			update: number
		): Promise<RepairObservation & { answerProbabilities: number[] }> => {
			let loss = 0,
				correct = 0;
			const answerProbabilities: number[] = [];
			for (let offset = 0; offset < TEST_EXAMPLES.length; offset += cfg.batchSize) {
				const examples = TEST_EXAMPLES.slice(offset, offset + cfg.batchSize);
				const rows = await this.predict(examples, lesion, parameters);
				rows.forEach((row, index) => {
					loss -= row[examples[index].answerId];
					answerProbabilities.push(Math.exp(row[examples[index].answerId]));
					correct += +(argmax(row) === examples[index].answerId);
				});
			}
			return {
				answerProbabilities,
				meanAnswerProbability:
					answerProbabilities.reduce((a, b) => a + b, 0) / answerProbabilities.length,
				update,
				loss: loss / TEST_EXAMPLES.length,
				accuracy: correct / TEST_EXAMPLES.length
			};
		};
		this.stopRequested = false;
		const intact = await evaluate(this.params, null, 0);
		const lesioned = await evaluate(this.params, lesionNeuron, 0);
		const solver = fusedAdam(
			(p: Params, tokens: any, positions: any, targets: any, lesions: any, selected: Params) =>
				answerLoss(freezeUnselected(p, selected), tokens, positions, targets, lesions),
			{ lr: cfg.learningRate, b1: 0.9, b2: 0.99 }
		);
		const arms: RepairArm[] = [];
		const finalProbabilities: number[][] = [];
		try {
			for (const selection of selections) {
				emit({
					type: 'status',
					message: `Repair comparison: ${selection.method}, ${count} neurons, ${steps} matched updates…`
				});
				let parameters = deserializeTree(this.params, serialized);
				let state = solver.init(parameters);
				const mask = trainableMask(this.params, selection.neurons);
				const random = new RandomStream(this.random.state);
				const curve: RepairObservation[] = [lesioned];
				try {
					for (let update = 1; update <= steps; update++) {
						if (this.stopRequested)
							throw new Error('Repair experiment cancelled; original checkpoint preserved');
						const batch = oneHotBatch(
							Array.from({ length: cfg.batchSize }, () => sampleExample('train', random))
						);
						const [loss, nextParams, nextState] = solver.step(
							parameters,
							state,
							batch.tokens,
							batch.positions,
							batch.targets,
							lesionMask(lesionNeuron),
							tree.ref(mask)
						);
						parameters = nextParams;
						state = nextState;
						const value = (await loss.data())[0];
						if (!Number.isFinite(value))
							throw new Error('Repair diverged; original checkpoint preserved');
						if (update % 10 === 0 || update === steps)
							curve.push(await evaluate(parameters, lesionNeuron, update));
						await yieldTask();
					}
					finalProbabilities.push(
						(await evaluate(parameters, lesionNeuron, steps)).answerProbabilities
					);
					arms.push({ ...selection, trainableParameters: count * cfg.width * 2, curve });
				} finally {
					tree.dispose(parameters);
					tree.dispose(state.m);
					tree.dispose(state.v);
					tree.dispose(mask);
				}
			}
		} finally {
			solver.dispose();
		}
		return {
			step: this.step,
			seed: this.seed,
			lesionNeuron,
			steps,
			neighborCount: count,
			intact,
			lesioned,
			arms,
			testExamples: TEST_EXAMPLES.length,
			testPredictions: TEST_EXAMPLES.map((example, index) => ({
				id: example.id,
				prompt: example.tokens.join(''),
				answer: example.answer,
				intactProbability: intact.answerProbabilities[index],
				lesionedProbability: lesioned.answerProbabilities[index],
				armProbabilities: arms.map((arm, armIndex) => ({
					method: arm.method,
					probability: finalProbabilities[armIndex][index]
				}))
			})),
			completedAt: new Date().toISOString(),
			caveat:
				'Single-seed pilot. Same layer, parameter count, batches, optimizer reset, and update budget; random control is not matched for activation strength. Selecting lesions after viewing test damage introduces selection bias.'
		};
	}
	dispose() {
		this.stopRequested = true;
		if (this.params) tree.dispose(this.params);
		if (this.state) {
			tree.dispose(this.state.m);
			tree.dispose(this.state.v);
		}
		this.optimizer?.dispose();
		this.inference?.dispose();
		this.capture?.dispose();
		this.params = null;
		this.state = null;
		this.optimizer = null;
		this.inference = null;
		this.capture = null;
		this.latestMetrics = null;
	}
}
