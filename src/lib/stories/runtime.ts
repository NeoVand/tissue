/* eslint-disable @typescript-eslint/no-explicit-any */
import { init, defaultDevice, numpy as np, jit, tree } from '@jax-js/jax';
import { fusedAdam, type AdamState, type FusedAdam } from '../lab/model/adam';
import {
	STORY_PRESETS,
	storyParameterCount,
	storyUnitCount,
	validateStoryConfig,
	validateStoryCheckpoint,
	type StoryAtlas,
	type StoryBackend,
	type StoryCheckpoint,
	type StoryConfig,
	type StoryEvent,
	type StoryGeneration,
	type StoryGenerationOptions,
	type StoryInitialization,
	type StoryMetrics,
	type StoryPreset,
	type StoryProbe
} from './protocol';
import {
	STORY_CORPUS_ID,
	StoryRandom,
	decodeStory,
	encodeStoryPrompt,
	loadStoryCorpus,
	type StoryCorpus
} from './dataset';
import {
	initializeStoryParameters,
	storyForward,
	storyLoss,
	storyBatch,
	storyMask,
	serializeStoryTree,
	deserializeStoryTree,
	type StoryParams
} from './model';

const yieldTask = () => new Promise<void>((resolve) => setTimeout(resolve, 0));
const argmax = (row: ArrayLike<number>) => {
	let best = 0;
	for (let i = 1; i < row.length; i++) if (row[i] > row[best]) best = i;
	return best;
};
type Emit = (event: StoryEvent) => void;

/** Sole owner of GPU model state. The worker serializes every operation except pause. */
export class StoryRuntime {
	private params: StoryParams | null = null;
	private state: AdamState | null = null;
	private optimizer: FusedAdam<any> | null = null;
	private inference: any = null;
	private capture: any = null;
	private config!: StoryConfig;
	private corpus!: StoryCorpus;
	private seed = 42;
	private modelId = '';
	private step = 0;
	private elapsedMs = 0;
	private stepMs = 0;
	private trainLoss: number | null = null;
	private random = new StoryRandom(42);
	private backend: StoryBackend = 'cpu';
	private stopped = false;
	private metrics: StoryMetrics | null = null;
	async initialize(
		preset: StoryPreset | StoryConfig,
		seed = 42,
		preference: StoryBackend | 'auto' = 'auto',
		corpus?: StoryCorpus,
		emit: Emit = () => {},
		checkpoint?: StoryCheckpoint
	): Promise<StoryInitialization> {
		const config = typeof preset === 'string' ? STORY_PRESETS[preset] : preset;
		validateStoryConfig(config);
		if (!['auto', 'webgpu', 'wasm', 'cpu'].includes(preference))
			throw new Error('Unknown story backend');
		if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff)
			throw new Error('Seed must be an unsigned 32-bit integer');
		if (checkpoint) validateStoryCheckpoint(checkpoint);
		emit({ type: 'status', message: 'Loading and verifying the TinyStories corpus…' });
		const data = corpus ?? (await loadStoryCorpus(config));
		if (data.info.id !== STORY_CORPUS_ID) throw new Error('Unknown TinyStories corpus');
		this.dispose();
		const candidates: StoryBackend[] =
			preference === 'auto'
				? storyParameterCount(config) <= storyParameterCount(STORY_PRESETS.small)
					? ['webgpu', 'wasm']
					: ['webgpu']
				: [preference];
		let selected: StoryBackend | undefined;
		for (const backend of candidates) {
			emit({
				type: 'status',
				message: `Initializing ${backend.toUpperCase()} for ${storyParameterCount(config).toLocaleString()} parameters…`
			});
			try {
				const devices = await init(backend);
				if (!devices.includes(backend)) continue;
				defaultDevice(backend);
				const check = await np
					.array(new Float32Array([1, 2, 3]))
					.add(2)
					.data();
				if (check[0] !== 3 || check[1] !== 4 || check[2] !== 5)
					throw new Error('Backend failed its numerical known-answer check');
				selected = backend;
				break;
			} catch (reason) {
				if (preference !== 'auto') throw reason;
			}
		}
		if (!selected)
			throw new Error(
				'This model needs WebGPU. Use the Small preset for a practical WASM fallback.'
			);
		this.backend = selected;
		this.config = { ...config };
		this.corpus = data;
		this.seed = seed;
		this.modelId = crypto.randomUUID();
		this.step = 0;
		this.elapsedMs = 0;
		this.stepMs = 0;
		this.trainLoss = null;
		this.random = new StoryRandom(seed ^ 0x7374726e);
		this.stopped = false;
		this.params = checkpoint
			? deserializeStoryTree(config, checkpoint.parameters)
			: initializeStoryParameters(config, seed);
		this.optimizer = fusedAdam((p, t, pos, y, mask) => storyLoss(p, config, t, pos, y, mask), {
			lr: config.learningRate,
			b1: 0.9,
			b2: 0.99
		});
		this.state = checkpoint
			? {
					m: deserializeStoryTree(config, checkpoint.optimizer.m),
					v: deserializeStoryTree(config, checkpoint.optimizer.v),
					t: checkpoint.step
				}
			: this.optimizer.init(this.params);
		if (checkpoint) this.restoreMetadata(checkpoint);
		this.inference = jit(
			(p: StoryParams, t: any, pos: any, mask: any) =>
				storyForward(p, config, t, pos, mask).logprobs
		);
		this.capture = jit((p: StoryParams, t: any, pos: any, mask: any) => {
			const output = storyForward(p, config, t, pos, mask, true);
			return [output.logprobs, ...output.activations];
		});
		emit({ type: 'status', message: 'Compiling fixed held-out next-character evaluation…' });
		await this.evaluate();
		return this.initialization();
	}
	private initialization(): StoryInitialization {
		return {
			modelId: this.modelId,
			config: { ...this.config },
			seed: this.seed,
			backend: this.backend,
			parameterCount: storyParameterCount(this.config),
			unitCount: storyUnitCount(this.config),
			corpus: this.corpus.info,
			metrics: this.metrics!
		};
	}
	private requireReady() {
		if (!this.params || !this.state) throw new Error('Initialize a story model first');
	}
	private validateProbabilities(flat: Float32Array) {
		for (let offset = 0; offset < flat.length; offset += 96) {
			let mass = 0;
			for (let j = 0; j < 96; j++) {
				const value = flat[offset + j];
				if (!Number.isFinite(value)) throw new Error('Nonfinite story model probabilities');
				mass += Math.exp(value);
			}
			if (Math.abs(mass - 1) > 1e-3) throw new Error(`Story probability mass is invalid (${mass})`);
		}
	}
	private async predict(
		windows: ArrayLike<number>[],
		neuron?: number | null
	): Promise<Float32Array> {
		this.requireReady();
		const mask = storyMask(this.config, neuron);
		const batch = storyBatch(this.config, windows);
		batch.targets.dispose();
		const flat = (await this.inference(
			tree.ref(this.params),
			batch.tokens,
			batch.positions,
			mask
		).data()) as Float32Array;
		this.validateProbabilities(flat);
		return flat;
	}
	async evaluate(): Promise<StoryMetrics> {
		this.requireReady();
		const c = this.config,
			examples = this.corpus.evaluation;
		let loss = 0,
			correct = 0,
			unigramLoss = 0,
			unigramCorrect = 0;
		const unigramBest = argmax(this.corpus.unigram);
		for (let start = 0; start < examples.length; start += c.batchSize) {
			const windows = examples.slice(start, start + c.batchSize),
				lp = await this.predict(windows);
			windows.forEach((window, b) => {
				for (let t = 0; t < c.context; t++) {
					const target = window[t + 1],
						offset = (b * c.context + t) * 96;
					loss -= lp[offset + target];
					correct += +(argmax(lp.subarray(offset, offset + 96)) === target);
					unigramLoss -= Math.log(this.corpus.unigram[target]);
					unigramCorrect += +(unigramBest === target);
				}
			});
		}
		const count = examples.length * c.context;
		this.metrics = {
			step: this.step,
			trainLoss: this.trainLoss,
			validationLoss: loss / count,
			validationAccuracy: correct / count,
			unigramLoss: unigramLoss / count,
			unigramAccuracy: unigramCorrect / count,
			uniformLoss: Math.log(96),
			evaluationTokens: count,
			trainedTokens: this.step * c.batchSize * c.context,
			elapsedMs: this.elapsedMs,
			stepMs: this.stepMs,
			backend: this.backend
		};
		return this.metrics;
	}
	async train(steps: number, emit: Emit = () => {}): Promise<StoryMetrics> {
		this.requireReady();
		if (!Number.isInteger(steps) || steps < 1 || steps > 100)
			throw new Error('A story training burst must contain 1 to 100 updates');
		this.stopped = false;
		const c = this.config;
		emit({
			type: 'status',
			message:
				this.step === 0
					? 'Compiling story gradients and Adam…'
					: 'Training next-character prediction…'
		});
		for (let i = 0; i < steps && !this.stopped; i++) {
			const started = performance.now();
			const windows = Array.from({ length: c.batchSize }, () => {
				const start = Math.floor(this.random.next() * (this.corpus.info.trainTokens - c.context));
				return this.corpus.tokens.subarray(start, start + c.context + 1);
			});
			const batch = storyBatch(c, windows);
			const [loss, parameters, state] = this.optimizer!.step(
				this.params,
				this.state!,
				batch.tokens,
				batch.positions,
				batch.targets,
				storyMask(c)
			);
			this.params = parameters;
			this.state = state;
			this.trainLoss = Number((await loss.data())[0]);
			this.step++;
			this.stepMs = performance.now() - started;
			this.elapsedMs += this.stepMs;
			if (!Number.isFinite(this.trainLoss) || this.trainLoss <= 0)
				throw new Error(`Invalid story training loss at step ${this.step}`);
			if (this.step % 25 === 0 || i === steps - 1)
				emit({ type: 'metrics', metrics: await this.evaluate() });
			await yieldTask();
		}
		if (this.metrics?.step !== this.step) emit({ type: 'metrics', metrics: await this.evaluate() });
		return this.metrics!;
	}
	pause() {
		this.stopped = true;
	}
	private async captured(
		window: ArrayLike<number>,
		neuron?: number | null
	): Promise<{ logprobs: Float32Array; layers: Float32Array[] }> {
		this.requireReady();
		const mask = storyMask(this.config, neuron);
		const batch = storyBatch(this.config, [window]);
		batch.targets.dispose();
		const arrays = this.capture(
			tree.ref(this.params),
			batch.tokens,
			batch.positions,
			mask
		) as any[];
		const [logprobs, ...layers] = await Promise.all(
			arrays.map(async (array) => (await array.data()) as Float32Array)
		);
		this.validateProbabilities(logprobs);
		if (layers.some((layer) => layer.some((value) => !Number.isFinite(value) || value < 0)))
			throw new Error('Invalid post-ReLU story activation');
		return { logprobs, layers };
	}
	async captureAtlas(emit: Emit = () => {}): Promise<StoryAtlas> {
		this.requireReady();
		this.stopped = false;
		const started = performance.now(),
			c = this.config,
			unitCount = storyUnitCount(c);
		const positions = Array.from({ length: 16 }, (_, i) => Math.round((i * (c.context - 1)) / 15));
		const examples = this.corpus.calibration,
			dimensions = examples.length * positions.length;
		const fingerprints = new Float32Array(unitCount * dimensions);
		for (let i = 0; i < examples.length; i++) {
			if (this.stopped) throw new Error('Story atlas measurement cancelled');
			const captured = await this.captured(examples[i]);
			for (let unit = 0; unit < unitCount; unit++)
				for (let p = 0; p < positions.length; p++)
					fingerprints[unit * dimensions + i * positions.length + p] =
						captured.layers[Math.floor(unit / c.hidden)][
							positions[p] * c.hidden + (unit % c.hidden)
						];
			emit({ type: 'measurement', completed: i + 1, total: examples.length, step: this.step });
			await yieldTask();
		}
		if (this.stopped) throw new Error('Story atlas measurement cancelled');
		return {
			version: 1,
			modelId: this.modelId,
			config: { ...c },
			seed: this.seed,
			step: this.step,
			backend: this.backend,
			capturedAt: new Date().toISOString(),
			elapsedMs: performance.now() - started,
			corpusId: this.corpus.info.id,
			unitCount,
			dimensions,
			fingerprints,
			positions,
			examples: examples.map((window) => decodeStory(window.subarray(0, c.context)))
		};
	}
	async probe(text: string, neuron?: number | null): Promise<StoryProbe> {
		this.requireReady();
		const c = this.config,
			prompt = encodeStoryPrompt(text, c.context);
		const { logprobs, layers } = await this.captured(prompt.tokenIds, neuron),
			unitCount = storyUnitCount(c);
		const activations = new Float32Array(prompt.tokenIds.length * unitCount);
		for (let token = 0; token < prompt.tokenIds.length; token++)
			for (let unit = 0; unit < unitCount; unit++)
				activations[token * unitCount + unit] =
					layers[Math.floor(unit / c.hidden)][token * c.hidden + (unit % c.hidden)];
		const row = logprobs.subarray((prompt.tokenIds.length - 1) * 96, prompt.tokenIds.length * 96);
		return {
			modelId: this.modelId,
			config: { ...c },
			seed: this.seed,
			step: this.step,
			prompt,
			activations,
			unitCount,
			probabilities: Float32Array.from(row, Math.exp),
			predictedToken: argmax(row),
			lesionNeuron: neuron ?? null
		};
	}
	async generate(text: string, options: StoryGenerationOptions = {}): Promise<StoryGeneration> {
		this.requireReady();
		const prompt = encodeStoryPrompt(text, this.config.context),
			maxTokens = options.maxTokens ?? 96,
			temperature = options.temperature ?? 0.8,
			topK = options.topK ?? 20,
			samplingSeed = options.seed ?? 71;
		if (
			!Number.isInteger(maxTokens) ||
			maxTokens < 1 ||
			maxTokens > 128 ||
			!Number.isFinite(temperature) ||
			temperature < 0.05 ||
			temperature > 2 ||
			!Number.isInteger(topK) ||
			topK < 1 ||
			topK > 96 ||
			!Number.isInteger(samplingSeed) ||
			samplingSeed < 0 ||
			samplingSeed > 0xffffffff
		)
			throw new Error(
				'Invalid generation settings (1–128 tokens, temperature 0.05–2, top-k 1–96, unsigned seed)'
			);
		this.stopped = false;
		const random = new StoryRandom(samplingSeed),
			tokenIds: number[] = [];
		let context = [...prompt.tokenIds];
		for (let i = 0; i < maxTokens && !this.stopped; i++) {
			const lp = await this.predict([context]),
				offset = (context.length - 1) * 96;
			const ranked = Array.from({ length: 96 }, (_, id) => ({ id, logp: lp[offset + id] }))
				.sort((a, b) => b.logp - a.logp || a.id - b.id)
				.slice(0, topK);
			const mass = ranked.map((x) => Math.exp((x.logp - ranked[0].logp) / temperature));
			let draw = random.next() * mass.reduce((a, b) => a + b, 0),
				id = ranked.at(-1)!.id;
			for (let j = 0; j < ranked.length; j++) {
				draw -= mass[j];
				if (draw <= 0) {
					id = ranked[j].id;
					break;
				}
			}
			tokenIds.push(id);
			context = [...context, id].slice(-this.config.context);
			await yieldTask();
		}
		return {
			modelId: this.modelId,
			step: this.step,
			prompt,
			completion: decodeStory(tokenIds),
			tokenIds,
			samplingSeed,
			temperature,
			topK,
			requestedTokens: maxTokens,
			cancelled: tokenIds.length < maxTokens
		};
	}
	async exportCheckpoint(): Promise<StoryCheckpoint> {
		this.requireReady();
		return {
			version: 1,
			architecture: 'stories-transformer-v1',
			modelId: this.modelId,
			config: { ...this.config },
			seed: this.seed,
			step: this.step,
			elapsedMs: this.elapsedMs,
			trainLoss: this.trainLoss,
			trainRngState: this.random.state,
			corpus: this.corpus.info,
			parameters: await serializeStoryTree(this.params!),
			optimizer: {
				m: await serializeStoryTree(this.state!.m),
				v: await serializeStoryTree(this.state!.v),
				t: this.state!.t
			}
		};
	}
	async loadCheckpoint(
		checkpoint: StoryCheckpoint,
		preference: StoryBackend | 'auto' = 'auto',
		emit: Emit = () => {}
	): Promise<StoryInitialization> {
		validateStoryCheckpoint(checkpoint);
		if (checkpoint.corpus.id !== STORY_CORPUS_ID)
			throw new Error('Checkpoint uses a different corpus');
		if (
			!this.params ||
			JSON.stringify(this.config) !== JSON.stringify(checkpoint.config) ||
			(preference !== 'auto' && preference !== this.backend)
		)
			return this.initialize(
				checkpoint.config,
				checkpoint.seed,
				preference,
				undefined,
				emit,
				checkpoint
			);
		const parameters = deserializeStoryTree(this.config, checkpoint.parameters),
			m = deserializeStoryTree(this.config, checkpoint.optimizer.m),
			v = deserializeStoryTree(this.config, checkpoint.optimizer.v);
		tree.dispose(this.params);
		tree.dispose(this.state!.m);
		tree.dispose(this.state!.v);
		this.params = parameters;
		this.state = { m, v, t: checkpoint.step };
		this.restoreMetadata(checkpoint);
		await this.evaluate();
		return this.initialization();
	}
	private restoreMetadata(checkpoint: StoryCheckpoint) {
		this.modelId = checkpoint.modelId;
		this.seed = checkpoint.seed;
		this.step = checkpoint.step;
		this.elapsedMs = checkpoint.elapsedMs;
		this.trainLoss = checkpoint.trainLoss;
		this.stepMs = 0;
		this.random = new StoryRandom(checkpoint.trainRngState);
		this.stopped = false;
	}
	dispose() {
		this.stopped = true;
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
		this.metrics = null;
	}
}
