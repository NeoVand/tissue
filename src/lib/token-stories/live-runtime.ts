/* eslint-disable @typescript-eslint/no-explicit-any */
import { init, defaultDevice, numpy as np, jit, tree } from '@jax-js/jax';
import type { LiveTokenStoryFrame, LiveTokenStoryFrameCallback } from './live-protocol';
import { fusedAdam, type AdamState, type FusedAdam } from '../lab/model/adam';
import {
	TOKEN_STORY_PRESETS,
	tokenStoryParameterCount,
	tokenStoryUnitCount,
	tokenStoryUnitAddress,
	validateTokenStoryConfig,
	validateTokenStoryCheckpoint,
	type TokenStoryAtlas,
	type TokenStoryBackend,
	type TokenStoryCheckpoint,
	type TokenStoryConfig,
	type TokenStoryEvent,
	type TokenStoryGeneration,
	type TokenStoryGenerationOptions,
	type TokenStoryInitialization,
	type TokenStoryMetrics,
	type TokenStoryPreset,
	type TokenStoryProbe
} from './protocol';
import {
	TOKEN_STORY_CORPUS_ID,
	TokenStoryRandom,
	TOKEN_STORY_TOKENIZER_ID,
	sampleTokenStoryTrainingWindow,
	encodeTokenStoryPrompt,
	loadTokenStoryCorpus,
	type TokenStoryCorpus
} from './dataset';
import {
	initializeTokenStoryParameters,
	tokenStoryForward,
	tokenStoryLoss,
	tokenStoryBatch,
	tokenStoryMask,
	serializeTokenStoryTree,
	deserializeTokenStoryTree,
	type TokenStoryParams
} from './model';

const yieldTask = () => new Promise<void>((resolve) => setTimeout(resolve, 0));
const argmax = (row: ArrayLike<number>) => {
	let best = 0;
	for (let i = 1; i < row.length; i++) if (row[i] > row[best]) best = i;
	return best;
};
type Emit = (event: TokenStoryEvent) => void;

/** Sole owner of GPU model state. The worker serializes every operation except pause. */
export class LiveTokenStoryRuntime {
	private params: TokenStoryParams | null = null;
	private state: AdamState | null = null;
	private optimizer: FusedAdam<any> | null = null;
	private inference: any = null;
	private capture: any = null;
	private config!: TokenStoryConfig;
	private corpus!: TokenStoryCorpus;
	private seed = 42;
	private modelId = '';
	private step = 0;
	private trainedTokens = 0;
	private elapsedMs = 0;
	private stepMs = 0;
	private trainLoss: number | null = null;
	private random = new TokenStoryRandom(42);
	private backend: TokenStoryBackend = 'cpu';
	private stopped = false;
	private cancelLiveWait: (() => void) | null = null;
	private metrics: TokenStoryMetrics | null = null;
	async initialize(
		preset: TokenStoryPreset | TokenStoryConfig,
		seed = 42,
		preference: TokenStoryBackend | 'auto' = 'auto',
		corpus?: TokenStoryCorpus,
		emit: Emit = () => {},
		checkpoint?: TokenStoryCheckpoint
	): Promise<TokenStoryInitialization> {
		const config = typeof preset === 'string' ? TOKEN_STORY_PRESETS[preset] : preset;
		validateTokenStoryConfig(config);
		if (!['auto', 'webgpu', 'wasm', 'cpu'].includes(preference))
			throw new Error('Unknown BPE story backend');
		if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff)
			throw new Error('Seed must be an unsigned 32-bit integer');
		if (checkpoint) validateTokenStoryCheckpoint(checkpoint);
		emit({ type: 'status', message: 'Loading and verifying the TinyStories corpus…' });
		const data = corpus ?? (await loadTokenStoryCorpus(config));
		if (data.info.id !== TOKEN_STORY_CORPUS_ID) throw new Error('Unknown TinyStories corpus');
		this.dispose();
		const candidates: TokenStoryBackend[] =
			preference === 'auto'
				? tokenStoryParameterCount(config) <= tokenStoryParameterCount(TOKEN_STORY_PRESETS.small)
					? ['webgpu', 'wasm']
					: ['webgpu']
				: [preference];
		let selected: TokenStoryBackend | undefined;
		for (const backend of candidates) {
			emit({
				type: 'status',
				message: `Initializing ${backend.toUpperCase()} for ${tokenStoryParameterCount(config).toLocaleString()} parameters…`
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
		this.trainedTokens = 0;
		this.elapsedMs = 0;
		this.stepMs = 0;
		this.trainLoss = null;
		this.random = new TokenStoryRandom(seed ^ 0x7374726e);
		this.stopped = false;
		this.params = checkpoint
			? deserializeTokenStoryTree(config, checkpoint.parameters)
			: initializeTokenStoryParameters(config, seed);
		this.optimizer = fusedAdam(
			(p, t, pos, y, lossMask, mask) => tokenStoryLoss(p, config, t, pos, y, lossMask, mask),
			{
				lr: config.learningRate,
				b1: 0.9,
				b2: 0.99
			}
		);
		this.state = checkpoint
			? {
					m: deserializeTokenStoryTree(config, checkpoint.optimizer.m),
					v: deserializeTokenStoryTree(config, checkpoint.optimizer.v),
					t: checkpoint.step
				}
			: this.optimizer.init(this.params);
		if (checkpoint) this.restoreMetadata(checkpoint);
		this.inference = jit(
			(p: TokenStoryParams, t: any, pos: any, mask: any) =>
				tokenStoryForward(p, config, t, pos, mask).logprobs
		);
		this.capture = jit((p: TokenStoryParams, t: any, pos: any, mask: any) => {
			const output = tokenStoryForward(p, config, t, pos, mask, true);
			return [output.logprobs, ...output.activations];
		});
		emit({ type: 'status', message: 'Compiling fixed held-out next-token evaluation…' });
		await this.evaluate();
		return this.initialization();
	}
	private initialization(): TokenStoryInitialization {
		return {
			modelId: this.modelId,
			config: { ...this.config },
			seed: this.seed,
			backend: this.backend,
			parameterCount: tokenStoryParameterCount(this.config),
			unitCount: tokenStoryUnitCount(this.config),
			corpus: this.corpus.info,
			tokenizer: this.corpus.tokenizerData,
			metrics: this.metrics!
		};
	}
	private requireReady() {
		if (!this.params || !this.state) throw new Error('Initialize a BPE story model first');
	}
	private validateProbabilities(flat: Float32Array) {
		for (let offset = 0; offset < flat.length; offset += this.config.vocabularySize) {
			let mass = 0;
			for (let j = 0; j < this.config.vocabularySize; j++) {
				const value = flat[offset + j];
				if (!Number.isFinite(value)) throw new Error('Nonfinite BPE story model probabilities');
				mass += Math.exp(value);
			}
			if (Math.abs(mass - 1) > 1e-3)
				throw new Error(`BPE story probability mass is invalid (${mass})`);
		}
	}
	private async predict(
		windows: ArrayLike<number>[],
		neuron?: number | null
	): Promise<Float32Array> {
		this.requireReady();
		const mask = tokenStoryMask(this.config, neuron);
		const batch = tokenStoryBatch(this.config, windows);
		batch.targets.dispose();
		batch.lossMask.dispose();
		const flat = (await this.inference(
			tree.ref(this.params),
			batch.tokens,
			batch.positions,
			mask
		).data()) as Float32Array;
		this.validateProbabilities(flat);
		return flat;
	}
	async evaluate(): Promise<TokenStoryMetrics> {
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
				for (let t = 0; t < Math.min(c.context, window.length - 1); t++) {
					const target = window[t + 1],
						offset = (b * c.context + t) * c.vocabularySize;
					loss -= lp[offset + target];
					correct += +(argmax(lp.subarray(offset, offset + c.vocabularySize)) === target);
					unigramLoss -= Math.log(this.corpus.unigram[target]);
					unigramCorrect += +(unigramBest === target);
				}
			});
		}
		const count = examples.reduce(
			(total, window) => total + Math.min(c.context, window.length - 1),
			0
		);
		this.metrics = {
			step: this.step,
			trainLoss: this.trainLoss,
			validationLoss: loss / count,
			validationAccuracy: correct / count,
			unigramLoss: unigramLoss / count,
			unigramAccuracy: unigramCorrect / count,
			uniformLoss: Math.log(c.vocabularySize),
			evaluationTokens: count,
			trainedTokens: this.trainedTokens,
			elapsedMs: this.elapsedMs,
			stepMs: this.stepMs,
			backend: this.backend
		};
		return this.metrics;
	}
	async train(steps: number, emit: Emit = () => {}): Promise<TokenStoryMetrics> {
		this.requireReady();
		if (!Number.isInteger(steps) || steps < 1 || steps > 100)
			throw new Error('A BPE story training burst must contain 1 to 100 updates');
		this.stopped = false;
		const c = this.config;
		emit({
			type: 'status',
			message:
				this.step === 0
					? 'Compiling BPE story gradients and Adam…'
					: 'Training next-token prediction…'
		});
		for (let i = 0; i < steps && !this.stopped; i++) {
			const started = performance.now();
			const windows = Array.from({ length: c.batchSize }, () =>
				sampleTokenStoryTrainingWindow(this.corpus, this.random)
			);
			const batch = tokenStoryBatch(c, windows);
			const [loss, parameters, state] = this.optimizer!.step(
				this.params,
				this.state!,
				batch.tokens,
				batch.positions,
				batch.targets,
				batch.lossMask,
				tokenStoryMask(c)
			);
			this.params = parameters;
			this.state = state;
			this.trainLoss = Number((await loss.data())[0]);
			this.step++;
			this.trainedTokens += windows.reduce((total, window) => total + window.length - 1, 0);
			this.stepMs = performance.now() - started;
			this.elapsedMs += this.stepMs;
			if (!Number.isFinite(this.trainLoss) || this.trainLoss <= 0)
				throw new Error(`Invalid BPE story training loss at step ${this.step}`);
			if (this.step % 25 === 0 || i === steps - 1)
				emit({ type: 'metrics', metrics: await this.evaluate() });
			await yieldTask();
		}
		if (this.metrics?.step !== this.step) emit({ type: 'metrics', metrics: await this.evaluate() });
		return this.metrics!;
	}
	pause() {
		this.stopped = true;
		this.cancelLiveWait?.();
	}
	private async captured(
		window: ArrayLike<number>,
		neuron?: number | null
	): Promise<{ logprobs: Float32Array; layers: Float32Array[] }> {
		this.requireReady();
		const mask = tokenStoryMask(this.config, neuron);
		const batch = tokenStoryBatch(this.config, [window]);
		batch.targets.dispose();
		batch.lossMask.dispose();
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
			throw new Error('Invalid post-ReLU BPE story activation');
		return { logprobs, layers };
	}
	async captureAtlas(emit: Emit = () => {}): Promise<TokenStoryAtlas> {
		this.requireReady();
		this.stopped = false;
		const started = performance.now(),
			c = this.config,
			unitCount = tokenStoryUnitCount(c);
		const positions = Array.from({ length: 16 }, (_, i) => Math.round((i * (c.context - 1)) / 15));
		const examples = this.corpus.calibration,
			dimensions = examples.length * positions.length;
		const tokenIds = examples.map((window) =>
			Array.from(window.subarray(0, Math.min(c.context, window.length - 1)))
		);
		const tokenPositions = tokenIds.map((ids) =>
			Array.from({ length: 16 }, (_, p) => Math.round((p * (ids.length - 1)) / 15))
		);
		const fingerprints = new Float32Array(unitCount * dimensions);
		for (let i = 0; i < examples.length; i++) {
			if (this.stopped) throw new Error('TokenStory atlas measurement cancelled');
			const captured = await this.captured(examples[i]);
			for (let unit = 0; unit < unitCount; unit++)
				for (let p = 0; p < positions.length; p++)
					fingerprints[unit * dimensions + i * positions.length + p] =
						captured.layers[Math.floor(unit / c.hidden)][
							tokenPositions[i][p] * c.hidden + (unit % c.hidden)
						];
			emit({ type: 'measurement', completed: i + 1, total: examples.length, step: this.step });
			await yieldTask();
		}
		if (this.stopped) throw new Error('TokenStory atlas measurement cancelled');
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
			tokenizerId: this.corpus.tokenizer.id,
			tokenPositions,
			tokenIds,
			unitCount,
			dimensions,
			fingerprints,
			positions,
			examples: tokenIds.map((ids) => this.corpus.tokenizer.decode(ids))
		};
	}
	async probe(text: string, neuron?: number | null): Promise<TokenStoryProbe> {
		this.requireReady();
		const c = this.config,
			prompt = encodeTokenStoryPrompt(text, c.context, this.corpus.tokenizer);
		const { logprobs, layers } = await this.captured(prompt.tokenIds, neuron),
			unitCount = tokenStoryUnitCount(c);
		const activations = new Float32Array(prompt.tokenIds.length * unitCount);
		for (let token = 0; token < prompt.tokenIds.length; token++)
			for (let unit = 0; unit < unitCount; unit++)
				activations[token * unitCount + unit] =
					layers[Math.floor(unit / c.hidden)][token * c.hidden + (unit % c.hidden)];
		const row = logprobs.subarray(
			(prompt.tokenIds.length - 1) * c.vocabularySize,
			prompt.tokenIds.length * c.vocabularySize
		);
		return {
			modelId: this.modelId,
			config: { ...c },
			seed: this.seed,
			step: this.step,
			prompt,
			tokenizerId: this.corpus.tokenizer.id,
			corpusId: this.corpus.info.id,
			unitAddresses: Array.from({ length: unitCount }, (_, id) => tokenStoryUnitAddress(c, id)),
			activations,
			unitCount,
			probabilities: Float32Array.from(row, Math.exp),
			predictedToken: argmax(row),
			lesionNeuron: neuron ?? null
		};
	}
	async generate(
		text: string,
		options: TokenStoryGenerationOptions = {}
	): Promise<TokenStoryGeneration> {
		this.requireReady();
		const prompt = encodeTokenStoryPrompt(text, this.config.context, this.corpus.tokenizer),
			maxTokens = options.maxTokens ?? 128,
			temperature = options.temperature ?? 0.8,
			topK = options.topK ?? 40,
			samplingSeed = options.seed ?? 71;
		if (
			!Number.isInteger(maxTokens) ||
			maxTokens < 1 ||
			maxTokens > 256 ||
			!Number.isFinite(temperature) ||
			temperature < 0.05 ||
			temperature > 2 ||
			!Number.isInteger(topK) ||
			topK < 1 ||
			topK > this.config.vocabularySize - 1 ||
			!Number.isInteger(samplingSeed) ||
			samplingSeed < 0 ||
			samplingSeed > 0xffffffff
		)
			throw new Error(
				'Invalid generation settings (1–256 tokens, temperature 0.05–2, top-k 1–(vocabulary−1), unsigned seed)'
			);
		this.stopped = false;
		const random = new TokenStoryRandom(samplingSeed),
			tokenIds: number[] = [];
		let context = [...prompt.tokenIds],
			stoppedOnEos = false;
		for (let i = 0; i < maxTokens && !this.stopped; i++) {
			const lp = await this.predict([context]),
				offset = (context.length - 1) * this.config.vocabularySize;
			const ranked = Array.from({ length: this.config.vocabularySize }, (_, id) => ({
				id,
				logp: lp[offset + id]
			}))
				.filter((x) => x.id !== this.corpus.tokenizer.bosId)
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
			if (id === this.corpus.tokenizer.eosId) {
				stoppedOnEos = true;
				break;
			}
			context = [...context, id].slice(-this.config.context);
			await yieldTask();
		}
		return {
			modelId: this.modelId,
			step: this.step,
			prompt,
			completion: this.corpus.tokenizer.decode(tokenIds),
			pieces: tokenIds.map((id) => this.corpus.tokenizer.tokenPiece(id)),
			tokenizerId: this.corpus.tokenizer.id,
			stoppedOnEos,
			tokenIds,
			samplingSeed,
			temperature,
			topK,
			requestedTokens: maxTokens,
			cancelled: !stoppedOnEos && tokenIds.length < maxTokens
		};
	}
	/** A user may pause playback indefinitely; only computation is subject to the client watchdog. */
	private acknowledgeFrame(
		frame: LiveTokenStoryFrame,
		onFrame: LiveTokenStoryFrameCallback
	): Promise<boolean> {
		return new Promise((resolve, reject) => {
			let settled = false;
			const finish = (accepted: boolean, error?: unknown) => {
				if (settled) return;
				settled = true;
				this.cancelLiveWait = null;
				if (error !== undefined) reject(error);
				else resolve(accepted);
			};
			this.cancelLiveWait = () => finish(false);
			Promise.resolve()
				.then(() => onFrame(frame))
				.then(
					() => finish(!this.stopped),
					(error) => finish(false, error)
				);
		});
	}
	async generateLive(
		text: string,
		options: TokenStoryGenerationOptions,
		onFrame: LiveTokenStoryFrameCallback,
		initiallyCancelled = false
	): Promise<TokenStoryGeneration> {
		this.requireReady();
		const c = this.config,
			prompt = encodeTokenStoryPrompt(text, c.context, this.corpus.tokenizer),
			maxTokens = options.maxTokens ?? 128,
			temperature = options.temperature ?? 0.8,
			topK = options.topK ?? 40,
			samplingSeed = options.seed ?? 71;
		if (
			!Number.isInteger(maxTokens) ||
			maxTokens < 1 ||
			maxTokens > 256 ||
			!Number.isFinite(temperature) ||
			temperature < 0.05 ||
			temperature > 2 ||
			!Number.isInteger(topK) ||
			topK < 1 ||
			topK > c.vocabularySize - 1 ||
			!Number.isInteger(samplingSeed) ||
			samplingSeed < 0 ||
			samplingSeed > 0xffffffff ||
			typeof onFrame !== 'function'
		)
			throw new Error('Invalid live generation settings');
		this.stopped = initiallyCancelled;
		const random = new TokenStoryRandom(samplingSeed),
			tokenIds: number[] = [],
			unitCount = tokenStoryUnitCount(c);
		let context = [...prompt.tokenIds],
			stoppedOnEos = false;
		for (let index = 0; index < maxTokens && !this.stopped; index++) {
			// The sampled ID stream is the input. Decoding and re-encoding can change BPE boundaries.
			const { logprobs, layers } = await this.captured(context);
			if (this.stopped) break;
			const position = context.length - 1,
				row = logprobs.subarray(position * c.vocabularySize, (position + 1) * c.vocabularySize);
			const ranked = Array.from(row, (logp, id) => ({ id, logp }))
				.filter((x) => x.id !== this.corpus.tokenizer.bosId)
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
			const activations = new Float32Array(unitCount);
			for (let unit = 0; unit < unitCount; unit++)
				activations[unit] =
					layers[Math.floor(unit / c.hidden)][position * c.hidden + (unit % c.hidden)];
			const frame: LiveTokenStoryFrame = {
				version: 1,
				modelId: this.modelId,
				step: this.step,
				trainedTokens: this.trainedTokens,
				seed: this.seed,
				backend: this.backend,
				config: { ...c },
				tokenizerId: this.corpus.tokenizer.id,
				corpusId: this.corpus.info.id,
				index,
				prompt: text,
				context: {
					tokenIds: [...context],
					pieces: context.map((id) => this.corpus.tokenizer.tokenPiece(id)),
					text: this.corpus.tokenizer.decode(context),
					truncatedTokens:
						prompt.truncatedTokens + Math.max(0, prompt.tokenIds.length + index - c.context),
					includesBos: context[0] === this.corpus.tokenizer.bosId
				},
				position,
				unitCount,
				activations,
				probabilities: Float32Array.from(row, Math.exp),
				sampledToken: id,
				sampledPiece: this.corpus.tokenizer.tokenPiece(id),
				isEos: id === this.corpus.tokenizer.eosId
			};
			if (!(await this.acknowledgeFrame(frame, onFrame)) || this.stopped) break;
			tokenIds.push(id);
			if (id === this.corpus.tokenizer.eosId) {
				stoppedOnEos = true;
				break;
			}
			context = [...context, id].slice(-c.context);
			await yieldTask();
		}
		return {
			modelId: this.modelId,
			step: this.step,
			prompt,
			completion: this.corpus.tokenizer.decode(tokenIds),
			pieces: tokenIds.map((id) => this.corpus.tokenizer.tokenPiece(id)),
			tokenizerId: this.corpus.tokenizer.id,
			stoppedOnEos,
			tokenIds,
			samplingSeed,
			temperature,
			topK,
			requestedTokens: maxTokens,
			cancelled: !stoppedOnEos && tokenIds.length < maxTokens
		};
	}
	async exportCheckpoint(): Promise<TokenStoryCheckpoint> {
		this.requireReady();
		return {
			version: 1,
			architecture: 'token-stories-transformer-v1',
			modelId: this.modelId,
			config: { ...this.config },
			seed: this.seed,
			step: this.step,
			elapsedMs: this.elapsedMs,
			trainLoss: this.trainLoss,
			trainRngState: this.random.state,
			corpusId: this.corpus.info.id,
			tokenizerId: this.corpus.tokenizer.id,
			trainedTokens: this.trainedTokens,
			parameters: await serializeTokenStoryTree(this.params!),
			optimizer: {
				m: await serializeTokenStoryTree(this.state!.m),
				v: await serializeTokenStoryTree(this.state!.v),
				t: this.state!.t
			}
		};
	}
	async loadCheckpoint(
		checkpoint: TokenStoryCheckpoint,
		preference: TokenStoryBackend | 'auto' = 'auto',
		emit: Emit = () => {}
	): Promise<TokenStoryInitialization> {
		validateTokenStoryCheckpoint(checkpoint);
		if (
			checkpoint.corpusId !== TOKEN_STORY_CORPUS_ID ||
			checkpoint.tokenizerId !== TOKEN_STORY_TOKENIZER_ID
		)
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
		const parameters = deserializeTokenStoryTree(this.config, checkpoint.parameters),
			m = deserializeTokenStoryTree(this.config, checkpoint.optimizer.m),
			v = deserializeTokenStoryTree(this.config, checkpoint.optimizer.v);
		tree.dispose(this.params);
		tree.dispose(this.state!.m);
		tree.dispose(this.state!.v);
		this.params = parameters;
		this.state = { m, v, t: checkpoint.step };
		this.restoreMetadata(checkpoint);
		await this.evaluate();
		return this.initialization();
	}
	private restoreMetadata(checkpoint: TokenStoryCheckpoint) {
		this.modelId = checkpoint.modelId;
		this.seed = checkpoint.seed;
		this.step = checkpoint.step;
		this.trainedTokens = checkpoint.trainedTokens;
		this.elapsedMs = checkpoint.elapsedMs;
		this.trainLoss = checkpoint.trainLoss;
		this.stepMs = 0;
		this.random = new TokenStoryRandom(checkpoint.trainRngState);
		this.stopped = false;
	}
	dispose() {
		this.pause();
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
