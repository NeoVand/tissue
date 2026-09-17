import type { TokenStoryTokenizerData, TokenStoryCorpusMetadata } from './tokenizer';
export type TokenStoryBackend = 'webgpu' | 'wasm' | 'cpu';
export type TokenStoryPreset = 'small' | 'medium' | 'large';
export interface TokenStoryConfig {
	layers: number;
	width: number;
	heads: number;
	hidden: number;
	context: number;
	batchSize: number;
	vocabularySize: number;
	learningRate: number;
}
export const TOKEN_STORY_PRESETS: Record<TokenStoryPreset, TokenStoryConfig> = {
	small: {
		layers: 4,
		width: 128,
		heads: 4,
		hidden: 512,
		context: 128,
		batchSize: 4,
		vocabularySize: 4096,
		learningRate: 0.001
	},
	medium: {
		layers: 4,
		width: 256,
		heads: 8,
		hidden: 1024,
		context: 256,
		batchSize: 2,
		vocabularySize: 4096,
		learningRate: 0.0006
	},
	large: {
		layers: 6,
		width: 384,
		heads: 8,
		hidden: 1536,
		context: 256,
		batchSize: 1,
		vocabularySize: 4096,
		learningRate: 0.0004
	}
};
export const tokenStoryUnitCount = (c: TokenStoryConfig) => c.layers * c.hidden;
export const tokenStoryParameterCount = (c: TokenStoryConfig) =>
	2 * c.vocabularySize * c.width +
	c.context * c.width +
	c.layers * (4 * c.width * c.width + 2 * c.width * c.hidden);
export interface TokenStoryMetrics {
	step: number;
	trainLoss: number | null;
	validationLoss: number;
	validationAccuracy: number;
	unigramLoss: number;
	unigramAccuracy: number;
	uniformLoss: number;
	evaluationTokens: number;
	trainedTokens: number;
	elapsedMs: number;
	stepMs: number;
	backend: TokenStoryBackend;
}
export interface TokenStoryInitialization {
	modelId: string;
	config: TokenStoryConfig;
	seed: number;
	backend: TokenStoryBackend;
	parameterCount: number;
	unitCount: number;
	corpus: TokenStoryCorpusMetadata;
	tokenizer: TokenStoryTokenizerData;
	metrics: TokenStoryMetrics;
}
export interface TokenStoryEncodedPrompt {
	original: string;
	text: string;
	tokenIds: number[];
	pieces: string[];
	truncatedTokens: number;
	unsupportedCharacters: string[];
	includesBos: boolean;
}
export interface TokenStoryUnitAddress {
	id: number;
	layer: number;
	channel: number;
	incomingTensor: string;
	incomingColumn: number;
	outgoingTensor: string;
	outgoingRow: number;
}
export function tokenStoryUnitAddress(c: TokenStoryConfig, id: number): TokenStoryUnitAddress {
	if (!Number.isInteger(id) || id < 0 || id >= tokenStoryUnitCount(c))
		throw new Error('Token-story neuron is outside the model');
	const layer = Math.floor(id / c.hidden),
		channel = id % c.hidden;
	return {
		id,
		layer,
		channel,
		incomingTensor: `layers.${layer}.mlpFc1`,
		incomingColumn: channel,
		outgoingTensor: `layers.${layer}.mlpFc2`,
		outgoingRow: channel
	};
}
export interface TokenStoryProbe {
	modelId: string;
	config: TokenStoryConfig;
	seed: number;
	step: number;
	tokenizerId: string;
	corpusId: string;
	prompt: TokenStoryEncodedPrompt;
	activations: Float32Array;
	unitCount: number;
	unitAddresses: TokenStoryUnitAddress[];
	probabilities: Float32Array;
	predictedToken: number;
	lesionNeuron: number | null;
}
export interface TokenStoryAtlas {
	version: 1;
	modelId: string;
	config: TokenStoryConfig;
	seed: number;
	step: number;
	backend: TokenStoryBackend;
	capturedAt: string;
	elapsedMs: number;
	corpusId: string;
	tokenizerId: string;
	unitCount: number;
	dimensions: number;
	fingerprints: Float32Array;
	/** Canonical sixteen knots; actual token locations are in tokenPositions. */ positions: number[];
	tokenPositions: number[][];
	tokenIds: number[][];
	examples: string[];
}
export interface TokenStoryGenerationOptions {
	maxTokens?: number;
	temperature?: number;
	topK?: number;
	seed?: number;
}
export interface TokenStoryGeneration {
	modelId: string;
	step: number;
	tokenizerId: string;
	prompt: TokenStoryEncodedPrompt;
	completion: string;
	tokenIds: number[];
	pieces: string[];
	samplingSeed: number;
	temperature: number;
	topK: number;
	requestedTokens: number;
	cancelled: boolean;
	stoppedOnEos: boolean;
}
export interface TokenStoryTensor {
	shape: number[];
	values: Float32Array;
}
export interface TokenStoryCheckpoint {
	version: 1;
	architecture: 'token-stories-transformer-v1';
	modelId: string;
	config: TokenStoryConfig;
	seed: number;
	step: number;
	trainedTokens: number;
	elapsedMs: number;
	trainLoss: number | null;
	trainRngState: number;
	corpusId: string;
	tokenizerId: string;
	parameters: TokenStoryTensor[];
	optimizer: { m: TokenStoryTensor[]; v: TokenStoryTensor[]; t: number };
}
export type TokenStoryEvent =
	| { type: 'status'; message: string }
	| { type: 'metrics'; metrics: TokenStoryMetrics }
	| { type: 'measurement'; completed: number; total: number; step: number };
export function validateTokenStoryConfig(value: unknown): asserts value is TokenStoryConfig {
	const c = value as TokenStoryConfig;
	if (
		!c ||
		!Number.isInteger(c.layers) ||
		c.layers < 1 ||
		c.layers > 6 ||
		!Number.isInteger(c.width) ||
		c.width < 8 ||
		c.width > 384 ||
		!Number.isInteger(c.heads) ||
		c.heads < 1 ||
		c.heads > 8 ||
		c.width % c.heads !== 0 ||
		c.hidden !== c.width * 4 ||
		!Number.isInteger(c.context) ||
		c.context < 8 ||
		c.context > 256 ||
		!Number.isInteger(c.batchSize) ||
		c.batchSize < 1 ||
		c.batchSize > 4 ||
		!Number.isInteger(c.vocabularySize) ||
		c.vocabularySize < 98 ||
		c.vocabularySize > 4096 ||
		!Number.isFinite(c.learningRate) ||
		c.learningRate < 1e-5 ||
		c.learningRate > 0.01
	)
		throw new Error('Unsupported token-story model configuration');
}
export function tokenStoryParameterShapes(c: TokenStoryConfig): number[][] {
	validateTokenStoryConfig(c);
	return [
		[c.vocabularySize, c.width],
		[c.context, c.width],
		[c.width, c.vocabularySize],
		...Array.from({ length: c.layers }, () => [
			...Array.from({ length: 4 }, () => [c.width, c.width]),
			[c.width, c.hidden],
			[c.hidden, c.width]
		]).flat()
	];
}
export function validateTokenStoryCheckpoint(
	value: unknown
): asserts value is TokenStoryCheckpoint {
	const c = value as TokenStoryCheckpoint;
	if (!c || c.version !== 1 || c.architecture !== 'token-stories-transformer-v1')
		throw new Error('Unsupported token-story checkpoint');
	validateTokenStoryConfig(c.config);
	if (
		typeof c.modelId !== 'string' ||
		!c.modelId.length ||
		c.modelId.length > 200 ||
		!Number.isInteger(c.seed) ||
		c.seed < 0 ||
		c.seed > 0xffffffff ||
		!Number.isSafeInteger(c.step) ||
		c.step < 0 ||
		!Number.isSafeInteger(c.trainedTokens) ||
		c.trainedTokens < c.step * c.config.batchSize ||
		c.trainedTokens > c.step * c.config.batchSize * c.config.context ||
		!Number.isInteger(c.trainRngState) ||
		c.trainRngState < 0 ||
		c.trainRngState > 0xffffffff ||
		!Number.isFinite(c.elapsedMs) ||
		c.elapsedMs < 0 ||
		(c.trainLoss !== null && (!Number.isFinite(c.trainLoss) || c.trainLoss < 0)) ||
		!/^tinystories-bpe-corpus-v1:[a-f0-9]{64}$/.test(c.corpusId) ||
		!/^tinystories-bpe-v1:[a-f0-9]{64}$/.test(c.tokenizerId) ||
		c.optimizer?.t !== c.step
	)
		throw new Error('Invalid token-story checkpoint metadata');
	const shapes = tokenStoryParameterShapes(c.config);
	for (const [label, leaves] of [
		['parameters', c.parameters],
		['m', c.optimizer.m],
		['v', c.optimizer.v]
	] as const) {
		if (!Array.isArray(leaves) || leaves.length !== shapes.length)
			throw new Error(`Invalid token-story ${label} tensor count`);
		leaves.forEach((leaf, i) => {
			if (
				!leaf ||
				!Array.isArray(leaf.shape) ||
				leaf.shape.length !== 2 ||
				leaf.shape.some((d, j) => d !== shapes[i][j]) ||
				!(leaf.values instanceof Float32Array) ||
				leaf.values.length !== shapes[i][0] * shapes[i][1] ||
				leaf.values.some((v) => !Number.isFinite(v) || (label === 'v' && v < 0))
			)
				throw new Error(`Invalid token-story ${label} tensor ${i}`);
		});
	}
}
