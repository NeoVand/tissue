export type StoryBackend = 'webgpu' | 'wasm' | 'cpu';
export type StoryPreset = 'small' | 'medium' | 'large';
export interface StoryConfig {
	layers: number;
	width: number;
	heads: number;
	hidden: number;
	context: number;
	batchSize: number;
	vocabularySize: 96;
	learningRate: number;
}
export const STORY_PRESETS: Record<StoryPreset, StoryConfig> = {
	small: {
		layers: 4,
		width: 128,
		heads: 4,
		hidden: 512,
		context: 128,
		batchSize: 4,
		vocabularySize: 96,
		learningRate: 0.001
	},
	medium: {
		layers: 4,
		width: 256,
		heads: 8,
		hidden: 1024,
		context: 128,
		batchSize: 4,
		vocabularySize: 96,
		learningRate: 0.0006
	},
	large: {
		layers: 6,
		width: 384,
		heads: 8,
		hidden: 1536,
		context: 128,
		batchSize: 2,
		vocabularySize: 96,
		learningRate: 0.0004
	}
};
export const STORY_CHARACTERS = [
	'\n',
	...Array.from({ length: 95 }, (_, i) => String.fromCharCode(i + 32))
];
export const storyUnitCount = (config: StoryConfig) => config.layers * config.hidden;
export const storyParameterCount = (config: StoryConfig) =>
	2 * config.vocabularySize * config.width +
	config.context * config.width +
	config.layers * (4 * config.width * config.width + 2 * config.width * config.hidden);
export interface StoryCorpusInfo {
	id: string;
	dataset: 'TinyStories';
	trainTokens: number;
	calibrationTokens: number;
	evaluationTokens: number;
	trainStories: number;
	calibrationStories: number;
	evaluationStories: number;
	vocabulary: string[];
	source: string;
	license: string;
}
export interface StoryMetrics {
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
	backend: StoryBackend;
}
export interface StoryInitialization {
	modelId: string;
	config: StoryConfig;
	seed: number;
	backend: StoryBackend;
	parameterCount: number;
	unitCount: number;
	corpus: StoryCorpusInfo;
	metrics: StoryMetrics;
}
export interface StoryEncodedPrompt {
	original: string;
	/** Last context-length characters; unsupported characters cause rejection rather than replacement. */
	text: string;
	tokenIds: number[];
	truncatedCharacters: number;
	unsupportedCharacters: string[];
}
export interface StoryProbe {
	modelId: string;
	config: StoryConfig;
	seed: number;
	step: number;
	prompt: StoryEncodedPrompt;
	/** Token-major [actual prompt token][layer-major unit], raw post-ReLU after any lesion. */
	activations: Float32Array;
	unitCount: number;
	probabilities: Float32Array;
	predictedToken: number;
	lesionNeuron: number | null;
}
export interface StoryAtlas {
	version: 1;
	modelId: string;
	config: StoryConfig;
	seed: number;
	step: number;
	backend: StoryBackend;
	capturedAt: string;
	elapsedMs: number;
	corpusId: string;
	unitCount: number;
	dimensions: number;
	/** Unit-major raw ReLU [unit][8 fixed calibration windows × 16 positions]. */
	fingerprints: Float32Array;
	positions: number[];
	examples: string[];
}
export interface StoryGenerationOptions {
	maxTokens?: number;
	temperature?: number;
	topK?: number;
	seed?: number;
}
export interface StoryGeneration {
	modelId: string;
	step: number;
	prompt: StoryEncodedPrompt;
	completion: string;
	tokenIds: number[];
	samplingSeed: number;
	temperature: number;
	topK: number;
	requestedTokens: number;
	cancelled: boolean;
}
export interface StoryTensor {
	shape: number[];
	values: Float32Array;
}
export interface StoryCheckpoint {
	version: 1;
	architecture: 'stories-transformer-v1';
	modelId: string;
	config: StoryConfig;
	seed: number;
	step: number;
	elapsedMs: number;
	trainLoss: number | null;
	trainRngState: number;
	corpus: StoryCorpusInfo;
	parameters: StoryTensor[];
	optimizer: { m: StoryTensor[]; v: StoryTensor[]; t: number };
}
export type StoryEvent =
	| { type: 'status'; message: string }
	| { type: 'metrics'; metrics: StoryMetrics }
	| { type: 'measurement'; completed: number; total: number; step: number };

export function validateStoryConfig(value: unknown): asserts value is StoryConfig {
	const c = value as StoryConfig;
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
		c.context > 128 ||
		!Number.isInteger(c.batchSize) ||
		c.batchSize < 1 ||
		c.batchSize > 4 ||
		c.vocabularySize !== 96 ||
		!Number.isFinite(c.learningRate) ||
		c.learningRate < 1e-5 ||
		c.learningRate > 0.01
	)
		throw new Error('Unsupported story model configuration');
}
/** Actual Params insertion order, retained by JaxJS tree.leaves. */
export function storyParameterShapes(config: StoryConfig): number[][] {
	validateStoryConfig(config);
	return [
		[96, config.width],
		[config.context, config.width],
		[config.width, 96],
		...Array.from({ length: config.layers }, () => [
			...Array.from({ length: 4 }, () => [config.width, config.width]),
			[config.width, config.hidden],
			[config.hidden, config.width]
		]).flat()
	];
}
export function validateStoryCheckpoint(value: unknown): asserts value is StoryCheckpoint {
	const c = value as StoryCheckpoint;
	if (!c || c.version !== 1 || c.architecture !== 'stories-transformer-v1')
		throw new Error('Unsupported story checkpoint');
	validateStoryConfig(c.config);
	if (
		typeof c.modelId !== 'string' ||
		!c.modelId.length ||
		c.modelId.length > 200 ||
		!Number.isInteger(c.seed) ||
		c.seed < 0 ||
		c.seed > 0xffffffff ||
		!Number.isSafeInteger(c.step) ||
		c.step < 0 ||
		!Number.isInteger(c.trainRngState) ||
		c.trainRngState < 0 ||
		c.trainRngState > 0xffffffff ||
		!Number.isFinite(c.elapsedMs) ||
		c.elapsedMs < 0 ||
		(c.trainLoss !== null && (!Number.isFinite(c.trainLoss) || c.trainLoss < 0)) ||
		!c.corpus ||
		typeof c.corpus.id !== 'string' ||
		!/^tinystories-ascii-v1:[a-f0-9]{64}$/.test(c.corpus.id) ||
		c.corpus.dataset !== 'TinyStories' ||
		c.corpus.trainTokens !== 1773785 ||
		c.corpus.calibrationTokens !== 21719 ||
		c.corpus.evaluationTokens !== 108433 ||
		c.corpus.trainStories !== 2097 ||
		c.corpus.calibrationStories !== 32 ||
		c.corpus.evaluationStories !== 152 ||
		c.corpus.source !== 'https://huggingface.co/datasets/roneneldan/TinyStories' ||
		c.corpus.license !== 'CDLA-Sharing-1.0' ||
		!Array.isArray(c.corpus.vocabulary) ||
		c.corpus.vocabulary.length !== 96 ||
		c.corpus.vocabulary.some((character, index) => character !== STORY_CHARACTERS[index]) ||
		c.optimizer?.t !== c.step
	)
		throw new Error('Invalid story checkpoint metadata');
	const shapes = storyParameterShapes(c.config);
	for (const [label, leaves] of [
		['parameters', c.parameters],
		['m', c.optimizer.m],
		['v', c.optimizer.v]
	] as const) {
		if (!Array.isArray(leaves) || leaves.length !== shapes.length)
			throw new Error(`Invalid story ${label} tensor count`);
		leaves.forEach((leaf, i) => {
			if (
				!leaf ||
				!Array.isArray(leaf.shape) ||
				leaf.shape.length !== 2 ||
				leaf.shape.some((dimension, index) => dimension !== shapes[i][index]) ||
				!(leaf.values instanceof Float32Array) ||
				leaf.values.length !== shapes[i][0] * shapes[i][1] ||
				leaf.values.some((v) => !Number.isFinite(v) || (label === 'v' && v < 0))
			)
				throw new Error(`Invalid story ${label} tensor ${i}`);
		});
	}
}
