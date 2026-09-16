/** Versioned browser-worker contract. Every observation identifies its checkpoint. */
export const MODEL_CONFIG = {
	layers: 2,
	width: 32,
	heads: 4,
	hidden: 128,
	sequenceLength: 14,
	vocabulary: ['a', 'b', 'c', '=', ';', '?', '0', '1', '2', '3', '4', '5', '6', '7'],
	batchSize: 32,
	learningRate: 0.002
} as const;
export const NEURON_COUNT = MODEL_CONFIG.layers * MODEL_CONFIG.hidden;
export type Backend = 'webgpu' | 'wasm' | 'cpu';
export interface Metrics {
	step: number;
	trainLoss: number | null;
	validationLoss: number;
	accuracy: number;
	elapsedMs: number;
	backend: Backend;
	validationExamples: number;
	chanceAccuracy: number;
}
export interface BindingExample {
	id: string;
	tokens: string[];
	tokenIds: number[];
	answer: string;
	answerId: number;
	assignment: number[];
	query: string;
	split: 'train' | 'calibration' | 'test';
}
export interface Probe {
	step: number;
	exampleIndex: number;
	example: BindingExample;
	probabilities: number[];
	otherProbability: number;
	predictedAnswer: string;
	/** token × neuron, post-ReLU and post-lesion, layer-major neuron order. */
	activations: number[][];
	lesionNeuron: number | null;
}
export interface NeuronMetadata {
	id: number;
	layer: number;
	channel: number;
}
export interface Atlas {
	step: number;
	seed: number;
	capturedAt: string;
	neurons: NeuronMetadata[];
	/** Neuron × (fixed calibration example, token), raw post-ReLU values. */
	activationFingerprints: number[][];
	/** Neuron × (fixed calibration example, answer value), lesion minus intact probabilities. */
	effectFingerprints?: number[][];
	outgoingWeights: number[][];
	examples: BindingExample[];
	calibrationAccuracy: number;
	intervention: 'zero-all-token-positions' | 'not-measured';
}
export interface Initialization {
	seed: number;
	backend: Backend;
	parameterCount: number;
	metrics: Metrics;
	trainAssignments: number;
	calibrationAssignments: number;
	testAssignments: number;
}
export interface SerializedLeaf {
	shape: number[];
	values: number[];
}
export interface Checkpoint {
	version: 1;
	architecture: 'binding-transformer-v1';
	seed: number;
	step: number;
	elapsedMs: number;
	trainLoss: number | null;
	parameters: SerializedLeaf[];
	optimizer: { m: SerializedLeaf[]; v: SerializedLeaf[]; t: number };
	/** Exact sampling state: measuring or probing does not change this stream. */
	randomState: number;
}
export type EngineEvent =
	| { type: 'metrics'; metrics: Metrics }
	| { type: 'status'; message: string }
	| { type: 'measurement'; completed: number; total: number; step: number };
export interface RpcRequest {
	id: number;
	op: string;
	[key: string]: unknown;
}
export interface RpcResponse {
	id: number;
	ok?: boolean;
	result?: unknown;
	error?: string;
	event?: EngineEvent;
}
export interface RepairOptions {
	atlas: Atlas;
	lesionNeuron: number;
	steps?: number;
	neighborCount?: number;
}
export interface RepairObservation {
	meanAnswerProbability: number;
	update: number;
	loss: number;
	accuracy: number;
}
export interface RepairArm {
	method: 'activation' | 'intervention' | 'outgoing-weight' | 'random';
	neurons: number[];
	trainableParameters: number;
	curve: RepairObservation[];
}
export interface RepairResult {
	step: number;
	seed: number;
	lesionNeuron: number;
	steps: number;
	neighborCount: number;
	intact: RepairObservation;
	lesioned: RepairObservation;
	arms: RepairArm[];
	testExamples: number;
	testPredictions: {
		id: string;
		prompt: string;
		answer: string;
		intactProbability: number;
		lesionedProbability: number;
		armProbabilities: { method: RepairArm['method']; probability: number }[];
	}[];
	completedAt: string;
	caveat: string;
}
