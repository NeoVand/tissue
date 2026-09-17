import type { Backend, BindingExample } from './protocol';

/** The three examples share all 13 prefix tokens and query a, b, c in this order. */
export interface QueryGroup {
	id: string;
	assignment: number[];
	order: number[];
	examples: BindingExample[];
}
export interface QuerySplitMeasurement {
	/** Group-major, query a/b/c; each row contains probabilities for answer values 0..7. */
	intactProbabilities: number[][];
	otherProbabilities: number[];
	/** Neuron × (group, query a/b/c): raw post-ReLU activation at the final query token. */
	activations: number[][];
	/** Neuron × (group, query a/b/c, answer 0..7): lesion minus intact probability. */
	effects: number[][];
	/** Argmax over the complete 14-token vocabulary, not conditional on answer tokens. */
	accuracy: number;
	/** Maximum intact activation difference among the three identical 13-token prefixes. */
	prefixMaxDifference: number;
	/** Maximum probability difference between capture and ordinary prediction, all 14 tokens. */
	capturePredictionMaxDifference: number;
	/** Maximum |sum(exp(logprob))-1| across intact and all lesioned predictions. */
	probabilityMassMaxError: number;
}
export interface QueryMeasurement {
	version: 1;
	design: 'paired-query-v1';
	seed: number;
	step: number;
	backend: Backend;
	capturedAt: string;
	elapsedMs: number;
	groups: { calibration: QueryGroup[]; test: QueryGroup[] };
	splits: { calibration: QuerySplitMeasurement; test: QuerySplitMeasurement };
	/** Neuron-major MLP output rows, each with 32 residual dimensions. */
	outgoingWeights: number[][];
	/** SHA-256 of UTF-8 JSON.stringify(exportCheckpoint()), including Adam and training RNG. */
	checkpointHash: string;
	checkpointHashAlgorithm: 'sha256-full-checkpoint-json-v1';
	/** Before/after serialized checkpoints are compared exactly before returning any result. */
	checkpointPreserved: true;
	intervention: 'zero-all-token-positions';
}
