import type { TokenStoryBackend, TokenStoryConfig } from './protocol';

/** Exact model input, retaining generated BPE IDs even when decoding/re-encoding would merge them. */
export interface LiveTokenStoryContext {
	tokenIds: number[];
	pieces: string[];
	text: string;
	/** Initial prompt truncation plus generated tokens dropped by the sliding window. */
	truncatedTokens: number;
	includesBos: boolean;
}

/** Provisional next-token measurement; its sample enters the result only after the frame is acknowledged. */
export interface LiveTokenStoryFrame {
	version: 1;
	modelId: string;
	step: number;
	trainedTokens: number;
	seed: number;
	backend: TokenStoryBackend;
	config: TokenStoryConfig;
	tokenizerId: string;
	corpusId: string;
	/** Zero-based index in this generation, not the position inside its context. */
	index: number;
	/** Original user prompt, before initial truncation. */
	prompt: string;
	context: LiveTokenStoryContext;
	/** Final real input position; this input predicts sampledToken. */
	position: number;
	unitCount: number;
	/** Post-ReLU MLP activations at position, flattened as layer * hidden + channel. */
	activations: Float32Array;
	/** Raw next-token probabilities before temperature, top-k filtering, or BOS exclusion. */
	probabilities: Float32Array;
	sampledToken: number;
	sampledPiece: string;
	isEos: boolean;
}

export type LiveTokenStoryFrameCallback = (frame: LiveTokenStoryFrame) => void | Promise<void>;
