/* eslint-disable @typescript-eslint/no-explicit-any */
import { numpy as np, nn, random, tree } from '@jax-js/jax';
import { tokenStoryUnitCount, type TokenStoryConfig, type TokenStoryTensor } from './protocol';

export interface TokenStoryParams {
	wte: any;
	wpe: any;
	lmHead: any;
	layers: { wq: any; wk: any; wv: any; wo: any; mlpFc1: any; mlpFc2: any }[];
}
export function initializeTokenStoryParameters(
	c: TokenStoryConfig,
	seed: number
): TokenStoryParams {
	const count = 3 + 6 * c.layers,
		keys = random.split(random.key(seed), count);
	let index = 0;
	const key = () => (++index < count ? keys.ref.slice(index - 1) : keys.slice(index - 1));
	const scale = Math.sqrt(3 / c.width);
	const matrix = (shape: number[], multiplier: number) =>
		random.uniform(key(), shape, { minval: -scale * multiplier, maxval: scale * multiplier });
	const p: TokenStoryParams = {
		wte: random.normal(key(), [c.vocabularySize, c.width]).mul(0.02),
		wpe: random.normal(key(), [c.context, c.width]).mul(0.02),
		lmHead: random.normal(key(), [c.width, c.vocabularySize]).mul(0.001),
		layers: []
	};
	for (let i = 0; i < c.layers; i++)
		p.layers.push({
			wq: matrix([c.width, c.width], 1),
			wk: matrix([c.width, c.width], 1),
			wv: matrix([c.width, c.width], 1),
			wo: matrix([c.width, c.width], 0.2),
			mlpFc1: matrix([c.width, c.hidden], 0.4),
			mlpFc2: matrix([c.hidden, c.width], 0.2)
		});
	return p;
}
const rms = (x: any) => x.div(np.sqrt(np.mean(np.square(x.ref), -1, { keepdims: true }).add(1e-5)));
/** Shared training/capture/inference computation. All tensor arguments are consumed. */
export function tokenStoryForward(
	p: TokenStoryParams,
	c: TokenStoryConfig,
	tokens: any,
	positions: any,
	masks: any,
	capture = false
) {
	const batch = tokens.shape[0],
		S = c.context,
		D = c.width;
	let x = rms(
		np
			.dot(tokens.reshape([-1, c.vocabularySize]), p.wte)
			.add(np.dot(positions.reshape([-1, S]), p.wpe))
	);
	const activations: any[] = [];
	for (let i = 0; i < c.layers; i++) {
		const l = p.layers[i],
			residual = x.ref;
		x = rms(x);
		const q = np.dot(x.ref, l.wq).reshape([batch, S, c.heads, D / c.heads]);
		const k = np.dot(x.ref, l.wk).reshape([batch, S, c.heads, D / c.heads]);
		const v = np.dot(x, l.wv).reshape([batch, S, c.heads, D / c.heads]);
		x = np
			.dot(nn.dotProductAttention(q, k, v, { isCausal: true }).reshape([-1, D]), l.wo)
			.add(residual);
		const mlpResidual = x.ref;
		const mask = (i === c.layers - 1 ? masks : masks.ref).slice(i);
		const hidden = nn.relu(np.dot(rms(x), l.mlpFc1)).mul(mask);
		if (capture) activations.push(hidden.ref.reshape([batch, S, c.hidden]));
		x = np.dot(hidden, l.mlpFc2).add(mlpResidual);
	}
	return { logprobs: nn.logSoftmax(np.dot(x, p.lmHead), -1), activations };
}
export function tokenStoryLoss(
	p: TokenStoryParams,
	c: TokenStoryConfig,
	tokens: any,
	positions: any,
	targets: any,
	lossMask: any,
	masks: any
) {
	const batch = tokens.shape[0],
		counts = np.sum(lossMask.ref, -1);
	const perToken = np
		.sum(
			tokenStoryForward(p, c, tokens, positions, masks).logprobs.mul(
				targets.reshape([-1, c.vocabularySize])
			),
			-1
		)
		.neg()
		.reshape([batch, c.context]);
	return np.mean(np.sum(perToken.mul(lossMask), -1).div(counts));
}
export function tokenStoryBatch(c: TokenStoryConfig, windows: ArrayLike<number>[]) {
	const input = new Int32Array(windows.length * c.context),
		targets = new Int32Array(input.length),
		positions = new Int32Array(input.length),
		valid = new Float32Array(input.length);
	windows.forEach((window, b) => {
		for (let t = 0; t < c.context; t++) {
			input[b * c.context + t] = window[t] ?? 97;
			targets[b * c.context + t] = window[t + 1] ?? 97;
			positions[b * c.context + t] = t;
			valid[b * c.context + t] = +(t + 1 < window.length);
		}
	});
	return {
		tokens: nn.oneHot(np.array(input).reshape([windows.length, c.context]), c.vocabularySize),
		positions: nn.oneHot(np.array(positions).reshape([windows.length, c.context]), c.context),
		targets: nn.oneHot(np.array(targets).reshape([windows.length, c.context]), c.vocabularySize),
		lossMask: np.array(valid).reshape([windows.length, c.context])
	};
}
export function tokenStoryMask(c: TokenStoryConfig, neuron?: number | null) {
	const data = new Float32Array(tokenStoryUnitCount(c)).fill(1);
	if (neuron !== undefined && neuron !== null) {
		if (!Number.isInteger(neuron) || neuron < 0 || neuron >= data.length)
			throw new Error('TokenStory neuron is outside the model');
		data[neuron] = 0;
	}
	return np.array(data).reshape([c.layers, c.hidden]);
}
export async function serializeTokenStoryTree(p: TokenStoryParams): Promise<TokenStoryTensor[]> {
	// Sequential reads keep peak staging memory bounded for large optimizer checkpoints.
	const output: TokenStoryTensor[] = [];
	for (const leaf of tree.leaves(p) as any[])
		output.push({ shape: [...leaf.shape], values: Float32Array.from(await leaf.ref.data()) });
	return output;
}
/** Rebuild the known leaf order without allocating a temporary random model. */
export function deserializeTokenStoryTree(
	config: TokenStoryConfig,
	leaves: TokenStoryTensor[]
): TokenStoryParams {
	let i = 0;
	const leaf = () => {
		const tensor = leaves[i++];
		return np.array(Float32Array.from(tensor.values)).reshape(tensor.shape);
	};
	return {
		wte: leaf(),
		wpe: leaf(),
		lmHead: leaf(),
		layers: Array.from({ length: config.layers }, () => ({
			wq: leaf(),
			wk: leaf(),
			wv: leaf(),
			wo: leaf(),
			mlpFc1: leaf(),
			mlpFc2: leaf()
		}))
	};
}
