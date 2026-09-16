/* eslint-disable @typescript-eslint/no-explicit-any */
import { numpy as np, nn, random, tree } from '@jax-js/jax';
import {
	MODEL_CONFIG as cfg,
	NEURON_COUNT,
	type BindingExample,
	type SerializedLeaf
} from '../protocol';

export type Params = {
	wte: any;
	wpe: any;
	lmHead: any;
	layers: { wq: any; wk: any; wv: any; wo: any; mlpFc1: any; mlpFc2: any }[];
};
const V = cfg.vocabulary.length,
	D = cfg.width,
	S = cfg.sequenceLength,
	H = cfg.hidden;
export function initializeParameters(seed: number): Params {
	const n = 3 + cfg.layers * 6;
	const keys = random.split(random.key(seed), n);
	let ki = 0;
	const nextKey = () => {
		ki++;
		return ki < n ? keys.ref.slice(ki - 1) : keys.slice(ki - 1);
	};
	const scale = Math.sqrt(3 / D);
	const matrix = (shape: number[], multiplier: number) =>
		random.uniform(nextKey(), shape, { minval: -scale * multiplier, maxval: scale * multiplier });
	const parameters: Params = {
		wte: random.normal(nextKey(), [V, D]).mul(0.02),
		wpe: random.normal(nextKey(), [S, D]).mul(0.02),
		lmHead: random.normal(nextKey(), [D, V]).mul(0.001),
		layers: []
	};
	for (let layer = 0; layer < cfg.layers; layer++)
		parameters.layers.push({
			wq: matrix([D, D], 1),
			wk: matrix([D, D], 1),
			wv: matrix([D, D], 1),
			wo: matrix([D, D], 0.2),
			mlpFc1: matrix([D, H], 0.4),
			mlpFc2: matrix([H, D], 0.2)
		});
	return parameters;
}
function rmsnorm(x: any) {
	return x.div(np.sqrt(np.mean(np.square(x.ref), -1, { keepdims: true }).add(1e-5)));
}
/** One implementation serves training, probes and interventions. All array inputs
 * are consumed. The mask zeros a channel at ALL token positions. */
export function forward(
	params: Params,
	tokenOH: any,
	positionOH: any,
	masks: any,
	capture = false
): { logprobs: any; activations: any[] } {
	const batch = tokenOH.shape[0];
	let x = rmsnorm(
		np
			.dot(tokenOH.reshape([-1, V]), params.wte)
			.add(np.dot(positionOH.reshape([-1, S]), params.wpe))
	);
	const activations: any[] = [];
	for (let l = 0; l < cfg.layers; l++) {
		const layer = params.layers[l];
		const residual = x.ref;
		x = rmsnorm(x);
		const q = np.dot(x.ref, layer.wq).reshape([batch, S, cfg.heads, D / cfg.heads]);
		const k = np.dot(x.ref, layer.wk).reshape([batch, S, cfg.heads, D / cfg.heads]);
		const v = np.dot(x, layer.wv).reshape([batch, S, cfg.heads, D / cfg.heads]);
		x = np
			.dot(nn.dotProductAttention(q, k, v, { isCausal: true }).reshape([-1, D]), layer.wo)
			.add(residual);
		const mlpResidual = x.ref;
		const layerMask = (l === cfg.layers - 1 ? masks : masks.ref).slice(l);
		const hidden = nn.relu(np.dot(rmsnorm(x), layer.mlpFc1)).mul(layerMask);
		if (capture) activations.push(hidden.ref.reshape([batch, S, H]));
		x = np.dot(hidden, layer.mlpFc2).add(mlpResidual);
	}
	const lastToken = x.reshape([batch, S, D]).slice([], S - 1, []);
	return { logprobs: nn.logSoftmax(np.dot(lastToken, params.lmHead), -1), activations };
}
/** Only the answer next-token is supervised: punctuation carries no loss. */
export function answerLoss(
	params: Params,
	tokenOH: any,
	positionOH: any,
	targetOH: any,
	masks: any
) {
	return np.mean(
		np.sum(forward(params, tokenOH, positionOH, masks).logprobs.mul(targetOH), -1).neg()
	);
}
export function oneHotBatch(examples: BindingExample[]) {
	const tokens = new Int32Array(examples.length * S);
	const positions = new Int32Array(tokens.length);
	const answers = new Int32Array(examples.length);
	examples.forEach((example, batch) => {
		tokens.set(example.tokenIds, batch * S);
		answers[batch] = example.answerId;
		for (let pos = 0; pos < S; pos++) positions[batch * S + pos] = pos;
	});
	return {
		tokens: nn.oneHot(np.array(tokens, { dtype: np.int32 }).reshape([examples.length, S]), V),
		positions: nn.oneHot(np.array(positions, { dtype: np.int32 }).reshape([examples.length, S]), S),
		targets: nn.oneHot(np.array(answers, { dtype: np.int32 }), V)
	};
}
export function lesionMask(neuron?: number | null) {
	const values = new Float32Array(NEURON_COUNT).fill(1);
	if (neuron !== undefined && neuron !== null) {
		if (!Number.isInteger(neuron) || neuron < 0 || neuron >= NEURON_COUNT)
			throw new Error('Neuron is outside the model');
		values[neuron] = 0;
	}
	return np.array(values).reshape([cfg.layers, H]);
}
export async function serializeTree(params: Params): Promise<SerializedLeaf[]> {
	return Promise.all(
		(tree.leaves(params) as any[]).map(async (leaf) => ({
			shape: [...leaf.shape],
			values: Array.from((await leaf.ref.data()) as Float32Array)
		}))
	);
}
/** Validate all leaves before allocating or changing the resident checkpoint. */
export function validateSerialized(template: Params, serialized: SerializedLeaf[]) {
	const leaves = tree.leaves(template) as any[];
	if (!Array.isArray(serialized) || leaves.length !== serialized.length)
		throw new Error('Checkpoint parameter count does not match architecture');
	leaves.forEach((leaf, i) => {
		const saved = serialized[i];
		if (
			!saved ||
			!Array.isArray(saved.shape) ||
			!Array.isArray(saved.values) ||
			saved.shape.join(',') !== leaf.shape.join(',') ||
			saved.values.length !== leaf.size ||
			!saved.values.every(Number.isFinite)
		)
			throw new Error(`Invalid checkpoint leaf ${i}`);
	});
}
export function deserializeTree(template: Params, serialized: SerializedLeaf[]): Params {
	validateSerialized(template, serialized);
	let index = 0;
	return tree.map(() => {
		const leaf = serialized[index++];
		return np.array(new Float32Array(leaf.values)).reshape(leaf.shape);
	}, template) as Params;
}
