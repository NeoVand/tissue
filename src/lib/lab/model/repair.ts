/* eslint-disable @typescript-eslint/no-explicit-any */
import { lax, numpy as np, tree } from '@jax-js/jax';
import { MODEL_CONFIG as cfg, type RepairArm, type Atlas } from '../protocol';
import { RandomStream } from './dataset';
import type { Params } from './transformer';

function normalized(row: number[], centered: boolean): number[] | null {
	const mean = centered ? row.reduce((a, b) => a + b, 0) / row.length : 0;
	const values = row.map((value) => value - mean);
	const norm = Math.sqrt(values.reduce((a, value) => a + value * value, 0));
	// Same directional validity threshold as the displayed geometry. A zero vector
	// has no direction and must never acquire neighbors from arbitrary index ties.
	return Number.isFinite(norm) && norm > 1e-12 ? values.map((value) => value / norm) : null;
}
/** All candidate sets are chosen using calibration fingerprints before repair. */
export function chooseRepairArms(
	atlas: Atlas,
	lesion: number,
	count: number,
	seed: number
): Pick<RepairArm, 'method' | 'neurons'>[] {
	const layer = Math.floor(lesion / cfg.hidden);
	const candidates = atlas.neurons
		.filter((node) => node.layer === layer && node.id !== lesion)
		.map((node) => node.id);
	const nearest = (fingerprints: number[][], centered: boolean, method: string) => {
		const source = normalized(fingerprints[lesion], centered);
		if (!source)
			throw new Error(
				`Cannot compare ${method} neighbors: the selected neuron's fingerprint has no measurable direction on these calibration probes. Select a resolved neuron.`
			);
		const resolved = candidates.flatMap((id) => {
			const target = normalized(fingerprints[id], centered);
			return target
				? [
						{
							id,
							distance: target.reduce((sum, value, index) => sum + (value - source[index]) ** 2, 0)
						}
					]
				: [];
		});
		if (resolved.length < count)
			throw new Error(
				`Cannot compare ${method} neighbors: only ${resolved.length} surviving same-layer neurons have a measurable direction; ${count} are required for this budget.`
			);
		return resolved
			.sort((a, b) => a.distance - b.distance || a.id - b.id)
			.slice(0, count)
			.map((item) => item.id);
	};
	const random = new RandomStream(seed ^ lesion ^ 0x72657061);
	const shuffled = [...candidates];
	for (let i = shuffled.length - 1; i > 0; i--) {
		const j = random.integer(i + 1);
		[shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
	}
	return [
		{ method: 'activation', neurons: nearest(atlas.activationFingerprints, true, 'activation') },
		...(atlas.effectFingerprints
			? [
					{
						method: 'intervention' as const,
						neurons: nearest(atlas.effectFingerprints, false, 'intervention')
					}
				]
			: []),
		{
			method: 'outgoing-weight',
			neurons: nearest(atlas.outgoingWeights, false, 'outgoing-weight')
		},
		{ method: 'random', neurons: shuffled.slice(0, count) }
	];
}
export function trainableMask(template: Params, neurons: number[]): Params {
	const mask = tree.map((leaf: any) => np.zeros(leaf.shape), template) as unknown as Params;
	for (let layer = 0; layer < cfg.layers; layer++) {
		const incoming = new Float32Array(cfg.width * cfg.hidden);
		const outgoing = new Float32Array(cfg.hidden * cfg.width);
		for (const neuron of neurons)
			if (Math.floor(neuron / cfg.hidden) === layer) {
				const channel = neuron % cfg.hidden;
				for (let width = 0; width < cfg.width; width++) {
					incoming[width * cfg.hidden + channel] = 1;
					outgoing[channel * cfg.width + width] = 1;
				}
			}
		mask.layers[layer].mlpFc1.dispose();
		mask.layers[layer].mlpFc2.dispose();
		mask.layers[layer].mlpFc1 = np.array(incoming).reshape([cfg.width, cfg.hidden]);
		mask.layers[layer].mlpFc2 = np.array(outgoing).reshape([cfg.hidden, cfg.width]);
	}
	return mask;
}
/** Identity in the forward pass; derivative is exactly the 0/1 selection mask. */
export function freezeUnselected(params: Params, mask: Params): Params {
	return tree.map(
		(parameter: any, selected: any) => {
			const frozen = lax
				.stopGradient(parameter.ref)
				.mul(np.onesLike(selected.ref).sub(selected.ref));
			return parameter.mul(selected).add(frozen);
		},
		params,
		mask
	) as Params;
}
