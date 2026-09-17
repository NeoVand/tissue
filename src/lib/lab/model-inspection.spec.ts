import { beforeAll, describe, expect, it } from 'vitest';
import { init, defaultDevice, tree } from '@jax-js/jax';
import { initializeParameters, serializeTree } from './model/transformer';
import {
	inspectNeuron,
	inspectActivations,
	neuronAddress,
	PARAMETER_DESCRIPTORS,
	MODEL_PARAMETER_COUNT,
	vectorStatistics
} from './model-inspection';
import { CALIBRATION_EXAMPLES } from './model/dataset';
import { MODEL_CONFIG as cfg, NEURON_COUNT, type Checkpoint, type Probe } from './protocol';

describe('actual checkpoint-to-neuron mapping', () => {
	beforeAll(async () => {
		await init('wasm');
		defaultDevice('wasm');
	});
	it('matches every declared tensor name and serialized leaf to the real Params tree', async () => {
		const params = initializeParameters(91);
		try {
			const serialized = await serializeTree(params);
			const named = [
				['wte', params.wte],
				['wpe', params.wpe],
				['lmHead', params.lmHead],
				...params.layers.flatMap((layer, index) => [
					[`layers[${index}].wq`, layer.wq],
					[`layers[${index}].wk`, layer.wk],
					[`layers[${index}].wv`, layer.wv],
					[`layers[${index}].wo`, layer.wo],
					[`layers[${index}].mlpFc1`, layer.mlpFc1],
					[`layers[${index}].mlpFc2`, layer.mlpFc2]
				])
			] as const;
			for (let index = 0; index < named.length; index++) {
				const [name, array] = named[index];
				expect(PARAMETER_DESCRIPTORS[index].name).toBe(name);
				expect(PARAMETER_DESCRIPTORS[index].leafIndex).toBe(index);
				expect(PARAMETER_DESCRIPTORS[index].shape).toEqual(array.shape);
				expect(serialized[index].values).toEqual(Array.from(await array.ref.data()));
			}
			expect(MODEL_PARAMETER_COUNT).toBe(25920);
		} finally {
			tree.dispose(params);
		}
	});
	it('extracts incoming columns and outgoing rows at both layer boundaries, without transposing them', () => {
		const checkpoint = {
			version: 1,
			architecture: 'binding-transformer-v1',
			step: 2000,
			parameters: PARAMETER_DESCRIPTORS.map((descriptor) => ({
				shape: [...descriptor.shape],
				values: Array.from(
					{ length: descriptor.shape[0] * descriptor.shape[1] },
					(_, offset) => descriptor.leafIndex * 100000 + offset
				)
			}))
		} as Checkpoint;
		for (const id of [0, 127, 128, 183, 255]) {
			const inspected = inspectNeuron(checkpoint, id);
			const layer = Math.floor(id / 128),
				channel = id % 128;
			expect(inspected.incoming.values).toEqual(
				Array.from(
					{ length: 32 },
					(_, dimension) => (7 + layer * 6) * 100000 + dimension * 128 + channel
				)
			);
			expect(inspected.outgoing.values).toEqual(
				Array.from(
					{ length: 32 },
					(_, dimension) => (8 + layer * 6) * 100000 + channel * 32 + dimension
				)
			);
			expect(inspected.incoming.selection).toBe(`[:, ${channel}]`);
			expect(inspected.outgoing.selection).toBe(`[${channel}, :]`);
		}
		checkpoint.parameters[7].shape = [128, 32];
		expect(() => inspectNeuron(checkpoint, 0)).toThrow(/Invalid tensor shape/);
	});
	it('preserves raw token activations and their own checkpoint/lesion provenance', () => {
		const probe = {
			step: 500,
			example: CALIBRATION_EXAMPLES[0],
			lesionNeuron: 5,
			activations: Array.from({ length: cfg.sequenceLength }, (_, token) =>
				Array.from({ length: NEURON_COUNT }, (_, id) => (id === 5 ? 0 : token * 1000 + id))
			)
		} as Probe;
		const profile = inspectActivations(probe, 183);
		expect(profile.values).toEqual(Array.from({ length: 14 }, (_, token) => token * 1000 + 183));
		expect(profile.step).toBe(500);
		expect(profile.lesionNeuron).toBe(5);
		expect(inspectActivations(probe, 5).values).toEqual(Array(14).fill(0));
		expect(() => inspectActivations({ ...probe, activations: [] }, 183)).toThrow(/sequence/);
	});
	it('computes signed statistics and rejects undefined or nonfinite channels', () => {
		expect(vectorStatistics([-3, 0, 4])).toEqual({
			min: -3,
			max: 4,
			mean: 1 / 3,
			l1: 7,
			l2: 5,
			rms: Math.sqrt(25 / 3),
			nonzero: 2
		});
		expect(() => vectorStatistics([NaN])).toThrow(/finite/);
		expect(() => neuronAddress(256)).toThrow(/outside/);
		expect(() => neuronAddress(1.5)).toThrow(/outside/);
	});
});
