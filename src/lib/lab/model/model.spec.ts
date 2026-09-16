import { describe, expect, it, beforeAll } from 'vitest';
import { init, defaultDevice, jit, tree, grad, numpy as np } from '@jax-js/jax';
import {
	assignmentSplits,
	CALIBRATION_EXAMPLES,
	TEST_EXAMPLES,
	RandomStream,
	sampleExample
} from './dataset';
import { forward, initializeParameters, lesionMask, oneHotBatch } from './transformer';
import { ModelRuntime } from './runtime';
import { chooseRepairArms, freezeUnselected, trainableMask } from './repair';
import { serializeTree } from './transformer';
import { MODEL_CONFIG, NEURON_COUNT, type Atlas } from '../protocol';

describe('binding corpus', () => {
	it('holds out whole binding assignments, independent of query and presentation order', () => {
		const keys = Object.values(assignmentSplits).flatMap((split) =>
			split.map((assignment) => assignment.join(','))
		);
		expect(new Set(keys).size).toBe(336);
		expect(keys).toHaveLength(336);
		const trainKeys = new Set(assignmentSplits.train.map((assignment) => assignment.join(',')));
		expect(
			[...CALIBRATION_EXAMPLES, ...TEST_EXAMPLES].every(
				(example) => !trainKeys.has(example.assignment.join(','))
			)
		).toBe(true);
		for (const examples of [CALIBRATION_EXAMPLES, TEST_EXAMPLES]) {
			expect(new Set(examples.map((example) => example.id)).size).toBe(examples.length);
			expect(
				Array.from(
					{ length: 8 },
					(_, digit) => examples.filter((example) => example.answer === String(digit)).length
				)
			).toEqual(Array(8).fill(examples.length / 8));
		}
	});
	it('has correct token targets and exactly three distinct candidate answers', () => {
		const random = new RandomStream(99);
		for (let i = 0; i < 30; i++) {
			const example = sampleExample('train', random);
			expect(example.tokenIds).toHaveLength(MODEL_CONFIG.sequenceLength);
			expect(new Set(example.assignment).size).toBe(3);
			expect(example.answerId).toBe(example.assignment[example.tokenIds.at(-1)!] + 6);
			expect(example.tokens.join('')).toMatch(/^[abc]=[0-7];[abc]=[0-7];[abc]=[0-7];\?[abc]$/);
		}
	});
});

describe('repair neighborhood validity', () => {
	function atlasWithOppositeDirections(): Atlas {
		// Invalid zero rows used to rank AHEAD of the valid opposite directions:
		// their distance was1 instead of the valid antipodal distance2.
		const rows = () =>
			Array.from({ length: NEURON_COUNT }, (_, id) =>
				id === 0 ? [1, -1] : id <= 8 ? [-1, 1] : [0, 0]
			);
		return {
			step: 0,
			seed: 42,
			capturedAt: new Date(0).toISOString(),
			neurons: Array.from({ length: NEURON_COUNT }, (_, id) => ({
				id,
				layer: Math.floor(id / MODEL_CONFIG.hidden),
				channel: id % MODEL_CONFIG.hidden
			})),
			activationFingerprints: rows(),
			effectFingerprints: rows(),
			outgoingWeights: rows(),
			examples: [],
			calibrationAccuracy: 0,
			intervention: 'zero-all-token-positions'
		};
	}
	it('excludes unresolved directions instead of treating zero vectors as nearer neighbors', () => {
		const selected = chooseRepairArms(atlasWithOppositeDirections(), 0, 8, 42);
		for (const arm of selected.filter((arm) => arm.method !== 'random'))
			expect(arm.neurons).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
	});
	it('rejects constant activation and zero intervention/outgoing targets before inventing neighborhoods', () => {
		const constant = atlasWithOppositeDirections();
		constant.activationFingerprints[0] = [4, 4];
		expect(() => chooseRepairArms(constant, 0, 8, 42)).toThrow(
			/activation.*no measurable direction/
		);
		const effects = atlasWithOppositeDirections();
		effects.effectFingerprints![0] = [0, 0];
		expect(() => chooseRepairArms(effects, 0, 8, 42)).toThrow(
			/intervention.*no measurable direction/
		);
		const outgoing = atlasWithOppositeDirections();
		outgoing.outgoingWeights[0] = [0, 0];
		expect(() => chooseRepairArms(outgoing, 0, 8, 42)).toThrow(
			/outgoing-weight.*no measurable direction/
		);
	});
	it('requires the full requested budget of resolved same-layer neighbors', () => {
		const atlas = atlasWithOppositeDirections();
		atlas.effectFingerprints![8] = [0, 0];
		// A valid unit in the other layer cannot fill the missing same-layer slot.
		atlas.effectFingerprints![MODEL_CONFIG.hidden] = [-1, 1];
		expect(() => chooseRepairArms(atlas, 0, 8, 42)).toThrow(/intervention.*only 7.*8 are required/);
	});
});

describe('transformer instrumentation', () => {
	beforeAll(async () => {
		await init('wasm');
		defaultDevice('wasm');
	});
	it('uses identical intact predictions with or without capture; lesion really zeros that channel', () => {
		const params = initializeParameters(42);
		const infer = jit((p, tokens, positions, mask) => forward(p, tokens, positions, mask));
		const capture = jit((p, tokens, positions, mask) => forward(p, tokens, positions, mask, true));
		const execute = (captureMode: boolean, neuron?: number) => {
			const batch = oneHotBatch([CALIBRATION_EXAMPLES[0]]);
			batch.targets.dispose();
			return (captureMode ? capture : infer)(
				tree.ref(params),
				batch.tokens,
				batch.positions,
				lesionMask(neuron)
			);
		};
		const intact = execute(false);
		const captured = execute(true);
		const a = Array.from(intact.logprobs.dataSync() as Float32Array);
		const b = Array.from(captured.logprobs.dataSync() as Float32Array);
		expect(Math.max(...a.map((value, i) => Math.abs(value - b[i])))).toBeLessThan(1e-6);
		tree.dispose(captured.activations);
		const damaged = execute(true, 7);
		const acts = damaged.activations[0].dataSync();
		expect(
			Array.from(
				{ length: MODEL_CONFIG.sequenceLength },
				(_, token) => acts[token * MODEL_CONFIG.hidden + 7]
			)
		).toEqual(Array(MODEL_CONFIG.sequenceLength).fill(0));
		damaged.logprobs.dispose();
		damaged.activations[1].dispose();
		expect(() => lesionMask(NEURON_COUNT)).toThrow();
		infer.dispose();
		capture.dispose();
		tree.dispose(params);
	});
	it('the repair gradient changes only the selected neurons incoming/outgoing weights', async () => {
		const parameters = initializeParameters(9);
		const mask = trainableMask(parameters, [2, 9, 135]);
		const derivative = jit(
			grad((p, selected) => {
				const frozen = freezeUnselected(p, selected);
				return tree
					.leaves(frozen)
					.map((leaf) => np.sum(leaf))
					.reduce((a, b) => a.add(b));
			})
		);
		const gradient = derivative(tree.ref(parameters), tree.ref(mask));
		const serializedMask = await serializeTree(mask);
		expect(await serializeTree(gradient)).toEqual(serializedMask);
		expect(
			serializedMask.flatMap((leaf) => leaf.values).reduce((sum, value) => sum + value, 0)
		).toBe(3 * 64);
		derivative.dispose();
		tree.dispose(gradient);
		tree.dispose(mask);
		tree.dispose(parameters);
	});

	it('restores exact optimizer/RNG trajectory; probes and measurements do not affect sampling', async () => {
		const model = new ModelRuntime();
		try {
			const initialized = await model.initialize(17, 'wasm');
			expect(initialized.metrics.validationLoss).toBeGreaterThan(2);
			expect(initialized.metrics.validationLoss).toBeLessThan(3);
			await model.train(2);
			const checkpoint = await model.exportCheckpoint();
			await model.train(2);
			const reference = await model.exportCheckpoint();
			await model.loadCheckpoint(checkpoint);
			await model.probe(0);
			const atlas = await model.captureAtlas(false);
			const repair = await model.controlledRepair({
				atlas,
				lesionNeuron: 6,
				steps: 2,
				neighborCount: 4
			});
			expect(repair.arms).toHaveLength(3);
			expect(await model.exportCheckpoint()).toEqual(checkpoint);
			expect(atlas.activationFingerprints).toHaveLength(NEURON_COUNT);
			expect(atlas.activationFingerprints[0]).toHaveLength(
				CALIBRATION_EXAMPLES.length * MODEL_CONFIG.sequenceLength
			);
			await model.train(2);
			const resumed = await model.exportCheckpoint();
			expect(resumed.randomState).toBe(reference.randomState);
			expect(resumed.parameters).toEqual(reference.parameters);
			expect(resumed.optimizer).toEqual(reference.optimizer);
		} finally {
			model.dispose();
		}
	}, 120000);
});
