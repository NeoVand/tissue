import { createHash } from 'node:crypto';
import { describe, expect, it } from 'vitest';
import { jit, tree } from '@jax-js/jax';
import { MODEL_CONFIG as cfg, NEURON_COUNT } from '../protocol';
import { assignmentSplits } from './dataset';
import { pairedQueryGroups, QUERY_BATCH_SIZE, QUERY_GROUP_COUNT } from './query-dataset';
import { ModelRuntime } from './runtime';
import {
	deserializeTree,
	forward,
	initializeParameters,
	lesionMask,
	oneHotBatch
} from './transformer';

describe('paired-query experimental design', () => {
	it('holds out entire assignments and changes only the final token within each triple', () => {
		const used = new Set<string>();
		const train = new Set(assignmentSplits.train.map((assignment) => assignment.join(',')));
		for (const split of ['calibration', 'test'] as const) {
			const groups = pairedQueryGroups(split);
			expect(groups).toHaveLength(QUERY_GROUP_COUNT);
			expect(groups).toEqual(pairedQueryGroups(split));
			const allowed = new Set(assignmentSplits[split].map((assignment) => assignment.join(',')));
			for (const group of groups) {
				const key = group.assignment.join(',');
				expect(used.has(key)).toBe(false);
				expect(train.has(key)).toBe(false);
				expect(allowed.has(key)).toBe(true);
				used.add(key);
				expect([...group.order].sort()).toEqual([0, 1, 2]);
				expect(group.examples.map((example) => example.query)).toEqual(['a', 'b', 'c']);
				expect(
					new Set(group.examples.map((example) => example.tokens.slice(0, -1).join(''))).size
				).toBe(1);
				expect(group.examples.map((example) => example.answer)).toEqual(
					group.assignment.map(String)
				);
				expect(group.examples.every((example) => example.split === split)).toBe(true);
			}
			expect((groups.length * 3) % QUERY_BATCH_SIZE).toBe(0);
		}
		expect(QUERY_BATCH_SIZE).toBeLessThanOrEqual(32);
	});

	it('returns fresh metadata so a consumer cannot change future measurement prompts', () => {
		const before = pairedQueryGroups('calibration');
		const changed = pairedQueryGroups('calibration');
		changed[0].assignment[0] = 99;
		changed[0].order.reverse();
		changed[0].examples[0].tokenIds[0] = 99;
		expect(pairedQueryGroups('calibration')).toEqual(before);
	});
});

describe('paired-query instrumentation', () => {
	it('measures real ablations, preserves the full checkpoint, and cancels without partial success', async () => {
		const model = new ModelRuntime();
		try {
			await model.initialize(7, 'wasm');
			await model.train(1);
			const before = await model.exportCheckpoint();
			const progress: number[] = [];
			const measurement = await model.measureQueryShifts((event) => {
				if (event.type === 'measurement') progress.push(event.completed);
			});
			expect(await model.exportCheckpoint()).toEqual(before);
			expect(measurement.checkpointHash).toBe(
				createHash('sha256').update(JSON.stringify(before)).digest('hex')
			);
			expect(measurement.checkpointPreserved).toBe(true);
			expect(measurement.step).toBe(1);
			expect(progress[0]).toBe(0);
			expect(progress.at(-1)).toBe(NEURON_COUNT * 2);
			for (const split of ['calibration', 'test'] as const) {
				const observed = measurement.splits[split];
				expect(observed.activations).toHaveLength(NEURON_COUNT);
				expect(
					observed.activations.every(
						(row) => row.length === 48 && row.every((value) => value >= 0 && Number.isFinite(value))
					)
				).toBe(true);
				expect(
					observed.effects.every((row) => row.length === 384 && row.every(Number.isFinite))
				).toBe(true);
				expect(observed.effects.flat().some((value) => Math.abs(value) > 1e-7)).toBe(true);
				expect(observed.intactProbabilities).toHaveLength(48);
				expect(observed.prefixMaxDifference).toBeLessThan(1e-6);
				expect(observed.capturePredictionMaxDifference).toBeLessThan(1e-6);
				expect(observed.probabilityMassMaxError).toBeLessThan(1e-5);
				for (let i = 0; i < 48; i++)
					expect(
						observed.intactProbabilities[i].reduce(
							(sum, value) => sum + value,
							observed.otherProbabilities[i]
						)
					).toBeCloseTo(1, 5);
			}

			// Independently run a differently sized batch through the actual forward function.
			// This catches swapped query/answer axes, wrong global IDs, and subtracting the wrong baseline.
			const template = initializeParameters(7);
			const params = deserializeTree(template, before.parameters);
			tree.dispose(template);
			const capture = jit((p, tokens, positions, mask) =>
				forward(p, tokens, positions, mask, true)
			);
			try {
				const neuron = 135;
				const examples = measurement.groups.test[5].examples;
				const batch = oneHotBatch(examples);
				batch.targets.dispose();
				const output = capture(tree.ref(params), batch.tokens, batch.positions, lesionMask(neuron));
				const [logprobs, ...layers] = await Promise.all(
					[output.logprobs, ...output.activations].map(async (array) => await array.data())
				);
				for (let query = 0; query < 3; query++) {
					for (let token = 0; token < cfg.sequenceLength; token++)
						expect(layers[1][(query * cfg.sequenceLength + token) * cfg.hidden + 7]).toBe(0);
					for (let digit = 0; digit < 8; digit++) {
						const index = (5 * 3 + query) * 8 + digit;
						const expected =
							Math.exp(logprobs[query * cfg.vocabulary.length + 6 + digit]) -
							measurement.splits.test.intactProbabilities[5 * 3 + query][digit];
						expect(measurement.splits.test.effects[neuron][index]).toBeCloseTo(expected, 6);
					}
				}
			} finally {
				capture.dispose();
				tree.dispose(params);
			}
			await expect(
				model.measureQueryShifts((event) => {
					if (event.type === 'measurement' && event.completed === 8) model.pause();
				})
			).rejects.toThrow(/cancelled/);
			expect(await model.exportCheckpoint()).toEqual(before);
		} finally {
			model.dispose();
		}
	}, 120000);
});
