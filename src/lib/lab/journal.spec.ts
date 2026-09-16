/// <reference types="node" />
import { afterEach, describe, expect, it, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import {
	getJournalWarnings,
	parseRun,
	readRuns,
	saveRun,
	validateRun,
	type MapFrame,
	type RunRecord
} from './journal';
import { CALIBRATION_EXAMPLES, TEST_EXAMPLES } from './model/dataset';
import {
	MODEL_CONFIG as cfg,
	NEURON_COUNT,
	type Atlas,
	type Checkpoint,
	type Metrics,
	type Probe
} from './protocol';

const timestamp = '2026-09-16T12:00:00.000Z';

/** A structurally complete synthetic fixture; no claimed training result. */
function fixture(): RunRecord {
	const shapes = [
		[14, 32],
		[14, 32],
		[32, 14],
		...Array.from({ length: 2 }, () => [
			[32, 32],
			[32, 32],
			[32, 32],
			[32, 32],
			[32, 128],
			[128, 32]
		]).flat()
	];
	const leaves = () =>
		shapes.map((shape) => ({
			shape: [...shape],
			values: Array<number>(shape[0] * shape[1]).fill(0)
		}));
	const metrics: Metrics = {
		step: 0,
		trainLoss: null,
		validationLoss: Math.log(14),
		accuracy: 0,
		elapsedMs: 0,
		backend: 'cpu',
		validationExamples: 96,
		chanceAccuracy: 0.125
	};
	const checkpoint: Checkpoint = {
		version: 1,
		architecture: 'binding-transformer-v1',
		seed: 42,
		step: 0,
		elapsedMs: 0,
		trainLoss: null,
		parameters: leaves(),
		optimizer: { m: leaves(), v: leaves(), t: 0 },
		randomState: 1
	};
	const map: MapFrame = {
		positions: Array.from({ length: NEURON_COUNT }, () => [0, 0, 0]),
		edges: [],
		magnitudes: Array<number>(NEURON_COUNT).fill(0),
		valid: Array<boolean>(NEURON_COUNT).fill(false),
		neighbors: Array.from({ length: NEURON_COUNT }, () => []),
		neighborRetention: null,
		explainedVariance: 0,
		rank: 0,
		boundaryDegenerate: false,
		alignment: { applied: false, rmsDisplacement: null }
	};
	const probe: Probe = {
		step: 0,
		exampleIndex: 0,
		example: structuredClone(CALIBRATION_EXAMPLES[0]),
		probabilities: Array<number>(8).fill(1 / 14),
		otherProbability: 6 / 14,
		predictedAnswer: 'a',
		activations: Array.from({ length: 14 }, () => Array<number>(NEURON_COUNT).fill(0)),
		lesionNeuron: null
	};
	return {
		version: 1,
		id: 'synthetic-fixture',
		title: 'Synthetic structural fixture',
		createdAt: timestamp,
		updatedAt: timestamp,
		seed: 42,
		initialization: {
			seed: 42,
			backend: 'cpu',
			parameterCount: 25920,
			metrics: structuredClone(metrics),
			trainAssignments: 236,
			calibrationAssignments: 50,
			testAssignments: 50
		},
		metrics: [metrics],
		snapshots: [{ step: 0, metrics: structuredClone(metrics), activation: map, probe }],
		observations: [
			{
				id: 'initial',
				time: timestamp,
				step: 0,
				kind: 'measurement',
				title: 'Fixture',
				detail: 'Synthetic test data.'
			}
		],
		checkpoint,
		provenance: { source: 'browser', appVersion: '0.1.0', userAgent: 'test fixture' }
	};
}

function atlas(): Atlas {
	return {
		step: 0,
		seed: 42,
		capturedAt: timestamp,
		neurons: Array.from({ length: NEURON_COUNT }, (_, id) => ({
			id,
			layer: Math.floor(id / cfg.hidden),
			channel: id % cfg.hidden
		})),
		activationFingerprints: Array.from({ length: NEURON_COUNT }, () =>
			Array<number>(16 * 14).fill(0)
		),
		effectFingerprints: Array.from({ length: NEURON_COUNT }, () => Array<number>(16 * 8).fill(0)),
		outgoingWeights: Array.from({ length: NEURON_COUNT }, () => Array<number>(32).fill(0)),
		examples: structuredClone(CALIBRATION_EXAMPLES),
		calibrationAccuracy: 0,
		intervention: 'zero-all-token-positions'
	};
}

afterEach(() => vi.unstubAllGlobals());

describe('experiment validation', () => {
	it('round-trips a complete experiment including optional raw measurements', () => {
		const run = fixture();
		run.atlas = atlas();
		expect(parseRun(JSON.stringify(run))).toEqual(run);
	});

	it('accepts the actual checked-in reference experiment and repair measurements', () => {
		const run = parseRun(
			readFileSync(
				new URL('../../../static/experiments/binding-seed-42.json', import.meta.url),
				'utf8'
			)
		);
		expect(run.checkpoint?.step).toBe(2000);
		expect(run.repairs?.[0].testPredictions).toHaveLength(TEST_EXAMPLES.length);
	});

	it.each([
		null,
		[],
		{},
		{ version: 1 },
		{
			version: 1,
			id: 'broken',
			title: 'Broken',
			seed: 42,
			initialization: {},
			metrics: [],
			snapshots: [],
			observations: [],
			provenance: {},
			checkpoint: { architecture: 'binding-transformer-v1' }
		}
	])('rejects partial imports without allowing them into local storage: %j', (input) => {
		expect(() => parseRun(JSON.stringify(input))).toThrow('Invalid experiment');
	});

	it.each([
		[
			'missing dates',
			(run: RunRecord) => {
				run.createdAt = '';
			}
		],
		[
			'invalid dates',
			(run: RunRecord) => {
				run.createdAt = 'yesterday';
			}
		],
		[
			'mismatched seed',
			(run: RunRecord) => {
				run.checkpoint!.seed = 19;
			}
		],
		[
			'impossible calendar date',
			(run: RunRecord) => {
				run.createdAt = '2026-02-30T12:00:00.000Z';
			}
		],
		[
			'missing optimizer',
			(run: RunRecord) => {
				delete (run.checkpoint as Partial<Checkpoint>).optimizer;
			}
		],
		[
			'wrong tensor shape',
			(run: RunRecord) => {
				run.checkpoint!.parameters[0].shape = [32, 14];
			}
		],
		[
			'truncated tensor',
			(run: RunRecord) => {
				run.checkpoint!.parameters[0].values.pop();
			}
		],
		[
			'negative Adam variance',
			(run: RunRecord) => {
				run.checkpoint!.optimizer.v[0].values[0] = -1;
			}
		],
		[
			'Float32 overflow',
			(run: RunRecord) => {
				run.checkpoint!.parameters[0].values[0] = 1e100;
			}
		],
		[
			'optimizer step mismatch',
			(run: RunRecord) => {
				run.checkpoint!.optimizer.t = 1;
			}
		],
		[
			'invalid accuracy',
			(run: RunRecord) => {
				run.metrics[0].accuracy = 1.2;
			}
		],
		[
			'nonfinite metric',
			(run: RunRecord) => {
				run.metrics[0].validationLoss = NaN;
			}
		],
		[
			'wrong evaluation set',
			(run: RunRecord) => {
				run.metrics[0].validationExamples = 16;
			}
		],
		[
			'duplicate steps',
			(run: RunRecord) => {
				run.metrics.push(structuredClone(run.metrics[0]));
			}
		],
		[
			'wrong activation shape',
			(run: RunRecord) => {
				run.snapshots[0].probe.activations[0].pop();
			}
		],
		[
			'wrong coordinate shape',
			(run: RunRecord) => {
				run.snapshots[0].activation.positions.pop();
			}
		],
		[
			'nonfinite coordinate',
			(run: RunRecord) => {
				run.snapshots[0].activation.positions[0][0] = Infinity;
			}
		],
		[
			'invalid probability mass',
			(run: RunRecord) => {
				run.snapshots[0].probe.otherProbability = 0;
			}
		],
		[
			'probe/checkpoint mismatch',
			(run: RunRecord) => {
				run.snapshots[0].probe.step = 2;
			}
		],
		[
			'wrong prompt identity',
			(run: RunRecord) => {
				run.snapshots[0].probe.example.tokens[0] = '?';
			}
		],
		[
			'out of range edge',
			(run: RunRecord) => {
				run.snapshots[0].activation.edges.push([0, 256]);
			}
		],
		[
			'unresolved neighbor',
			(run: RunRecord) => {
				run.snapshots[0].activation.neighbors[0].push(1);
			}
		],
		[
			'invalid retention',
			(run: RunRecord) => {
				run.snapshots[0].activation.neighborRetention = 1.1;
			}
		],
		[
			'duplicate observations',
			(run: RunRecord) => {
				run.observations.push(structuredClone(run.observations[0]));
			}
		],
		[
			'incomplete repair',
			(run: RunRecord) => {
				run.repairs = [{} as never];
			}
		]
	])('rejects %s', (_, mutate) => {
		const run = fixture();
		mutate(run);
		expect(() => validateRun(run)).toThrow('Invalid experiment');
	});

	it('rejects a snapshot newer than its durable weights', () => {
		const run = fixture();
		run.metrics.push({ ...run.metrics[0], step: 25 });
		run.snapshots[0].step = 25;
		run.snapshots[0].metrics.step = 25;
		run.snapshots[0].probe.step = 25;
		expect(() => validateRun(run)).toThrow('snapshot is newer than durable model weights');
	});

	it('keeps later observed metrics and failure notes without claiming they are resumable weights', () => {
		const run = fixture();
		run.metrics.push({ ...run.metrics[0], step: 25 });
		run.observations.push({
			id: 'failure',
			time: timestamp,
			step: 25,
			kind: 'failure',
			title: 'Interrupted',
			detail: 'Last weights are at step zero.'
		});
		expect(parseRun(JSON.stringify(run)).metrics.at(-1)?.step).toBe(25);
		expect(parseRun(JSON.stringify(run)).checkpoint?.step).toBe(0);
	});

	it('preserves checkpointless interrupted attempts as non-resumable evidence', () => {
		const run = fixture();
		delete run.checkpoint;
		run.snapshots = [];
		expect(parseRun(JSON.stringify(run)).checkpoint).toBeUndefined();
	});

	it('rejects wrong fingerprint dimensions and calibration provenance', () => {
		const run = fixture();
		run.atlas = atlas();
		run.atlas.effectFingerprints![0].pop();
		expect(() => validateRun(run)).toThrow('atlas.effectFingerprints');
		run.atlas = atlas();
		run.atlas.examples[0].split = 'test';
		expect(() => validateRun(run)).toThrow('fixed calibration prompt');
		run.atlas = atlas();
		run.atlas.effectFingerprints![0][0] = 2;
		expect(() => validateRun(run)).toThrow('atlas.effectFingerprints');
	});
});

describe('journal persistence boundaries', () => {
	it('rejects malformed writes before opening a database', async () => {
		const open = vi.fn();
		vi.stubGlobal('indexedDB', { open });
		await expect(saveRun({ version: 1 } as RunRecord)).rejects.toThrow('Invalid experiment');
		expect(open).not.toHaveBeenCalled();
	});

	it('isolates corrupt local records, reports their errors, and still returns valid runs', async () => {
		const first = fixture();
		const newer = fixture();
		newer.id = 'newer';
		newer.createdAt = '2026-09-16T13:00:00.000Z';
		newer.updatedAt = newer.createdAt;
		const close = vi.fn();
		const stored = [first, { id: 'bad', version: 1 }, newer];
		const getRequest: { result: unknown[]; onsuccess?: () => void } = { result: stored };
		const database = {
			close,
			transaction: () => ({
				objectStore: () => ({
					getAll: () => {
						queueMicrotask(() => getRequest.onsuccess?.());
						return getRequest;
					}
				})
			})
		};
		vi.stubGlobal('indexedDB', {
			open: () => {
				const request: { result: typeof database; onsuccess?: () => void } = { result: database };
				queueMicrotask(() => request.onsuccess?.());
				return request;
			}
		});
		const loaded = await readRuns();
		expect(loaded.map((run) => run.id)).toEqual(['newer', 'synthetic-fixture']);
		expect(getJournalWarnings()).toHaveLength(1);
		expect(getJournalWarnings()[0]).toContain('stored data was preserved');
		expect(stored).toHaveLength(3);
		expect(close).toHaveBeenCalledOnce();
	});
});
