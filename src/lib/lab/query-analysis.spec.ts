import { describe, expect, it } from 'vitest';
import {
	QUERY_ANALYSIS_DESIGN,
	QUERY_METHODS,
	analyzeQueryShifts,
	buildQueryCalibrationPlan,
	helmertQueryContrasts,
	queryRms,
	scoreQueryCalibrationPlan,
	type QueryCalibrationInput
} from './query-analysis';
import { QUERY_GROUPS } from './model/query-dataset';
import type { QueryMeasurement, QuerySplitMeasurement } from './query-protocol';

const vector = (index: number, dimensions = 4) =>
	Array.from({ length: dimensions }, (_, coordinate) => +(coordinate === index % dimensions));

function input(count = 32): QueryCalibrationInput {
	const rows = Array.from({ length: count }, (_, id) => vector(Math.floor(id / 8)));
	return {
		layers: Array<number>(count).fill(0),
		fullEffects: structuredClone(rows),
		queryEffects: structuredClone(rows),
		queryActivations: structuredClone(rows),
		outgoingWeights: structuredClone(rows)
	};
}

describe('query contrasts', () => {
	it('removes arbitrary shared per-group output offsets and preserves within-query energy', () => {
		const raw = [[1, 4, 3, 2, 7, -2, 9, 0, 8, 3, 5, 6]];
		const offset = raw.map((row) =>
			row.map((value, i) => value + (i < 6 ? [23, -5][i % 2] : [-8, 17][i % 2]))
		);
		const contrast = helmertQueryContrasts(raw, 2, 2)[0];
		expect(helmertQueryContrasts(offset, 2, 2)[0]).toEqual(contrast);
		let energy = 0;
		for (let group = 0; group < 2; group++) {
			for (let coordinate = 0; coordinate < 2; coordinate++) {
				const values = [0, 1, 2].map((query) => raw[0][group * 6 + query * 2 + coordinate]);
				const average = values.reduce((sum, value) => sum + value, 0) / 3;
				energy += values.reduce((sum, value) => sum + (value - average) ** 2, 0);
			}
		}
		expect(contrast.reduce((sum, value) => sum + value ** 2, 0)).toBeCloseTo(energy, 12);
	});

	it('preserves signed geometry under a common reordering of query labels', () => {
		const raw = [
			[1, 2, 4, 8, 5, 3],
			[0, 4, -2, 6, 7, 1]
		];
		const squaredDistance = (rows: number[][]) =>
			rows[0].reduce((sum, x, i) => sum + (x - rows[1][i]) ** 2, 0);
		const expected = squaredDistance(helmertQueryContrasts(raw, 1, 2));
		for (const order of [
			[0, 2, 1],
			[1, 0, 2],
			[1, 2, 0],
			[2, 0, 1],
			[2, 1, 0]
		]) {
			const reordered = raw.map((row) =>
				order.flatMap((query) => row.slice(query * 2, query * 2 + 2))
			);
			expect(squaredDistance(helmertQueryContrasts(reordered, 1, 2))).toBeCloseTo(expected, 12);
		}
	});

	it('keeps zero contrasts unresolved and rejects malformed dimensions/nonfinite values', () => {
		expect(helmertQueryContrasts([[4, 4, 4]], 1, 1)).toEqual([[0, 0]]);
		expect(queryRms([3, 4])).toBeCloseTo(Math.sqrt(12.5));
		expect(() => helmertQueryContrasts([[1, 2]], 1, 1)).toThrow('dimensions');
		expect(() => helmertQueryContrasts([[1, NaN, 3]], 1, 1)).toThrow('finite');
	});
});

describe('calibration-only neighborhood plan', () => {
	it('uses the same eligible, within-layer, log-strength-matched pool for all methods', () => {
		const raw = input(90);
		raw.layers = raw.layers.map((_, id) => +(id >= 45));
		raw.fullEffects = raw.fullEffects.map((row, id) =>
			row.map((value) => value * Math.exp(id / 10))
		);
		// A single unresolved baseline excludes a unit from every method's pool.
		raw.outgoingWeights[10].fill(0);
		const plan = buildQueryCalibrationPlan(raw);
		const unit = plan.units[20];
		const expected = Array.from({ length: 45 }, (_, id) => id)
			.filter((id) => id !== 20 && id !== 10)
			.sort((a, b) => {
				const distance = (id: number) =>
					Math.abs(
						Math.log(queryRms(raw.fullEffects[id])) - Math.log(queryRms(raw.fullEffects[20]))
					);
				return distance(a) - distance(b) || a - b;
			})
			.slice(0, 32);
		expect(unit.candidatePool).toEqual(expected);
		expect(plan.units[10].unresolvedCalibrationMethods).toEqual(['outgoing-weight']);
		for (const { id: method } of QUERY_METHODS) {
			expect(unit.selections[method].neighbors).toHaveLength(6);
			expect(unit.selections[method].neighbors.every((id) => expected.includes(id))).toBe(true);
		}
		expect(unit.candidatePool.every((id) => id < 45 && id !== 20)).toBe(true);
	});

	it('uses an RMS floor rather than vector length, and does not silently reduce k', () => {
		const raw = input(6);
		const below = QUERY_ANALYSIS_DESIGN.rmsFloor / 2;
		raw.queryEffects[0] = Array(4).fill(below);
		const plan = buildQueryCalibrationPlan(raw);
		expect(plan.units[0].calibrationEligible).toBe(false);
		expect(plan.units[1].candidatePool).toHaveLength(4);
		expect(plan.units[1].selections['query-effect'].neighbors).toEqual([]);
		const result = scoreQueryCalibrationPlan(plan, raw.queryEffects, raw.fullEffects, 42);
		expect(result.overall.scoredUnits).toBe(0);
		expect(result.overall.insufficientCandidateUnits).toBe(5);
		expect(result.overall.methods['query-effect'].meanScore).toBeNull();
	});

	it('breaks true ties by neuron ID reproducibly', () => {
		const raw = input(40);
		for (const matrix of [
			raw.fullEffects,
			raw.queryEffects,
			raw.queryActivations,
			raw.outgoingWeights
		]) {
			matrix.forEach((row) => row.fill(1));
		}
		const plan = buildQueryCalibrationPlan(raw);
		expect(plan.units[39].candidatePool).toEqual(Array.from({ length: 32 }, (_, id) => id));
		expect(plan.units[39].selections['query-effect'].neighbors).toEqual([0, 1, 2, 3, 4, 5]);
		expect(buildQueryCalibrationPlan(raw)).toEqual(plan);
	});
});

describe('held-out neighborhood transfer', () => {
	it('gets perfect transfer for stable synthetic clusters, with the exact uniform-pool expectation', () => {
		const raw = input();
		const plan = buildQueryCalibrationPlan(raw);
		const result = scoreQueryCalibrationPlan(plan, raw.queryEffects, raw.fullEffects, 42);
		expect(result.overall.scoredUnits).toBe(32);
		expect(result.overall.methods['query-effect'].meanScore).toBe(1);
		expect(result.units[0].randomScore).toBe(7 / 31);
		expect(result.overall.methods['query-effect'].meanDelta).toBeCloseTo(1 - 7 / 31, 12);
		expect(result.overall.methods['query-effect'].meanShuffledDelta!).toBeLessThan(0.2);
		expect(result.shuffledTestIdentities).not.toEqual(plan.units.map((unit) => unit.id));
	});

	it('never changes selections in response to held-out data or mutate the plan', () => {
		const raw = input();
		const plan = buildQueryCalibrationPlan(raw);
		const before = structuredClone(plan);
		const intact = scoreQueryCalibrationPlan(plan, raw.queryEffects, raw.fullEffects, 42);
		const altered = scoreQueryCalibrationPlan(
			plan,
			raw.queryEffects.map((row, id) => row.map((value) => (id % 2 ? -value : value))),
			raw.fullEffects,
			42
		);
		expect(plan).toEqual(before);
		expect(altered.units.map((unit) => unit.methods['query-effect'].neighbors)).toEqual(
			intact.units.map((unit) => unit.methods['query-effect'].neighbors)
		);
		expect(altered.overall.methods['query-effect'].meanScore).not.toEqual(
			intact.overall.methods['query-effect'].meanScore
		);
	});

	it('gives unresolved test neighbors zero credit, reports coverage, and excludes test-zero focals uniformly', () => {
		const raw = input();
		const plan = buildQueryCalibrationPlan(raw);
		const target = structuredClone(raw.queryEffects);
		target[1].fill(0);
		const result = scoreQueryCalibrationPlan(plan, target, raw.fullEffects, 42);
		expect(result.units[0].methods['query-effect'].neighbors).toContain(1);
		expect(result.units[0].methods['query-effect'].score).toBeCloseTo(5 / 6, 12);
		expect(result.units[0].methods['query-effect'].coverage).toBe(5 / 6);
		expect(result.units[0].randomScore).toBe(6 / 31);
		expect(result.units[0].randomCoverage).toBe(30 / 31);
		expect(result.units[1].status).toBe('test-unresolved');
		expect(result.overall.scoredUnits).toBe(31);
		for (const { id: method } of QUERY_METHODS) {
			expect(result.units[1].methods[method].score).toBeNull();
			expect(result.overall.methods[method].selectedNeighbors).toBe(31 * 6);
		}
		expect(result.shuffledTestIdentities[1]).toBe(1);
	});

	it('preserves each layer and resolution stratum under a deterministic descriptive shuffle', () => {
		const raw = input(64);
		raw.layers = raw.layers.map((_, id) => +(id >= 32));
		const plan = buildQueryCalibrationPlan(raw);
		const targets = structuredClone(raw.queryEffects);
		for (const id of [1, 5, 35, 50]) targets[id].fill(0);
		const result = scoreQueryCalibrationPlan(plan, targets, raw.fullEffects, 7);
		result.shuffledTestIdentities.forEach((donor, id) => {
			expect(raw.layers[donor]).toBe(raw.layers[id]);
			expect(queryRms(targets[donor]) > 1e-8).toBe(queryRms(targets[id]) > 1e-8);
		});
		expect(new Set(result.shuffledTestIdentities).size).toBe(64);
		expect(result.shuffledTestIdentities).toEqual(
			scoreQueryCalibrationPlan(plan, targets, raw.fullEffects, 7).shuffledTestIdentities
		);
		expect(result.layers.map((layer) => layer.scoredUnits)).toEqual([30, 30]);
	});

	it('agrees with exhaustive sampling for the exact size-six random expectation', () => {
		const raw = input(8);
		raw.queryEffects = Array.from({ length: 8 }, (_, id) => [Math.cos(id), Math.sin(id)]);
		const plan = buildQueryCalibrationPlan(raw);
		const result = scoreQueryCalibrationPlan(plan, raw.queryEffects, raw.fullEffects, 42);
		const pool = plan.units[0].candidatePool;
		const sampleMeans = pool.map((omitted) => {
			const sample = pool.filter((id) => id !== omitted);
			return sample.reduce((sum, id) => sum + Math.cos(id), 0) / 6;
		});
		expect(result.units[0].randomScore).toBeCloseTo(
			sampleMeans.reduce((a, b) => a + b, 0) / sampleMeans.length,
			12
		);
	});

	it('returns explicit nulls for an entirely unresolved held-out cohort', () => {
		const raw = input();
		const result = scoreQueryCalibrationPlan(
			buildQueryCalibrationPlan(raw),
			raw.queryEffects.map((row) => row.map(() => 0)),
			raw.fullEffects,
			42
		);
		expect(result.overall.testUnresolvedFocals).toBe(32);
		expect(result.overall.randomMeanScore).toBeNull();
		expect(result.overall.methods['query-effect'].meanDelta).toBeNull();
		expect(JSON.stringify(result)).not.toContain('NaN');
	});
});

function measurementFixture(): QueryMeasurement {
	const split = (): QuerySplitMeasurement => ({
		intactProbabilities: Array.from({ length: 48 }, () => Array(8).fill(0.125)),
		otherProbabilities: Array(48).fill(0),
		activations: Array.from({ length: 256 }, () =>
			Array.from({ length: 48 }, (_, i) => 1 + (i % 3))
		),
		effects: Array.from({ length: 256 }, () =>
			Array.from({ length: 384 }, (_, i) => {
				const answer = i % 8;
				const query = Math.floor(i / 8) % 3;
				return answer === 0 ? (query - 1) * 0.01 : answer === 1 ? (1 - query) * 0.01 : 0;
			})
		),
		accuracy: 0.125,
		prefixMaxDifference: 0,
		capturePredictionMaxDifference: 0,
		probabilityMassMaxError: 0
	});
	return {
		version: 1,
		design: 'paired-query-v1',
		seed: 42,
		step: 2000,
		backend: 'cpu',
		capturedAt: new Date(0).toISOString(),
		elapsedMs: 0,
		groups: structuredClone(QUERY_GROUPS),
		splits: { calibration: split(), test: split() },
		outgoingWeights: Array.from({ length: 256 }, () => Array(32).fill(0.1)),
		checkpointHash: '0'.repeat(64),
		checkpointHashAlgorithm: 'sha256-full-checkpoint-json-v1',
		checkpointPreserved: true,
		intervention: 'zero-all-token-positions'
	};
}

describe('complete paired-query analysis contract', () => {
	it('fits geometry from calibration only and preserves provenance and a common scoring cohort', () => {
		const measurement = measurementFixture();
		const first = analyzeQueryShifts(measurement);
		measurement.splits.test.effects.forEach((row) => row.fill(0));
		const altered = analyzeQueryShifts(measurement);
		expect(first.checkpointHash).toBe(measurement.checkpointHash);
		expect(first.overall.scoredUnits).toBe(256);
		expect(first.overall.methods['query-effect'].meanScore).toBe(1);
		expect(first.overall.methods['query-effect'].meanDelta).toBe(0);
		expect(altered.overall.scoredUnits).toBe(0);
		expect(altered.calibrationGeometry).toEqual(first.calibrationGeometry);
		expect(altered.units.map((unit) => unit.candidatePool)).toEqual(
			first.units.map((unit) => unit.candidatePool)
		);
		expect(first.calibrationGeometry.valid.filter(Boolean)).toHaveLength(256);
	});

	it('rejects overlapping assignment splits or a query triple whose prefix changed', () => {
		const overlap = measurementFixture();
		overlap.groups.test[0].assignment = [...overlap.groups.calibration[0].assignment];
		expect(() => analyzeQueryShifts(overlap)).toThrow('unique and disjoint');
		const changedPrefix = measurementFixture();
		changedPrefix.groups.calibration[0].examples[1].tokenIds[0] += 1;
		expect(() => analyzeQueryShifts(changedPrefix)).toThrow('identical query prefixes');
	});
});
