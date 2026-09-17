import { describe, expect, it } from 'vitest';
import { QUERY_GROUPS } from './model/query-dataset';
import { validateQueryStudy, parseQueryStudy, type QueryStudyRecord } from './query-journal';

function fixture(): QueryStudyRecord {
	const split = () => ({
		intactProbabilities: Array.from({ length: 48 }, () => Array(8).fill(0.125)),
		otherProbabilities: Array(48).fill(0),
		activations: Array.from({ length: 256 }, () => Array(48).fill(0)),
		effects: Array.from({ length: 256 }, () => Array(384).fill(0)),
		accuracy: 1 / 8,
		prefixMaxDifference: 0,
		capturePredictionMaxDifference: 0,
		probabilityMassMaxError: 0
	});
	return {
		version: 1,
		kind: 'tissue-query-study',
		id: 'fixture',
		createdAt: '2026-09-17T00:00:00Z',
		source: 'browser',
		measurement: {
			version: 1,
			design: 'paired-query-v1',
			seed: 42,
			step: 2000,
			backend: 'wasm',
			capturedAt: '2026-09-17T00:00:00Z',
			elapsedMs: 1,
			groups: structuredClone(QUERY_GROUPS),
			splits: { calibration: split(), test: split() },
			outgoingWeights: Array.from({ length: 256 }, () => Array(32).fill(0)),
			checkpointHash: 'a'.repeat(64),
			checkpointHashAlgorithm: 'sha256-full-checkpoint-json-v1',
			checkpointPreserved: true,
			intervention: 'zero-all-token-positions'
		}
	};
}
describe('raw paired-query evidence import', () => {
	it('preserves full raw evidence across a JSON round trip without requiring object-key order', () => {
		const record = fixture();
		const group = record.measurement.groups.calibration[0];
		record.measurement.groups.calibration[0] = {
			examples: group.examples,
			order: group.order,
			assignment: group.assignment,
			id: group.id
		};
		expect(parseQueryStudy(JSON.stringify(record))).toEqual(record);
	});
	it('rejects a changed assignment prefix and a calibration/test group substitution', () => {
		const record = fixture();
		record.measurement.groups.calibration[0].examples[1].tokenIds[0] = 9;
		expect(() => validateQueryStudy(record)).toThrow('fixed paired design');
		const other = fixture();
		other.measurement.groups.test = other.measurement.groups.calibration;
		expect(() => validateQueryStudy(other)).toThrow('fixed paired design');
	});
	it('rejects impossible intervention probabilities even when each raw effect is in [-1, 1]', () => {
		const record = fixture();
		record.measurement.splits.test.effects[128][0] = 0.9;
		expect(() => validateQueryStudy(record)).toThrow('invalid lesioned probability');
	});
	it('rejects incomplete matrices, corrupt checkpoint identity, and failed prefix checks', () => {
		const record = fixture();
		record.measurement.splits.calibration.effects[0].pop();
		expect(() => validateQueryStudy(record)).toThrow('matrix column count');
		const second = fixture();
		second.measurement.checkpointHash = 'unknown';
		expect(() => validateQueryStudy(second)).toThrow('checkpoint hash');
		const third = fixture();
		third.measurement.splits.test.prefixMaxDifference = 0.1;
		expect(() => validateQueryStudy(third)).toThrow('causal prefix check');
	});
});
