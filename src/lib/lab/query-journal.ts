import { MODEL_CONFIG, NEURON_COUNT } from './protocol';
import type { QueryMeasurement } from './query-protocol';
import { QUERY_GROUPS } from './model/query-dataset';

export interface QueryStudyRecord {
	version: 1;
	kind: 'tissue-query-study';
	id: string;
	createdAt: string;
	source: 'browser' | 'reference';
	measurement: QueryMeasurement;
	provenance?: {
		commit: string;
		workingTreeDirty: boolean;
		sourceHashes: Record<string, string>;
		checkpointFile: string;
		checkpointFileHash: string;
		designHash: string;
	};
}

const DATABASE = 'tissue-query-studies';
const STORE = 'studies';
const HASH = /^[a-f0-9]{64}$/;
const check = (condition: unknown, message: string): void => {
	if (!condition) throw new Error(`Invalid query study: ${message}`);
};
const finite = (value: unknown, min = -Infinity, max = Infinity): value is number =>
	typeof value === 'number' && Number.isFinite(value) && value >= min && value <= max;
const integer = (value: unknown, max = Number.MAX_SAFE_INTEGER): value is number =>
	finite(value, 0, max) && Number.isSafeInteger(value);
function matrix(value: unknown, rows: number, columns: number, min: number, max: number) {
	check(Array.isArray(value) && value.length === rows, 'incorrect matrix row count');
	for (const row of value as unknown[]) {
		check(Array.isArray(row) && row.length === columns, 'incorrect matrix column count');
		check(
			(row as unknown[]).every((v) => finite(v, min, max)),
			'invalid matrix value'
		);
	}
}

function matchesDesign(value: unknown, expected: unknown): boolean {
	if (Array.isArray(expected))
		return (
			Array.isArray(value) &&
			value.length === expected.length &&
			expected.every((item, i) => matchesDesign(value[i], item))
		);
	if (expected && typeof expected === 'object')
		return (
			!!value &&
			typeof value === 'object' &&
			Object.entries(expected).every(([key, item]) =>
				matchesDesign((value as Record<string, unknown>)[key], item)
			)
		);
	return value === expected;
}

/** Validate raw evidence before it reaches the analysis worker or the saved archive. */
export function validateQueryMeasurement(value: unknown): QueryMeasurement {
	check(value !== null && typeof value === 'object', 'missing measurement');
	const m = value as QueryMeasurement;
	check(m.version === 1 && m.design === 'paired-query-v1', 'unsupported measurement design');
	check(integer(m.seed, 0xffffffff) && integer(m.step), 'invalid seed or checkpoint step');
	check(['webgpu', 'wasm', 'cpu'].includes(m.backend), 'unknown backend');
	check(
		typeof m.capturedAt === 'string' && Number.isFinite(Date.parse(m.capturedAt)),
		'invalid capture date'
	);
	check(finite(m.elapsedMs, 0), 'invalid measurement duration');
	check(
		typeof m.checkpointHash === 'string' && HASH.test(m.checkpointHash),
		'missing checkpoint hash'
	);
	check(
		m.checkpointHashAlgorithm === 'sha256-full-checkpoint-json-v1' &&
			m.checkpointPreserved === true,
		'checkpoint preservation was not verified'
	);
	check(m.intervention === 'zero-all-token-positions', 'unsupported intervention');
	check(!!m.groups && !!m.splits, 'missing paired groups');
	for (const split of ['calibration', 'test'] as const) {
		check(
			matchesDesign(m.groups[split], QUERY_GROUPS[split]),
			`${split} prompts do not match the fixed paired design`
		);
		const s = m.splits[split];
		check(!!s && typeof s === 'object', `missing ${split} measurements`);
		const examples = QUERY_GROUPS[split].length * 3;
		matrix(s.intactProbabilities, examples, 8, 0, 1);
		check(
			Array.isArray(s.otherProbabilities) &&
				s.otherProbabilities.length === examples &&
				s.otherProbabilities.every((v) => finite(v, 0, 1)),
			'invalid non-answer probability mass'
		);
		for (let i = 0; i < examples; i++) {
			const sum = s.intactProbabilities[i].reduce((a, b) => a + b, s.otherProbabilities[i]);
			check(Math.abs(sum - 1) <= 0.001, 'intact probability mass does not sum to one');
		}
		matrix(s.activations, NEURON_COUNT, examples, 0, 3.4028234663852886e38);
		matrix(s.effects, NEURON_COUNT, examples * 8, -1, 1);
		for (const effect of s.effects) {
			for (let i = 0; i < examples; i++) {
				let sum = 0;
				for (let answer = 0; answer < 8; answer++) {
					const probability = s.intactProbabilities[i][answer] + effect[i * 8 + answer];
					check(
						finite(probability, -0.000001, 1.000001),
						'effect implies an invalid lesioned probability'
					);
					sum += probability;
				}
				check(sum <= 1.001, 'lesioned answer probability mass exceeds one');
			}
		}
		check(finite(s.accuracy, 0, 1), 'invalid accuracy');
		check(finite(s.prefixMaxDifference, 0, 0.000001), 'causal prefix check failed');
		check(finite(s.capturePredictionMaxDifference, 0, 0.000001), 'capture parity check failed');
		check(finite(s.probabilityMassMaxError, 0, 0.001), 'probability mass check failed');
	}
	matrix(
		m.outgoingWeights,
		NEURON_COUNT,
		MODEL_CONFIG.width,
		-3.4028234663852886e38,
		3.4028234663852886e38
	);
	return m;
}

export function validateQueryStudy(value: unknown): QueryStudyRecord {
	check(value !== null && typeof value === 'object', 'missing record');
	const record = value as QueryStudyRecord;
	check(record.version === 1 && record.kind === 'tissue-query-study', 'unsupported record type');
	check(
		typeof record.id === 'string' && record.id.length > 0 && record.id.length < 200,
		'invalid identity'
	);
	check(
		typeof record.createdAt === 'string' && Number.isFinite(Date.parse(record.createdAt)),
		'invalid creation date'
	);
	check(record.source === 'browser' || record.source === 'reference', 'invalid source');
	validateQueryMeasurement(record.measurement);
	if (record.provenance !== undefined) {
		const p = record.provenance;
		check(
			p &&
				typeof p === 'object' &&
				typeof p.commit === 'string' &&
				typeof p.workingTreeDirty === 'boolean',
			'invalid source provenance'
		);
		check(
			typeof p.checkpointFile === 'string' &&
				HASH.test(p.checkpointFileHash) &&
				HASH.test(p.designHash),
			'invalid checkpoint or design provenance'
		);
		check(
			p.sourceHashes &&
				typeof p.sourceHashes === 'object' &&
				!Array.isArray(p.sourceHashes) &&
				Object.values(p.sourceHashes).every((v) => typeof v === 'string' && HASH.test(v)),
			'invalid implementation hashes'
		);
	}
	return record;
}
export function parseQueryStudy(text: string): QueryStudyRecord {
	if (text.length > 25_000_000) throw new Error('Query study exceeds the 25 MB import limit.');
	return validateQueryStudy(JSON.parse(text));
}
async function openDatabase(): Promise<IDBDatabase> {
	return new Promise((resolve, reject) => {
		const request = indexedDB.open(DATABASE, 1);
		request.onupgradeneeded = () => request.result.createObjectStore(STORE, { keyPath: 'id' });
		request.onsuccess = () => resolve(request.result);
		request.onerror = () =>
			reject(request.error ?? new Error('Could not open query study archive.'));
	});
}
export async function saveQueryStudy(record: QueryStudyRecord): Promise<void> {
	validateQueryStudy(record);
	const database = await openDatabase();
	try {
		await new Promise<void>((resolve, reject) => {
			const tx = database.transaction(STORE, 'readwrite');
			tx.objectStore(STORE).put(record);
			tx.oncomplete = () => resolve();
			tx.onerror = tx.onabort = () => reject(tx.error ?? new Error('Could not save query study.'));
		});
	} finally {
		database.close();
	}
}
export async function readQueryStudies(): Promise<{
	records: QueryStudyRecord[];
	warnings: string[];
}> {
	const database = await openDatabase();
	try {
		const values = await new Promise<unknown[]>((resolve, reject) => {
			const request = database.transaction(STORE).objectStore(STORE).getAll();
			request.onsuccess = () => resolve(request.result);
			request.onerror = () => reject(request.error ?? new Error('Could not read query studies.'));
		});
		const records: QueryStudyRecord[] = [];
		const warnings: string[] = [];
		for (const value of values) {
			try {
				records.push(validateQueryStudy(value));
			} catch (reason) {
				warnings.push(`${String(reason)}. The stored evidence was preserved.`);
			}
		}
		return {
			records: records.sort((a, b) => Date.parse(b.createdAt) - Date.parse(a.createdAt)),
			warnings
		};
	} finally {
		database.close();
	}
}
export function exportQueryStudy(record: QueryStudyRecord): void {
	validateQueryStudy(record);
	const url = URL.createObjectURL(new Blob([JSON.stringify(record)], { type: 'application/json' }));
	const link = document.createElement('a');
	link.href = url;
	link.download = `tissue-query-shifts-seed-${record.measurement.seed}-step-${record.measurement.step}.json`;
	link.click();
	setTimeout(() => URL.revokeObjectURL(url), 1000);
}
