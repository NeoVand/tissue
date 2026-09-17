import { publicAsset } from '../deployment/public-assets';
import {
	TOKEN_STORY_PRESETS,
	tokenStoryParameterCount,
	tokenStoryUnitCount,
	validateTokenStoryCheckpoint,
	type TokenStoryAtlas,
	type TokenStoryCheckpoint,
	type TokenStoryGeneration,
	type TokenStoryMetrics,
	type TokenStoryPreset,
	type TokenStoryTensor
} from './protocol';
import type { TokenStoryGeometryResult } from './geometry';
import { validateTokenStoryGeometry } from './geometry';
import {
	createTokenStoryTokenizer,
	validateTokenStoryTokenizerIdentity,
	type TokenStoryTokenizerData
} from './tokenizer';
import { encodeTokenStoryPrompt, TOKEN_STORY_CORPUS_ID, TOKEN_STORY_TOKENIZER_ID } from './dataset';

export interface TokenStoryRunRecord {
	version: 1;
	kind: 'tissue-token-story-run';
	tokenizer: TokenStoryTokenizerData;
	corpusId: string;
	id: string;
	createdAt: string;
	updatedAt: string;
	title: string;
	seed: number;
	presetId: TokenStoryPreset;
	metrics: TokenStoryMetrics[];
	snapshots: Array<{
		capturedAt: string;
		atlas: TokenStoryAtlas;
		geometry: TokenStoryGeometryResult;
	}>;
	checkpoint: TokenStoryCheckpoint | null;
	observations: Array<{ time: string; step: number; title: string; detail: string }>;
	samples?: TokenStoryGeneration[];
	interventions?: Array<{
		step: number;
		modelId: string;
		neuron: number;
		prompt: string;
		capturedAt: string;
		probabilities: Float32Array;
		lesionedProbabilities: Float32Array;
	}>;
	provenance?: {
		source: 'browser' | 'reference';
		commit?: string;
		workingTreeDirty?: boolean;
		sourceHashes?: Record<string, string>;
		browser?: string;
		parentRunId?: string;
	};
}

export interface TokenStoryRunSummary {
	id: string;
	title: string;
	createdAt: string;
	updatedAt: string;
	seed: number;
	presetId: TokenStoryPreset;
	checkpointStep: number | null;
	latestMetric: TokenStoryMetrics | null;
	snapshotCount: number;
	source: 'browser' | 'reference';
}

export interface TokenStoryReference {
	id: string;
	title: string;
	presetId: TokenStoryPreset;
	seed: number;
	step: number;
	parameterCount: number;
	unitCount: number;
	file: string;
	/** Ordered asset chunks; the checksum covers their concatenation. */
	parts?: string[];
	reportFile?: string;
	bytes: number;
	sha256: string;
	validationLoss: number;
	unigramLoss: number;
}

const DB = 'tissue-token-story-runs';
const MAGIC = new TextEncoder().encode('TISSUE-TOKEN-STORY-V1\n');
const MAX_BYTES = 512 * 1024 * 1024;
const MAX_HEADER_BYTES = 32 * 1024 * 1024;
const fail = (message: string): never => {
	throw new Error(`Invalid token-story run: ${message}`);
};
const check = (condition: unknown, message: string): void => {
	if (!condition) fail(message);
};
const finite = (x: unknown, min = -Infinity, max = Infinity): x is number =>
	typeof x === 'number' && Number.isFinite(x) && x >= min && x <= max;
const integer = (x: unknown, max = Number.MAX_SAFE_INTEGER): x is number =>
	finite(x, 0, max) && Number.isSafeInteger(x);
const date = (x: unknown): boolean => typeof x === 'string' && Number.isFinite(Date.parse(x));
const text = (x: unknown, max = 4096): x is string =>
	typeof x === 'string' && x.length > 0 && x.length <= max;
const sameConfig = (actual: unknown, preset: TokenStoryPreset): boolean =>
	!!actual &&
	typeof actual === 'object' &&
	Object.entries(TOKEN_STORY_PRESETS[preset]).every(
		([key, value]) => (actual as Record<string, unknown>)[key] === value
	);

function validateTensors(leaves: TokenStoryTensor[], expectedCount: number, deep: boolean): void {
	check(Array.isArray(leaves) && leaves.length > 0 && leaves.length < 128, 'invalid tensor list');
	let count = 0;
	for (const leaf of leaves) {
		check(
			leaf &&
				Array.isArray(leaf.shape) &&
				leaf.shape.length > 0 &&
				leaf.shape.length <= 4 &&
				leaf.shape.every((d) => integer(d, 16384) && d > 0),
			'invalid tensor shape'
		);
		const length = leaf.shape.reduce((a, b) => a * b, 1);
		check(
			leaf.values instanceof Float32Array && leaf.values.length === length,
			'tensor length mismatch'
		);
		if (deep) check(leaf.values.every(Number.isFinite), 'nonfinite checkpoint tensor');
		count += length;
	}
	check(count === expectedCount, 'parameter count does not match architecture');
}

/** Import performs full numeric validation; trusted worker saves can omit the costly float scan. */
export function validateTokenStoryRun(value: unknown, deep = true): TokenStoryRunRecord {
	check(!!value && typeof value === 'object', 'missing record');
	const r = value as TokenStoryRunRecord;
	const tokenizer = createTokenStoryTokenizer(r.tokenizer);
	check(
		r.corpusId === TOKEN_STORY_CORPUS_ID && tokenizer.id === TOKEN_STORY_TOKENIZER_ID,
		'invalid corpus or tokenizer identity'
	);
	check(r.version === 1 && r.kind === 'tissue-token-story-run', 'unsupported record version');
	check(text(r.id, 200) && text(r.title, 300), 'missing archive identity');
	check(date(r.createdAt) && date(r.updatedAt), 'invalid dates');
	check(['small', 'medium', 'large'].includes(r.presetId), 'unknown model preset');
	check(integer(r.seed, 0xffffffff), 'invalid initialization seed');
	const config = TOKEN_STORY_PRESETS[r.presetId];
	const units = tokenStoryUnitCount(config);
	check(
		Array.isArray(r.metrics) && r.metrics.length > 0 && r.metrics.length <= 100_000,
		'missing metrics'
	);
	let step = -1;
	let trainedTokens = 0;
	for (const m of r.metrics) {
		check(
			m &&
				integer(m.step) &&
				m.step >= step &&
				finite(m.validationLoss, 0) &&
				finite(m.validationAccuracy, 0, 1) &&
				finite(m.unigramLoss, 0) &&
				finite(m.unigramAccuracy, 0, 1) &&
				finite(m.uniformLoss, 0) &&
				(m.trainLoss === null || finite(m.trainLoss, 0)) &&
				integer(m.trainedTokens) &&
				integer(m.evaluationTokens) &&
				m.evaluationTokens > 0 &&
				finite(m.elapsedMs, 0) &&
				finite(m.stepMs, 0) &&
				['webgpu', 'wasm', 'cpu'].includes(m.backend),
			'invalid metrics'
		);
		check(
			m.trainedTokens >= trainedTokens &&
				m.trainedTokens <= m.step * config.batchSize * config.context,
			'trained token count disagrees with step'
		);
		step = m.step;
		trainedTokens = m.trainedTokens;
	}
	let modelId: string | undefined;
	if (r.checkpoint) {
		const c = r.checkpoint;
		check(
			c.version === 1 &&
				c.architecture === 'token-stories-transformer-v1' &&
				sameConfig(c.config, r.presetId),
			'incompatible checkpoint architecture'
		);
		check(
			text(c.modelId, 200) &&
				c.seed === r.seed &&
				integer(c.step) &&
				c.step <= step &&
				integer(c.trainRngState, 0xffffffff) &&
				finite(c.elapsedMs, 0) &&
				(c.trainLoss === null || finite(c.trainLoss, 0)),
			'invalid checkpoint state'
		);
		check(
			c.corpusId === r.corpusId && c.tokenizerId === tokenizer.id,
			'unknown corpus or tokenizer'
		);
		check(
			integer(c.trainedTokens) &&
				c.trainedTokens === r.metrics.find((m) => m.step === c.step)?.trainedTokens,
			'checkpoint supervised target count mismatch'
		);
		const count = tokenStoryParameterCount(config);
		validateTensors(c.parameters, count, deep);
		check(c.optimizer && c.optimizer.t === c.step, 'optimizer step mismatch');
		validateTensors(c.optimizer.m, count, deep);
		validateTensors(c.optimizer.v, count, deep);
		for (const moment of [c.optimizer.m, c.optimizer.v]) {
			check(
				moment.length === c.parameters.length &&
					moment.every((leaf, i) => leaf.shape.join(',') === c.parameters[i].shape.join(',')),
				'optimizer shapes mismatch'
			);
		}
		if (deep) validateTokenStoryCheckpoint(c);
		modelId = c.modelId;
	}
	check(Array.isArray(r.snapshots) && r.snapshots.length <= 1000, 'invalid snapshot history');
	let snapshotStep = -1;
	let calibrationIdentity: string | undefined;
	for (const snapshot of r.snapshots) {
		const a = snapshot?.atlas;
		check(
			date(snapshot?.capturedAt) &&
				a &&
				a.version === 1 &&
				sameConfig(a.config, r.presetId) &&
				a.seed === r.seed &&
				text(a.modelId, 200) &&
				(!modelId || a.modelId === modelId) &&
				integer(a.step) &&
				a.step >= snapshotStep &&
				a.step <= step &&
				a.unitCount === units &&
				a.dimensions === 128 &&
				a.corpusId === r.corpusId &&
				a.tokenizerId === tokenizer.id,
			'snapshot identity mismatch'
		);
		modelId = a.modelId;
		snapshotStep = a.step;
		check(
			a.fingerprints instanceof Float32Array && a.fingerprints.length === units * a.dimensions,
			'incomplete activation fingerprints'
		);
		if (deep)
			check(
				a.fingerprints.every((x) => finite(x, 0)),
				'invalid raw ReLU fingerprints'
			);
		check(
			Array.isArray(a.examples) &&
				a.examples.length === 8 &&
				Array.isArray(a.tokenIds) &&
				a.tokenIds.length === 8 &&
				a.tokenIds.every(
					(ids) =>
						Array.isArray(ids) &&
						ids.length > 0 &&
						ids.length <= config.context &&
						ids.every((id) => integer(id, tokenizer.vocabularySize - 1))
				) &&
				a.examples.every(
					(x, i) => typeof x === 'string' && x === tokenizer.decode(a.tokenIds[i])
				) &&
				Array.isArray(a.tokenPositions) &&
				a.tokenPositions.length === 8 &&
				a.tokenPositions.every(
					(positions, i) =>
						Array.isArray(positions) &&
						positions.length === 16 &&
						positions.every(
							(position, j) => position === Math.round((j * (a.tokenIds[i].length - 1)) / 15)
						)
				),
			'invalid calibration windows'
		);
		check(
			Array.isArray(a.positions) &&
				a.positions.length === 16 &&
				a.positions.every(
					(x, i) => integer(x, config.context - 1) && (i === 0 || x > a.positions[i - 1])
				),
			'invalid calibration positions'
		);
		const probeIdentity = JSON.stringify([a.examples, a.tokenIds, a.tokenPositions, a.positions]);
		check(
			!calibrationIdentity || probeIdentity === calibrationIdentity,
			'calibration probes changed within run'
		);
		calibrationIdentity = probeIdentity;
		check(
			date(a.capturedAt) && finite(a.elapsedMs, 0) && ['webgpu', 'wasm', 'cpu'].includes(a.backend),
			'invalid atlas metadata'
		);
		check(
			a.examples.every((example) => typeof example === 'string'),
			'invalid calibration characters'
		);
		check(
			a.positions.every((position, i) => position === Math.round((i * (config.context - 1)) / 15)),
			'unexpected calibration schedule'
		);
		const g = snapshot.geometry;
		validateTokenStoryGeometry(g, a);
		check(
			g &&
				g.version === 1 &&
				g.modelId === a.modelId &&
				g.corpusId === a.corpusId &&
				g.step === a.step &&
				g.unitCount === units &&
				g.dimensions === a.dimensions &&
				g.layerWidth === config.hidden &&
				Array.isArray(g.positions) &&
				g.positions.length === units &&
				g.positions.every((p) => Array.isArray(p) && p.length === 3 && p.every((x) => finite(x))),
			'invalid geometry positions'
		);
		check(
			Array.isArray(g.valid) &&
				g.valid.length === units &&
				g.valid.every((v) => typeof v === 'boolean') &&
				Array.isArray(g.magnitudes) &&
				g.magnitudes.length === units &&
				g.magnitudes.every((v) => finite(v, 0)) &&
				Array.isArray(g.edges) &&
				g.edges.length <= units * 128 &&
				g.edges.every(
					(e) => Array.isArray(e) && e.length === 2 && e.every((id) => integer(id, units - 1))
				) &&
				g.edgeScope === 'audit-sample' &&
				finite(g.explainedVariance, 0, 1.000001) &&
				Array.isArray(g.explainedVarianceByAxis) &&
				g.explainedVarianceByAxis.length === 3 &&
				g.explainedVarianceByAxis.every((v) => finite(v, 0, 1.000001)) &&
				g.audit &&
				Array.isArray(g.audit.ids) &&
				g.audit.ids.every((id) => integer(id, units - 1)) &&
				Array.isArray(g.audit.perFocal) &&
				(g.audit.neighborRetention === null || finite(g.audit.neighborRetention, 0, 1)) &&
				g.pca &&
				finite(g.pca.relativeResidual, 0) &&
				typeof g.pca.converged === 'boolean',
			'invalid projection diagnostics'
		);
	}
	check(
		Array.isArray(r.observations) &&
			r.observations.length <= 100_000 &&
			r.observations.every(
				(o) =>
					o &&
					date(o.time) &&
					integer(o.step) &&
					o.step <= step &&
					text(o.title, 300) &&
					text(o.detail, 20_000)
			),
		'invalid observations'
	);
	if (r.samples) {
		check(Array.isArray(r.samples) && r.samples.length <= 10_000, 'invalid generation history');
		for (const s of r.samples) {
			check(
				s &&
					text(s.modelId, 200) &&
					(!modelId || s.modelId === modelId) &&
					integer(s.step) &&
					s.step <= step,
				'generation identity mismatch'
			);
			modelId = s.modelId;
			check(s.prompt && typeof s.prompt.original === 'string', 'missing generation prompt');
			const encoded = encodeTokenStoryPrompt(s.prompt.original, config.context, tokenizer);
			check(
				s.tokenizerId === tokenizer.id &&
					s.prompt.text === encoded.text &&
					s.prompt.includesBos === encoded.includesBos &&
					JSON.stringify(s.prompt.pieces) === JSON.stringify(encoded.pieces) &&
					s.prompt.truncatedTokens === encoded.truncatedTokens &&
					JSON.stringify(s.prompt.tokenIds) === JSON.stringify(encoded.tokenIds) &&
					Array.isArray(s.prompt.unsupportedCharacters) &&
					s.prompt.unsupportedCharacters.length === 0,
				'invalid generation tokenization'
			);
			check(
				integer(s.requestedTokens, 256) &&
					s.requestedTokens > 0 &&
					integer(s.samplingSeed, 0xffffffff) &&
					finite(s.temperature, 0.05, 2) &&
					integer(s.topK, tokenizer.vocabularySize) &&
					s.topK > 0 &&
					typeof s.cancelled === 'boolean' &&
					typeof s.stoppedOnEos === 'boolean',
				'invalid sampling settings'
			);
			check(
				Array.isArray(s.tokenIds) &&
					s.tokenIds.length <= s.requestedTokens &&
					s.tokenIds.every(
						(id, i) =>
							integer(id, tokenizer.vocabularySize - 1) &&
							id !== tokenizer.bosId &&
							(id !== tokenizer.eosId || i === s.tokenIds.length - 1)
					) &&
					s.completion === tokenizer.decode(s.tokenIds) &&
					JSON.stringify(s.pieces) ===
						JSON.stringify(s.tokenIds.map((id) => tokenizer.tokenPiece(id))) &&
					s.stoppedOnEos === (s.tokenIds.at(-1) === tokenizer.eosId) &&
					(s.cancelled ||
						s.tokenIds.length === s.requestedTokens ||
						s.tokenIds.at(-1) === tokenizer.eosId),
				'invalid generated tokens'
			);
		}
	}
	if (r.interventions) {
		check(
			Array.isArray(r.interventions) && r.interventions.length <= 10_000,
			'invalid intervention history'
		);
		for (const event of r.interventions) {
			check(
				event &&
					text(event.modelId, 200) &&
					(!modelId || event.modelId === modelId) &&
					integer(event.step) &&
					event.step <= step &&
					integer(event.neuron, units - 1) &&
					date(event.capturedAt),
				'invalid intervention identity'
			);
			modelId = event.modelId;
			encodeTokenStoryPrompt(event.prompt, config.context, tokenizer);
			for (const probabilities of [event.probabilities, event.lesionedProbabilities]) {
				check(
					probabilities instanceof Float32Array &&
						probabilities.length === tokenizer.vocabularySize &&
						probabilities.every((v) => finite(v, 0, 1)) &&
						Math.abs(probabilities.reduce((sum, v) => sum + v, 0) - 1) < 1e-4,
					'invalid intervention probabilities'
				);
			}
		}
	}
	if (r.provenance) {
		check(['browser', 'reference'].includes(r.provenance.source), 'invalid provenance source');
		if (r.provenance.sourceHashes)
			check(
				Object.entries(r.provenance.sourceHashes).every(
					([key, value]) => text(key, 1024) && /^[a-f0-9]{64}$/.test(value)
				),
				'invalid source hashes'
			);
	}
	return r;
}

export function summarizeTokenStoryRun(record: TokenStoryRunRecord): TokenStoryRunSummary {
	return {
		id: record.id,
		title: record.title,
		createdAt: record.createdAt,
		updatedAt: record.updatedAt,
		seed: record.seed,
		presetId: record.presetId,
		checkpointStep: record.checkpoint?.step ?? null,
		latestMetric: record.metrics.at(-1) ?? null,
		snapshotCount: record.snapshots.length,
		source: record.provenance?.source ?? 'browser'
	};
}

function openDatabase(): Promise<IDBDatabase> {
	return new Promise((resolve, reject) => {
		const request = indexedDB.open(DB, 1);
		request.onupgradeneeded = () => {
			request.result.createObjectStore('runs', { keyPath: 'id' });
			request.result.createObjectStore('summaries', { keyPath: 'id' });
		};
		request.onsuccess = () => resolve(request.result);
		request.onerror = () =>
			reject(request.error ?? new Error('Could not open token-story archive'));
	});
}

export async function saveTokenStoryRun(record: TokenStoryRunRecord): Promise<void> {
	validateTokenStoryRun(record, false);
	const database = await openDatabase();
	try {
		await new Promise<void>((resolve, reject) => {
			const tx = database.transaction(['runs', 'summaries'], 'readwrite');
			tx.objectStore('runs').put(record);
			tx.objectStore('summaries').put(summarizeTokenStoryRun(record));
			tx.oncomplete = () => resolve();
			tx.onerror = tx.onabort = () =>
				reject(tx.error ?? new Error('Could not save token-story evidence'));
		});
	} finally {
		database.close();
	}
}

async function readStore<T>(store: string, id?: string): Promise<T> {
	const database = await openDatabase();
	try {
		return await new Promise<T>((resolve, reject) => {
			const source = database.transaction(store).objectStore(store);
			const request = id === undefined ? source.getAll() : source.get(id);
			request.onsuccess = () => resolve(request.result);
			request.onerror = () =>
				reject(request.error ?? new Error('Could not read token-story archive'));
		});
	} finally {
		database.close();
	}
}

let archiveWarnings: string[] = [];
export const getTokenStoryArchiveWarnings = (): string[] => [...archiveWarnings];
export async function listTokenStoryRuns(): Promise<TokenStoryRunSummary[]> {
	archiveWarnings = [];
	const summaries = await readStore<TokenStoryRunSummary[]>('summaries');
	return summaries
		.filter((s) => {
			const valid =
				s &&
				text(s.id, 200) &&
				text(s.title, 300) &&
				date(s.createdAt) &&
				date(s.updatedAt) &&
				integer(s.seed, 0xffffffff) &&
				['small', 'medium', 'large'].includes(s.presetId) &&
				(s.checkpointStep === null || integer(s.checkpointStep)) &&
				integer(s.snapshotCount, 1000) &&
				['browser', 'reference'].includes(s.source) &&
				(!s.latestMetric ||
					(integer(s.latestMetric.step) &&
						finite(s.latestMetric.validationLoss, 0) &&
						finite(s.latestMetric.unigramLoss, 0) &&
						finite(s.latestMetric.trainedTokens, 0)));
			if (!valid)
				archiveWarnings.push(
					'An invalid saved-run summary was hidden. Its underlying archive remains untouched.'
				);
			return valid;
		})
		.sort((a, b) => Date.parse(b.updatedAt) - Date.parse(a.updatedAt));
}
export async function loadTokenStoryRun(id: string): Promise<TokenStoryRunRecord | null> {
	const value = await readStore<unknown>('runs', id);
	if (!value) return null;
	const record = validateTokenStoryRun(value);
	await validateTokenStoryTokenizerIdentity(record.tokenizer);
	return record;
}

/** Catalog downloads are pinned to their recorded bytes before becoming local evidence. */
export async function loadTokenStoryReference(
	reference: TokenStoryReference
): Promise<TokenStoryRunRecord> {
	check(
		reference &&
			/^\/experiments\/token-stories-[a-z0-9-]+\.tissue$/.test(reference.file) &&
			/^[a-f0-9]{64}$/.test(reference.sha256) &&
			integer(reference.bytes, MAX_BYTES) &&
			['small', 'medium', 'large'].includes(reference.presetId),
		'invalid reference catalog entry'
	);
	const paths = reference.parts ?? [reference.file];
	check(
		Array.isArray(paths) &&
			paths.length > 0 &&
			paths.length <= 32 &&
			paths.every(
				(path, i) => path === (reference.parts ? `${reference.file}.part-${i}` : reference.file)
			),
		'invalid reference parts'
	);
	const chunks: ArrayBuffer[] = [];
	let total = 0;
	for (const path of paths) {
		const response = await fetch(publicAsset(path), { signal: AbortSignal.timeout(60000) });
		if (!response.ok) throw new Error('The recorded token-story model could not be loaded.');
		const chunk = await response.arrayBuffer();
		total += chunk.byteLength;
		check(total <= reference.bytes, 'reference exceeds declared size');
		chunks.push(chunk);
	}
	const buffer = chunks.length === 1 ? chunks[0] : await new Blob(chunks).arrayBuffer();
	check(buffer.byteLength === reference.bytes, 'reference byte count mismatch');
	const digest = await crypto.subtle.digest('SHA-256', buffer);
	const sha = Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join(
		''
	);
	check(sha === reference.sha256, 'reference checksum mismatch');
	const record = decodeTokenStoryRun(buffer);
	await validateTokenStoryTokenizerIdentity(record.tokenizer);
	check(
		record.presetId === reference.presetId &&
			record.seed === reference.seed &&
			record.metrics.at(-1)?.step === reference.step &&
			tokenStoryParameterCount(TOKEN_STORY_PRESETS[record.presetId]) === reference.parameterCount &&
			tokenStoryUnitCount(TOKEN_STORY_PRESETS[record.presetId]) === reference.unitCount,
		'reference model identity mismatch'
	);
	return record;
}

/** Packed typed arrays avoid turning a 129 MB optimizer checkpoint into millions of JSON numbers. */
export function encodeTokenStoryRun(record: TokenStoryRunRecord): Blob {
	validateTokenStoryRun(record, false);
	const arrays: Uint8Array<ArrayBuffer>[] = [];
	const header = new TextEncoder().encode(
		JSON.stringify(record, (_, value) => {
			if (value instanceof Float32Array) {
				const bytes = new Uint8Array(value.buffer, value.byteOffset, value.byteLength).slice();
				const index = arrays.push(bytes) - 1;
				return { __tissueFloat32: index, length: value.length };
			}
			return value;
		})
	);
	check(header.length <= MAX_HEADER_BYTES, 'metadata exceeds file limit');
	const size = new Uint8Array(4);
	new DataView(size.buffer).setUint32(0, header.length, true);
	const blob = new Blob([MAGIC, size, header, ...arrays], { type: 'application/octet-stream' });
	check(blob.size <= MAX_BYTES, 'archive exceeds 512 MB limit');
	return blob;
}

export function decodeTokenStoryRun(buffer: ArrayBuffer): TokenStoryRunRecord {
	check(
		buffer.byteLength <= MAX_BYTES && buffer.byteLength >= MAGIC.length + 4,
		'invalid file size'
	);
	const bytes = new Uint8Array(buffer);
	check(
		MAGIC.every((byte, i) => bytes[i] === byte),
		'expected a .tissue token-story archive'
	);
	const length = new DataView(buffer).getUint32(MAGIC.length, true);
	check(
		length > 0 && length <= MAX_HEADER_BYTES && MAGIC.length + 4 + length <= buffer.byteLength,
		'invalid metadata length'
	);
	const start = MAGIC.length + 4;
	let offset = start + length;
	let index = 0;
	const parsed = JSON.parse(
		new TextDecoder('utf-8', { fatal: true }).decode(bytes.subarray(start, offset)),
		(key, value) => {
			if (['__proto__', 'constructor', 'prototype'].includes(key)) fail('unsafe metadata key');
			if (value && typeof value === 'object' && '__tissueFloat32' in value) {
				check(
					value.__tissueFloat32 === index++ &&
						integer(value.length, MAX_BYTES / 4) &&
						offset + value.length * 4 <= buffer.byteLength,
					'invalid packed tensor'
				);
				const data = buffer.slice(offset, offset + value.length * 4);
				offset += value.length * 4;
				return new Float32Array(data);
			}
			return value;
		}
	);
	check(offset === buffer.byteLength, 'unexpected trailing bytes');
	return validateTokenStoryRun(parsed);
}

export async function importTokenStoryRun(file: File): Promise<TokenStoryRunRecord> {
	check(file.size <= MAX_BYTES, 'archive exceeds 512 MB limit');
	const record = decodeTokenStoryRun(await file.arrayBuffer());
	await validateTokenStoryTokenizerIdentity(record.tokenizer);
	return record;
}
export function exportTokenStoryRun(record: TokenStoryRunRecord): void {
	const url = URL.createObjectURL(encodeTokenStoryRun(record));
	const link = document.createElement('a');
	link.href = url;
	link.download = `tissue-token-stories-${record.presetId}-seed-${record.seed}-step-${record.checkpoint?.step ?? 'observed'}.tissue`;
	link.click();
	setTimeout(() => URL.revokeObjectURL(url), 1000);
}
