/** Independent packed-archive/provenance audit; does not execute a model or refit PCA.
 * Node 22.18+ (native TypeScript stripping): node scripts/audit-stories.mjs
 * Add --full-archives to require and verify the complete local runner checkpoints.
 * Run --self-test to exercise malformed binary rejection without recorded studies.
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve, relative, sep } from 'node:path';
import {
	STORY_PRESETS,
	STORY_CHARACTERS,
	storyParameterCount,
	storyUnitCount,
	validateStoryCheckpoint
} from '../src/lib/stories/protocol.ts';

const args = process.argv.slice(2);
const option = (name, fallback) => {
	const index = args.indexOf(name);
	if (index < 0) return fallback;
	assert(args[index + 1] && !args[index + 1].startsWith('--'), `${name} requires a value`);
	return args[index + 1];
};
const sha = (value) => createHash('sha256').update(value).digest('hex');
const json = async (path) => JSON.parse(await readFile(path, 'utf8'));
const magic = Buffer.from('TISSUE-STORY-V1\n');
const maximumBytes = 512 * 1024 * 1024;
const corpusHash = '67fb35357ec45562152b012c60d4a2d1aec5a95ad9d9a4fc6d534bc97fb4c6db';
const corpusId = `tinystories-ascii-v1:${corpusHash}`;
const finite = (value, min = -Infinity, max = Infinity) =>
	assert(Number.isFinite(value) && value >= min && value <= max, `Invalid number ${value}`);
const integer = (value, min = 0, max = Number.MAX_SAFE_INTEGER) => {
	finite(value, min, max);
	assert(Number.isSafeInteger(value), `Invalid integer ${value}`);
};
const close = (actual, expected, label, tolerance = 1e-12) =>
	assert(
		Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)),
		`${label}: ${actual} differs from ${expected}`
	);
const safeRepoPath = (path) => {
	assert(typeof path === 'string' && path.length > 0);
	const target = resolve(path),
		local = relative(process.cwd(), target);
	assert(local && local !== '..' && !local.startsWith(`..${sep}`), `Outside repository: ${path}`);
	return target;
};
const assetPath = (path) => {
	assert(/^\/experiments\/stories-[a-zA-Z0-9.-]+$/.test(path), `Invalid asset path: ${path}`);
	return safeRepoPath(`static${path}`);
};

/** Parse independently of the browser decoder, including ordinal, bounds, and trailing-byte checks. */
function decodeArchive(bytes) {
	assert(bytes.length >= magic.length + 4 && bytes.length <= maximumBytes, 'Archive size');
	assert(bytes.subarray(0, magic.length).equals(magic), 'Archive magic');
	const length = bytes.readUInt32LE(magic.length),
		start = magic.length + 4;
	assert(
		length > 0 && length <= 32 * 1024 * 1024 && start + length <= bytes.length,
		'Header length'
	);
	let offset = start + length,
		ordinal = 0;
	const value = JSON.parse(
		new TextDecoder('utf-8', { fatal: true }).decode(bytes.subarray(start, offset)),
		(key, item) => {
			assert(!['__proto__', 'constructor', 'prototype'].includes(key), 'Unsafe metadata key');
			if (item && typeof item === 'object' && '__tissueFloat32' in item) {
				assert.equal(item.__tissueFloat32, ordinal++, 'Packed tensor order');
				assert.deepEqual(Object.keys(item).sort(), ['__tissueFloat32', 'length']);
				integer(item.length, 0, maximumBytes / 4);
				const end = offset + item.length * 4;
				assert(end <= bytes.length, 'Packed tensor exceeds file');
				const copy = bytes.buffer.slice(bytes.byteOffset + offset, bytes.byteOffset + end);
				offset = end;
				return new Float32Array(copy);
			}
			return item;
		}
	);
	assert.equal(offset, bytes.length, 'Unexpected trailing bytes');
	return value;
}
function selfTest() {
	const pack = (header, data = Buffer.from(new Float32Array([1, 2]).buffer)) => {
		const text = Buffer.from(JSON.stringify(header)),
			length = Buffer.alloc(4);
		length.writeUInt32LE(text.length);
		return Buffer.concat([magic, length, text, data]);
	};
	const valid = pack({ tensor: { __tissueFloat32: 0, length: 2 } });
	assert.deepEqual(decodeArchive(valid).tensor, new Float32Array([1, 2]));
	assert.throws(() => decodeArchive(valid.subarray(0, valid.length - 1)), /exceeds file/);
	assert.throws(() => decodeArchive(Buffer.concat([valid, Buffer.alloc(1)])), /trailing bytes/);
	assert.throws(() => decodeArchive(pack({ tensor: { __tissueFloat32: 1, length: 2 } })), /order/);
	assert.throws(
		() => decodeArchive(pack({ tensor: { __tissueFloat32: 0, length: -1 } })),
		/number/
	);
	assert.throws(
		() => decodeArchive(pack(JSON.parse('{"__proto__":{}}'), Buffer.alloc(0))),
		/Unsafe/
	);
	console.log('Story binary audit self-test passed (round trip + five corruption cases).');
}

function random(seed) {
	let state = seed >>> 0;
	return () => {
		state = (state + 0x6d2b79f5) >>> 0;
		let x = Math.imul(state ^ (state >>> 15), state | 1);
		x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
		return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
	};
}
async function verifyCorpus() {
	const path = 'static/data/tinystories/provenance.json',
		provenance = await json(path);
	assert.equal(provenance.id, corpusId);
	for (const name of ['corpus.json', 'tokens.bin', 'LICENSE.html']) {
		const bytes = await readFile(`static/data/tinystories/${name}`);
		assert.equal(bytes.length, provenance.files[name].bytes, `Corpus size: ${name}`);
		assert.equal(sha(bytes), provenance.files[name].sha256, `Corpus checksum: ${name}`);
	}
	const bytes = await readFile('static/data/tinystories/tokens.bin');
	assert.equal(sha(bytes), corpusHash);
	const metadata = await json('static/data/tinystories/corpus.json');
	assert.deepEqual(metadata.chars, STORY_CHARACTERS);
	const decode = (tokens) => Array.from(tokens, (id) => STORY_CHARACTERS[id]).join('');
	const stories = decode(bytes.subarray(1773785)).split('\n\n').filter(Boolean);
	assert.equal(stories.length, 184);
	const windows = (source, count, seed) => {
		const next = random(seed);
		return Array.from({ length: count }, (_, i) => {
			const story = source[Math.floor((i * source.length) / count)];
			const start = Math.floor(next() * (story.length - 128));
			return story.slice(start, start + 129);
		});
	};
	const calibration = windows(stories.slice(0, 32), 8, 0x73746361).map((text) =>
		text.slice(0, 128)
	);
	const evaluation = windows(stories.slice(32), 16, 0x73746576);
	const frequencies = new Float64Array(96).fill(1);
	for (const id of bytes.subarray(0, 1773785)) frequencies[id]++;
	for (let id = 0; id < 96; id++) frequencies[id] /= 1773785 + 96;
	let best = 0,
		loss = 0,
		correct = 0;
	for (let id = 1; id < 96; id++) if (frequencies[id] > frequencies[best]) best = id;
	for (const window of evaluation)
		for (const char of window.slice(1)) {
			const target = STORY_CHARACTERS.indexOf(char);
			loss -= Math.log(frequencies[target]);
			correct += +(target === best);
		}
	return {
		calibration,
		unigramLoss: loss / 2048,
		unigramAccuracy: correct / 2048,
		provenanceHash: sha(await readFile(path))
	};
}

const requiredSources = [
	'src/lib/stories/protocol.ts',
	'src/lib/stories/dataset.ts',
	'src/lib/stories/model.ts',
	'src/lib/stories/runtime.ts',
	'src/lib/stories/worker.ts',
	'src/lib/stories/engine.ts',
	'src/lib/stories/geometry.ts',
	'src/lib/stories/geometry-worker.ts',
	'src/lib/stories/geometry-engine.ts',
	'src/lib/stories/archive.ts',
	'src/lib/stories/validate-geometry.ts',
	'src/lib/lab/model/adam.ts',
	'src/lib/lab/geometry.ts',
	'pnpm-lock.yaml',
	'docs/stories-design.md',
	'scripts/measure-stories.mjs',
	'static/data/tinystories/provenance.json'
];
async function verifyProvenance(provenance) {
	assert.equal(provenance.source, 'reference');
	assert.match(provenance.commit, /^[a-f0-9]{40}$/);
	assert.equal(typeof provenance.workingTreeDirty, 'boolean');
	assert.equal(typeof provenance.browser, 'string');
	for (const path of requiredSources)
		assert(path in provenance.sourceHashes, `Missing source hash: ${path}`);
	for (const [path, expected] of Object.entries(provenance.sourceHashes)) {
		assert.match(expected, /^[a-f0-9]{64}$/);
		assert.equal(sha(await readFile(safeRepoPath(path))), expected, `Current source hash: ${path}`);
	}
}
function verifyRecord(record, corpus) {
	assert.equal(record.version, 1);
	assert.equal(record.kind, 'tissue-story-run');
	const config = STORY_PRESETS[record.presetId];
	assert(config, 'Unknown preset');
	integer(record.seed, 0, 0xffffffff);
	const units = storyUnitCount(config),
		count = storyParameterCount(config);
	assert(record.metrics.length > 0 && record.snapshots.length > 0);
	let previous = -1;
	for (const metric of record.metrics) {
		integer(metric.step);
		assert(metric.step > previous, 'Duplicate or unordered metrics');
		previous = metric.step;
		assert.equal(metric.trainedTokens, metric.step * config.context * config.batchSize);
		assert.equal(metric.evaluationTokens, 2048);
		finite(metric.validationLoss, 0);
		finite(metric.validationAccuracy, 0, 1);
		close(
			metric.validationAccuracy * 2048,
			Math.round(metric.validationAccuracy * 2048),
			'Integer accuracy'
		);
		if (metric.trainLoss !== null) finite(metric.trainLoss, 0);
		finite(metric.elapsedMs, 0);
		finite(metric.stepMs, 0);
		assert(['webgpu', 'wasm', 'cpu'].includes(metric.backend));
		close(metric.unigramLoss, corpus.unigramLoss, 'Training unigram held-out CE');
		close(metric.unigramAccuracy, corpus.unigramAccuracy, 'Training unigram held-out accuracy');
		close(metric.uniformLoss, Math.log(96), 'Uniform CE');
	}
	const final = record.metrics.at(-1),
		modelId = record.snapshots[0].atlas.modelId;
	previous = -1;
	for (const { atlas, geometry } of record.snapshots) {
		assert.equal(atlas.version, 1);
		assert.equal(atlas.modelId, modelId);
		assert.equal(atlas.seed, record.seed);
		assert.equal(atlas.corpusId, corpusId);
		assert.deepEqual(atlas.config, config);
		assert.equal(atlas.unitCount, units);
		assert.equal(atlas.dimensions, 128);
		integer(atlas.step);
		assert(atlas.step > previous && atlas.step <= final.step);
		previous = atlas.step;
		assert(
			record.metrics.some((m) => m.step === atlas.step),
			'Atlas has no matching metrics'
		);
		assert.deepEqual(atlas.examples, corpus.calibration);
		assert.deepEqual(
			atlas.positions,
			Array.from({ length: 16 }, (_, i) => Math.round((i * 127) / 15))
		);
		assert(atlas.fingerprints instanceof Float32Array);
		assert.equal(atlas.fingerprints.length, units * 128);
		assert.equal(geometry.modelId, modelId);
		assert.equal(geometry.step, atlas.step);
		assert.equal(geometry.corpusId, corpusId);
		assert.equal(geometry.unitCount, units);
		assert.equal(geometry.dimensions, 128);
		assert.equal(geometry.layerWidth, config.hidden);
		assert.equal(geometry.rmsFloor, 1e-8);
		assert.equal(geometry.positions.length, units);
		assert.equal(geometry.magnitudes.length, units);
		assert.equal(geometry.valid.length, units);
		for (let unit = 0; unit < units; unit++) {
			const row = atlas.fingerprints.subarray(unit * 128, (unit + 1) * 128);
			let mean = 0,
				squared = 0;
			for (const value of row) {
				finite(value, 0);
				mean += value / 128;
			}
			for (const value of row) squared += (value - mean) ** 2;
			const magnitude = Math.sqrt(squared / 128);
			close(geometry.magnitudes[unit], magnitude, 'Activation variation');
			assert.equal(geometry.valid[unit], magnitude > geometry.rmsFloor);
			assert.equal(geometry.positions[unit].length, 3);
			geometry.positions[unit].forEach((x) => finite(x));
		}
		assert.equal(geometry.audit.validPopulation, geometry.valid.filter(Boolean).length);
		const scored = geometry.audit.perFocal.filter((focal) => focal.retention !== null);
		assert.equal(geometry.audit.scoredFocals, scored.length);
		for (const focal of scored) finite(focal.retention, 0, 1);
		if (scored.length)
			close(
				geometry.audit.neighborRetention,
				scored.reduce((sum, f) => sum + f.retention, 0) / scored.length,
				'Focal retention mean'
			);
		else assert.equal(geometry.audit.neighborRetention, null);
		close(
			geometry.explainedVariance,
			geometry.explainedVarianceByAxis.reduce((a, b) => a + b, 0),
			'Projection variance sum'
		);
		finite(geometry.explainedVariance, 0, 1.000001);
		assert.equal(geometry.pca.converged, geometry.pca.relativeResidual <= geometry.pca.tolerance);
	}
	if (record.checkpoint) {
		validateStoryCheckpoint(record.checkpoint);
		assert.equal(record.checkpoint.modelId, modelId);
		assert.equal(record.checkpoint.corpus.id, corpusId);
		assert.equal(record.checkpoint.seed, record.seed);
		assert.deepEqual(record.checkpoint.config, config);
		assert.equal(record.checkpoint.step, final.step);
		assert.equal(record.checkpoint.trainLoss, final.trainLoss);
	}
	for (const sample of record.samples ?? []) {
		assert.equal(sample.modelId, modelId);
		integer(sample.step, 0, final.step);
		assert.deepEqual(
			sample.prompt.tokenIds.map((id) => STORY_CHARACTERS[id]).join(''),
			sample.prompt.text
		);
		assert.equal(sample.prompt.text, sample.prompt.original.slice(-128));
		assert.equal(
			sample.prompt.truncatedCharacters,
			Math.max(0, sample.prompt.original.length - 128)
		);
		assert.deepEqual(sample.prompt.unsupportedCharacters, []);
		sample.tokenIds.forEach((id) => integer(id, 0, 95));
		assert.equal(sample.tokenIds.map((id) => STORY_CHARACTERS[id]).join(''), sample.completion);
		assert.equal(sample.cancelled, sample.tokenIds.length < sample.requestedTokens);
	}
	for (const intervention of record.interventions ?? []) {
		assert.equal(intervention.modelId, modelId);
		integer(intervention.step, 0, final.step);
		integer(intervention.neuron, 0, units - 1);
		for (const row of [intervention.probabilities, intervention.lesionedProbabilities]) {
			assert(row instanceof Float32Array);
			assert.equal(row.length, 96);
			row.forEach((v) => finite(v, 0, 1));
			close(
				row.reduce((sum, v) => sum + v, 0),
				1,
				'Intervention probability mass',
				1e-4
			);
		}
	}
	return { units, count, final, modelId };
}

function verifyRestoration(receipt, record, report, intervention) {
	assert.equal(receipt.version, 1);
	assert.equal(record.presetId, 'large');
	assert.equal(receipt.modelId, intervention.modelId, 'Restoration model identity');
	assert.equal(receipt.step, intervention.step, 'Restoration checkpoint step');
	assert.equal(receipt.prompt, intervention.prompt, 'Restoration prompt');
	assert.equal(receipt.unit, intervention.neuron, 'Restoration unit');
	assert.equal(receipt.backend, record.metrics.at(-1).backend, 'Restoration backend');
	assert.equal(
		receipt.sourceArchiveSHA256,
		report.fullArchive.sha256,
		'Restoration checkpoint SHA'
	);
	assert.deepEqual(
		receipt.sourceHashes,
		record.provenance.sourceHashes,
		'Restoration source hashes'
	);
	assert.equal(receipt.original.capturedAt, intervention.capturedAt);
	for (const time of [receipt.recordedAt, receipt.original.capturedAt, receipt.restored.capturedAt])
		assert(Number.isFinite(Date.parse(time)), 'Invalid restoration timestamp');
	assert(typeof receipt.browser === 'string' && receipt.browser.length > 0);
	for (const group of [receipt.original, receipt.restored])
		for (const row of [group.intact, group.lesioned]) {
			assert(Array.isArray(row));
			assert.equal(row.length, 96, 'Restoration probability count');
			row.forEach((value) => finite(value, 0, 1));
			close(
				row.reduce((sum, value) => sum + value, 0),
				1,
				'Restoration probability mass',
				1e-6
			);
		}
	assert.deepEqual(receipt.original.intact, Array.from(intervention.probabilities));
	assert.deepEqual(receipt.original.lesioned, Array.from(intervention.lesionedProbabilities));
	const difference = (a, b) => Math.max(...a.map((value, i) => Math.abs(value - b[i])));
	const differences = {
		maxIntactDifference: difference(receipt.original.intact, receipt.restored.intact),
		maxLesionedDifference: difference(receipt.original.lesioned, receipt.restored.lesioned),
		originalMaxEffect: difference(receipt.original.intact, receipt.original.lesioned),
		restoredMaxEffect: difference(receipt.restored.intact, receipt.restored.lesioned)
	};
	for (const [key, value] of Object.entries(differences))
		assert.equal(receipt[key], value, `Restoration ${key}`);
	assert.equal(
		receipt.originalLoss,
		report.metrics.at(-1).validationLoss,
		'Restoration original loss'
	);
	finite(receipt.restoredLoss, 0);
	return {
		modelId: receipt.modelId,
		step: receipt.step,
		prompt: receipt.prompt,
		unit: receipt.unit,
		backend: receipt.backend,
		sourceArchiveSHA256: receipt.sourceArchiveSHA256,
		...differences,
		originalLoss: receipt.originalLoss,
		restoredLoss: receipt.restoredLoss,
		limitation:
			'One same-checkpoint cross-worker comparison; not a calibrated numerical noise floor.'
	};
}

async function audit() {
	const corpus = await verifyCorpus();
	const catalogPath = option('--catalog', 'static/experiments/stories-index.json'),
		catalog = await json(catalogPath);
	assert.equal(catalog.version, 1);
	assert(catalog.references.length > 0, 'No published story references to audit');
	assert.equal(
		new Set(catalog.references.map((r) => r.id)).size,
		catalog.references.length,
		'Duplicate catalog IDs'
	);
	const restorationPath = 'static/experiments/stories-large-restoration.json';
	let restorationReceipt;
	try {
		restorationReceipt = await json(restorationPath);
	} catch (error) {
		if (error.code !== 'ENOENT') throw error;
	}
	let restoration = null;
	const runs = [];
	for (const reference of catalog.references) {
		const paths = reference.parts ?? [reference.file];
		assert(paths.length > 0 && new Set(paths).size === paths.length, 'Invalid archive parts');
		const parts = await Promise.all(
			paths.map(async (path) => {
				const bytes = await readFile(assetPath(path));
				if (reference.parts)
					assert(
						bytes.length > 0 && bytes.length <= 20 * 1024 * 1024,
						'Archive part exceeds budget'
					);
				return { path, bytes };
			})
		);
		const bytes = Buffer.concat(parts.map((part) => part.bytes));
		assert.equal(bytes.length, reference.bytes, 'Catalog archive size');
		assert.equal(sha(bytes), reference.sha256, 'Catalog archive checksum');
		const record = decodeArchive(bytes),
			checked = verifyRecord(record, corpus);
		await verifyProvenance(record.provenance);
		for (const key of ['id', 'title', 'presetId', 'seed'])
			assert.equal(reference[key], record[key], `Catalog ${key}`);
		assert.equal(reference.step, checked.final.step);
		assert.equal(reference.parameterCount, checked.count);
		assert.equal(reference.unitCount, checked.units);
		assert.equal(reference.validationLoss, checked.final.validationLoss);
		assert.equal(reference.unigramLoss, checked.final.unigramLoss);
		const reportPath = reference.reportFile
			? assetPath(reference.reportFile)
			: `static/experiments/stories-${reference.presetId}-report.json`;
		const report = await json(reportPath);
		for (const key of ['id', 'title', 'presetId', 'seed'])
			assert.equal(report[key], record[key], `Report ${key}`);
		assert.deepEqual(report.published, reference);
		assert.deepEqual(report.metrics, record.metrics);
		assert.deepEqual(report.provenance, record.provenance);
		assert.deepEqual(report.generation, record.samples.at(-1));
		const last = record.snapshots.at(-1).geometry;
		assert.deepEqual(report.projection, {
			validUnits: last.valid.filter(Boolean).length,
			variance: last.explainedVariance,
			retention: last.audit.neighborRetention,
			audit: last.audit,
			pca: last.pca
		});
		assert.equal(report.timings.length, record.snapshots.length);
		for (let i = 0; i < report.timings.length; i++) {
			assert.equal(report.timings[i].step, record.snapshots[i].atlas.step);
			assert.equal(report.timings[i].atlasMs, record.snapshots[i].atlas.elapsedMs);
			finite(report.timings[i].geometryMs, 0);
			finite(report.timings[i].captureAndCheckpointMs, 0);
		}
		const intervention = record.interventions?.find(
			(event) => event.step === checked.final.step && event.neuron === report.probe.unit
		);
		assert(intervention, 'Missing raw intervention for reported probability difference');
		assert.equal(
			report.probe.unit,
			Math.floor(checked.units / 2),
			'Prespecified intervention unit'
		);
		const difference = Math.max(
			...Array.from(intervention.probabilities, (p, i) =>
				Math.abs(p - intervention.lesionedProbabilities[i])
			)
		);
		assert.equal(report.probe.maxProbabilityChange, difference, 'Reported intervention difference');
		assert.equal(report.probe.checkpointPreserved, true);
		if (
			restorationReceipt?.modelId === checked.modelId &&
			restorationReceipt.step === checked.final.step
		) {
			assert.equal(restoration, null, 'Duplicate restoration source run');
			restoration = {
				file: restorationPath,
				sha256: sha(await readFile(restorationPath)),
				...verifyRestoration(restorationReceipt, record, report, intervention)
			};
		}
		let fullLocalVerified = false;
		if (args.includes('--full-archives')) {
			const fullBytes = await readFile(report.fullArchive.path);
			assert.equal(sha(fullBytes), report.fullArchive.sha256, 'Full local archive checksum');
			const full = decodeArchive(fullBytes);
			verifyRecord(full, corpus);
			assert(full.checkpoint, 'Full local checkpoint missing');
			assert.equal(full.id, record.id);
			assert.deepEqual(full.metrics, record.metrics);
			assert.deepEqual(full.snapshots, record.snapshots);
			assert.deepEqual(full.samples, record.samples);
			assert.deepEqual(full.interventions, record.interventions);
			assert.deepEqual(full.provenance, record.provenance);
			if (record.checkpoint) assert.deepEqual(full.checkpoint, record.checkpoint);
			fullLocalVerified = true;
		}
		runs.push({
			id: record.id,
			presetId: record.presetId,
			seed: record.seed,
			step: checked.final.step,
			trainedTokens: checked.final.trainedTokens,
			parameters: checked.count,
			units: checked.units,
			resumable: !!record.checkpoint,
			file: reference.file,
			bytes: bytes.length,
			sha256: sha(bytes),
			parts: parts.map(({ path, bytes }) => ({ path, bytes: bytes.length, sha256: sha(bytes) })),
			reportFile: relative(process.cwd(), resolve(reportPath)),
			reportHash: sha(await readFile(reportPath)),
			validationLoss: checked.final.validationLoss,
			unigramLoss: checked.final.unigramLoss,
			interventionMaximumProbabilityChange: difference,
			fullLocalVerified
		});
	}
	if (restorationReceipt)
		assert(restoration, 'Restoration receipt has no matching published checkpoint');
	const result = {
		version: 1,
		auditedAt: new Date().toISOString(),
		auditScriptHash: sha(await readFile('scripts/audit-stories.mjs')),
		catalogHash: sha(await readFile(catalogPath)),
		corpusProvenanceHash: corpus.provenanceHash,
		methods:
			'Independent packed-file parsing; source/corpus/catalog/report checksums; tensor/config validation; independent fixed-window unigram baseline; raw activation variation; saved projection summary arithmetic; raw intervention probability difference. Optional restoration receipt: exact source/checkpoint identity and independent arithmetic over all four probability vectors. No model forward pass or PCA recomputation. Checkpoint-preservation is a recorded runtime assertion, not independently re-executed here.',
		restoration,
		runs
	};
	await writeFile(
		option('--output', 'static/experiments/stories-audit.json'),
		`${JSON.stringify(result, null, 2)}\n`
	);
	console.log(JSON.stringify(runs, null, 2));
}
if (args.includes('--self-test')) selfTest();
else await audit();
