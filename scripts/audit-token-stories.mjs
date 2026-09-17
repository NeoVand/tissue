/** Independent BPE archive, corpus and arithmetic audit; no model execution or PCA refit.
 * node scripts/audit-token-stories.mjs [--write] [--output path]
 * node scripts/audit-token-stories.mjs --file run.tissue --report report.json
 * node scripts/audit-token-stories.mjs --self-test
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { relative, resolve, sep } from 'node:path';
import {
	TOKEN_STORY_PRESETS,
	tokenStoryParameterCount,
	tokenStoryUnitCount,
	validateTokenStoryCheckpoint
} from '../src/lib/token-stories/protocol.ts';

const args = process.argv.slice(2);
const option = (name, fallback) => {
	const i = args.indexOf(name);
	if (i < 0) return fallback;
	assert(args[i + 1] && !args[i + 1].startsWith('--'), `${name} requires a value`);
	return args[i + 1];
};
const sha = (bytes) => createHash('sha256').update(bytes).digest('hex');
const readJSON = async (path) => JSON.parse(await readFile(path, 'utf8'));
const magic = Buffer.from('TISSUE-TOKEN-STORY-V1\n');
const maxBytes = 512 * 1024 * 1024;
const integer = (n, low = 0, high = Number.MAX_SAFE_INTEGER) =>
	assert(Number.isSafeInteger(n) && n >= low && n <= high, `Invalid integer ${n}`);
const finite = (n, low = -Infinity, high = Infinity) =>
	assert(Number.isFinite(n) && n >= low && n <= high, `Invalid number ${n}`);
const close = (a, b, label, tolerance = 1e-12) =>
	assert(
		Math.abs(a - b) <= tolerance * Math.max(1, Math.abs(b)),
		`${label}: ${a} differs from ${b}`
	);
const safeRepoPath = (path) => {
	const target = resolve(path),
		local = relative(process.cwd(), target);
	assert(local && local !== '..' && !local.startsWith(`..${sep}`), `Outside repository: ${path}`);
	return target;
};
const assetPath = (path) => {
	assert(
		/^\/experiments\/token-stories-[a-zA-Z0-9.-]+$/.test(path),
		`Unexpected asset path: ${path}`
	);
	return safeRepoPath(`static${path}`);
};

/** Independent parser: checks ordinal placeholders, bounds, unsafe keys and trailing bytes. */
function decodeArchive(bytes) {
	assert(bytes.length >= magic.length + 4 && bytes.length <= maxBytes, 'Archive size');
	assert(bytes.subarray(0, magic.length).equals(magic), 'Archive magic');
	const length = bytes.readUInt32LE(magic.length),
		start = magic.length + 4;
	assert(
		length > 0 && length <= 32 * 1024 * 1024 && start + length <= bytes.length,
		'Header length'
	);
	let offset = start + length,
		ordinal = 0;
	const result = JSON.parse(
		new TextDecoder('utf-8', { fatal: true }).decode(bytes.subarray(start, offset)),
		(key, value) => {
			assert(!['__proto__', 'constructor', 'prototype'].includes(key), 'Unsafe metadata key');
			if (value && typeof value === 'object' && '__tissueFloat32' in value) {
				assert.equal(value.__tissueFloat32, ordinal++, 'Packed tensor order');
				assert.deepEqual(Object.keys(value).sort(), ['__tissueFloat32', 'length']);
				integer(value.length, 0, maxBytes / 4);
				const end = offset + 4 * value.length;
				assert(end <= bytes.length, 'Packed tensor exceeds file');
				const output = new Float32Array(
					bytes.buffer.slice(bytes.byteOffset + offset, bytes.byteOffset + end)
				);
				offset = end;
				assert(output.every(Number.isFinite), 'Nonfinite packed tensor');
				return output;
			}
			return value;
		}
	);
	assert.equal(offset, bytes.length, 'Unexpected trailing bytes');
	return result;
}

function selfTest() {
	const pack = (header, values = new Float32Array([1, 2])) => {
		const json = Buffer.from(JSON.stringify(header)),
			length = Buffer.alloc(4);
		length.writeUInt32LE(json.length);
		return Buffer.concat([magic, length, json, Buffer.from(values.buffer)]);
	};
	const valid = pack({ x: { __tissueFloat32: 0, length: 2 } });
	assert.deepEqual(decodeArchive(valid).x, new Float32Array([1, 2]));
	assert.throws(() => decodeArchive(valid.subarray(0, -1)), /exceeds/);
	assert.throws(() => decodeArchive(Buffer.concat([valid, Buffer.alloc(1)])), /trailing/);
	assert.throws(() => decodeArchive(pack({ x: { __tissueFloat32: 1, length: 2 } })), /order/);
	assert.throws(() => decodeArchive(pack({ x: { __tissueFloat32: 0, length: -1 } })), /integer/);
	assert.throws(
		() => decodeArchive(pack({ x: { __tissueFloat32: 0, length: 2 } }, new Float32Array([1, NaN]))),
		/Nonfinite/
	);
	assert.throws(
		() => decodeArchive(pack(JSON.parse('{"__proto__":{}}'), new Float32Array())),
		/Unsafe/
	);
	console.log('BPE archive audit self-test passed: round trip and six corruption cases.');
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
	const root = 'static/data/tinystories-bpe',
		provenance = await readJSON(`${root}/provenance.json`);
	for (const [name, expected] of Object.entries(provenance.files)) {
		assert(['tokens.bin', 'tokenizer.json', 'corpus.json', 'LICENSE.html'].includes(name));
		const bytes = await readFile(`${root}/${name}`);
		assert.equal(bytes.length, expected.bytes, `Corpus bytes: ${name}`);
		assert.equal(sha(bytes), expected.sha256, `Corpus checksum: ${name}`);
	}
	assert.equal(provenance.heldOutUsedToFitTokenizer, false);
	const metadata = await readJSON(`${root}/corpus.json`),
		tokenizer = await readJSON(`${root}/tokenizer.json`);
	const canonical = JSON.stringify({
		version: tokenizer.version,
		kind: tokenizer.kind,
		baseCharacters: tokenizer.baseCharacters,
		merges: tokenizer.merges,
		pieces: tokenizer.pieces,
		vocabularySize: tokenizer.vocabularySize,
		bosId: tokenizer.bosId,
		eosId: tokenizer.eosId,
		trainingTextSha256: tokenizer.trainingTextSha256
	});
	assert.equal(tokenizer.id, `tinystories-bpe-v1:${sha(canonical)}`);
	assert.equal(tokenizer.id, metadata.tokenizer.id);
	assert.equal(metadata.tokenizer.sha256, sha(await readFile(`${root}/tokenizer.json`)));
	assert.equal(tokenizer.vocabularySize, 4096);
	assert.equal(tokenizer.bosId, 96);
	assert.equal(tokenizer.eosId, 97);
	const base = ['\n', ...Array.from({ length: 95 }, (_, i) => String.fromCharCode(32 + i))];
	assert.deepEqual(tokenizer.baseCharacters, base);
	const pieces = [...base, '<|bos|>', '<|eos|>'];
	tokenizer.merges.forEach(([a, b], i) => {
		integer(a, 0, 97 + i);
		integer(b, 0, 97 + i);
		assert(a !== 96 && a !== 97 && b !== 96 && b !== 97);
		pieces.push(pieces[a] + pieces[b]);
	});
	assert.deepEqual(pieces, tokenizer.pieces);
	const decode = (ids) =>
		Array.from(ids, (id) => {
			integer(id, 0, 4095);
			return id === 96 || id === 97 ? '' : pieces[id];
		}).join('');
	// Independent ordered merge passes; no import from the runtime encoder.
	const encode = (text) => {
		assert(/^[\n\x20-\x7e]*$/.test(text), 'Unsupported prompt characters');
		const charClass = (c) => (c === '\n' ? 0 : /[A-Za-z]/.test(c) ? 1 : /[0-9]/.test(c) ? 2 : 3);
		const words = [];
		let start = 0;
		for (let i = 1; i < text.length; i++)
			if (
				text[i] === '\n' ||
				text[i - 1] === '\n' ||
				text[i] === ' ' ||
				(text[i - 1] !== ' ' && charClass(text[i]) !== charClass(text[i - 1]))
			) {
				words.push(text.slice(start, i));
				start = i;
			}
		if (start < text.length) words.push(text.slice(start));
		return words.flatMap((word) => {
			let ids = Array.from(word, (c) => base.indexOf(c));
			tokenizer.merges.forEach(([a, b], rank) => {
				const next = [];
				for (let i = 0; i < ids.length; i++) {
					if (ids[i] === a && ids[i + 1] === b) {
						next.push(98 + rank);
						i++;
					} else next.push(ids[i]);
				}
				ids = next;
			});
			return ids;
		});
	};
	const bytes = await readFile(`${root}/tokens.bin`);
	assert.equal(metadata.tokens.sha256, sha(bytes));
	assert.equal(metadata.tokens.encoding, 'uint16-le');
	const tokens = Uint16Array.from({ length: bytes.length / 2 }, (_, i) =>
		bytes.readUInt16LE(i * 2)
	);
	assert.equal(tokens.length, metadata.tokens.count);
	assert.equal(bytes.length, metadata.tokens.bytes);
	const texts = {};
	let end = 0;
	for (const [name, count] of [
		['train', 2097],
		['calibration', 32],
		['evaluation', 152]
	]) {
		const split = metadata.splits[name];
		assert.equal(split.offset, end);
		assert.equal(split.stories, count);
		texts[name] = [];
		for (let i = 0; i < count; i++) {
			assert.equal(split.storyOffsets[i], end);
			integer(split.storyLengths[i], 3);
			const ids = tokens.subarray(end, end + split.storyLengths[i]);
			assert.equal(ids[0], 96);
			assert.equal(ids.at(-1), 97);
			assert(ids.subarray(1, -1).every((id) => id !== 96 && id !== 97));
			const text = decode(ids);
			assert.equal(text.length, split.storyCharacters[i]);
			texts[name].push(text);
			end += ids.length;
		}
		assert.equal(end, split.offset + split.length);
		assert.equal(
			texts[name].reduce((sum, t) => sum + t.length, 0),
			split.characters
		);
	}
	assert.equal(end, tokens.length);
	assert.equal(sha(texts.train.join('\n\n') + '\n\n'), tokenizer.trainingTextSha256);
	const asciiMeta = await readJSON('static/data/tinystories/corpus.json'),
		asciiBytes = await readFile('static/data/tinystories/tokens.bin');
	assert.equal(
		[...texts.train, ...texts.calibration, ...texts.evaluation].join('\n\n') + '\n\n',
		Array.from(asciiBytes, (id) => asciiMeta.chars[id]).join('')
	);
	assert.equal(new Set([...texts.calibration, ...texts.evaluation]).size, 184);
	assert(![...texts.calibration, ...texts.evaluation].some((text) => texts.train.includes(text)));
	const identity = sha(
		JSON.stringify({
			tokenizerId: tokenizer.id,
			tokensSha256: sha(bytes),
			splits: metadata.splits,
			asciiSourceSha256: sha(asciiBytes),
			sourceRevision: metadata.sourceRevision
		})
	);
	assert.equal(metadata.id, `tinystories-bpe-corpus-v1:${identity}`);
	assert.equal(metadata.id, provenance.id);
	const byContext = (context) => {
		const windows = (name, count, seed) => {
			const split = metadata.splits[name],
				next = random(seed);
			return Array.from({ length: count }, (_, i) => {
				const story = Math.floor((i * split.stories) / count),
					length = split.storyLengths[story],
					size = Math.min(context + 1, length);
				const start = Math.floor(next() * (length - size + 1));
				return tokens.slice(
					split.storyOffsets[story] + start,
					split.storyOffsets[story] + start + size
				);
			});
		};
		const calibration = windows('calibration', 8, 0x62706361).map((ids) =>
			Array.from(ids.subarray(0, Math.min(context, ids.length - 1)))
		);
		const evaluation = windows('evaluation', 16, 0x62706576),
			unigram = new Float64Array(4096).fill(1);
		const train = metadata.splits.train;
		let trainingTargets = 0;
		for (let i = 0; i < train.stories; i++)
			for (let j = 1; j < train.storyLengths[i]; j++) {
				unigram[tokens[train.storyOffsets[i] + j]]++;
				trainingTargets++;
			}
		for (let i = 0; i < unigram.length; i++) unigram[i] /= trainingTargets + 4096;
		const best = unigram.indexOf(Math.max(...unigram));
		let loss = 0,
			correct = 0,
			count = 0;
		for (const window of evaluation)
			for (const target of window.subarray(1)) {
				loss -= Math.log(unigram[target]);
				correct += +(target === best);
				count++;
			}
		return {
			calibration,
			evaluationTokens: count,
			unigramLoss: loss / count,
			unigramAccuracy: correct / count
		};
	};
	return {
		metadata,
		tokenizer,
		pieces,
		decode,
		encode,
		byContext,
		provenanceHash: sha(await readFile(`${root}/provenance.json`))
	};
}

const requiredSources = [
	'src/lib/token-stories/protocol.ts',
	'src/lib/token-stories/dataset.ts',
	'src/lib/token-stories/model.ts',
	'src/lib/token-stories/runtime.ts',
	'src/lib/token-stories/worker.ts',
	'src/lib/token-stories/engine.ts',
	'src/lib/token-stories/tokenizer.ts',
	'src/lib/token-stories/archive.ts',
	'src/lib/token-stories/geometry.ts',
	'src/lib/token-stories/geometry-engine.ts',
	'src/lib/stories/geometry.ts',
	'src/lib/stories/geometry-worker.ts',
	'src/lib/stories/geometry-engine.ts',
	'src/lib/stories/validate-geometry.ts',
	'src/lib/lab/model/adam.ts',
	'src/lib/lab/geometry.ts',
	'pnpm-lock.yaml',
	'docs/token-stories-design.md',
	'scripts/measure-token-stories.mjs',
	'static/data/tinystories-bpe/provenance.json',
	'static/data/tinystories-bpe/tokenizer.json',
	'static/data/tinystories-bpe/corpus.json'
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
		assert.equal(sha(await readFile(safeRepoPath(path))), expected, `Source checksum: ${path}`);
	}
}

function verifyGeometry(atlas, geometry) {
	const n = atlas.unitCount,
		p = atlas.dimensions,
		normalized = new Float64Array(n * p);
	assert.equal(p, 128);
	assert.equal(atlas.fingerprints.length, n * p);
	assert.equal(geometry.rmsFloor, 1e-8);
	assert.equal(geometry.positions.length, n);
	const valid = Array(n).fill(false);
	for (let i = 0; i < n; i++) {
		const row = atlas.fingerprints.subarray(i * p, (i + 1) * p);
		let mean = 0,
			square = 0;
		for (const x of row) {
			finite(x, 0);
			mean += x / p;
		}
		for (const x of row) square += (x - mean) ** 2;
		const magnitude = Math.sqrt(square / p);
		valid[i] = magnitude > 1e-8;
		close(geometry.magnitudes[i], magnitude, 'Raw activation standard deviation');
		assert.equal(geometry.valid[i], valid[i]);
		if (valid[i])
			for (let j = 0; j < p; j++) normalized[i * p + j] = (row[j] - mean) / Math.sqrt(square);
		assert.equal(geometry.positions[i].length, 3);
		geometry.positions[i].forEach((x) => finite(x));
	}
	const ids = Array.from({ length: n }, (_, i) => i),
		next = random(0x51a7c0de);
	for (let i = 0; i < Math.min(n, 128); i++) {
		const j = i + Math.floor(next() * (n - i));
		[ids[i], ids[j]] = [ids[j], ids[i]];
	}
	const selected = ids.slice(0, Math.min(n, 128)).sort((a, b) => a - b);
	assert.deepEqual(geometry.audit.ids, selected);
	assert.equal(geometry.audit.k, 6);
	assert.equal(geometry.audit.validPopulation, valid.filter(Boolean).length);
	const perFocal = selected.map((id) => {
		const original = [],
			projected = [];
		if (valid[id])
			for (let other = 0; other < n; other++) {
				if (other === id || !valid[other]) continue;
				let square = 0;
				for (let j = 0; j < p; j++)
					square += (normalized[id * p + j] - normalized[other * p + j]) ** 2;
				original.push({ id: other, distance: Math.sqrt(square) });
				projected.push(
					Math.hypot(...geometry.positions[id].map((x, j) => x - geometry.positions[other][j]))
				);
			}
		original.sort((a, b) => a.distance - b.distance || a.id - b.id);
		projected.sort((a, b) => a - b);
		const neighbors = original.slice(0, 6),
			radius = projected[neighbors.length - 1];
		const retention =
			!neighbors.length || geometry.pca.totalVariance === 0
				? null
				: neighbors.filter(
						(neighbor) =>
							Math.hypot(
								...geometry.positions[id].map((x, j) => x - geometry.positions[neighbor.id][j])
							) <=
							radius + Math.max(1e-12, radius * 1e-9)
					).length / neighbors.length;
		return { id, neighborCount: neighbors.length, retention };
	});
	assert.deepEqual(
		geometry.audit.perFocal,
		perFocal,
		'Independent raw/3D nearest-neighbor retention'
	);
	const scored = perFocal.filter((focal) => focal.retention !== null);
	assert.equal(geometry.audit.scoredFocals, scored.length);
	const retention = scored.length
		? scored.reduce((sum, f) => sum + f.retention, 0) / scored.length
		: null;
	assert.equal(geometry.audit.neighborRetention, retention);
	close(
		geometry.explainedVariance,
		geometry.explainedVarianceByAxis.reduce((a, b) => a + b, 0),
		'Variance summary'
	);
	finite(geometry.explainedVariance, 0, 1.000001);
	assert.equal(geometry.pca.converged, geometry.pca.relativeResidual <= geometry.pca.tolerance);
	return {
		step: atlas.step,
		validUnits: valid.filter(Boolean).length,
		scoredFocals: scored.length,
		retention,
		variance: geometry.explainedVariance
	};
}

function verifyRecord(record, corpus) {
	assert.equal(record.version, 1);
	assert.equal(record.kind, 'tissue-token-story-run');
	assert.equal(record.corpusId, corpus.metadata.id);
	assert.deepEqual(record.tokenizer, corpus.tokenizer);
	const config = TOKEN_STORY_PRESETS[record.presetId];
	assert(config, 'Unknown preset');
	const expected = corpus.byContext(config.context),
		units = tokenStoryUnitCount(config),
		count = tokenStoryParameterCount(config);
	assert(record.metrics.length && record.snapshots.length, 'Missing measured evidence');
	let step = -1,
		trained = -1;
	for (const metric of record.metrics) {
		integer(metric.step);
		assert(metric.step > step);
		step = metric.step;
		integer(metric.trainedTokens, 0, step * config.batchSize * config.context);
		assert(metric.trainedTokens >= trained);
		trained = metric.trainedTokens;
		assert.equal(metric.evaluationTokens, expected.evaluationTokens);
		finite(metric.validationLoss, 0);
		finite(metric.validationAccuracy, 0, 1);
		close(
			metric.validationAccuracy * expected.evaluationTokens,
			Math.round(metric.validationAccuracy * expected.evaluationTokens),
			'Integer accuracy'
		);
		close(metric.unigramLoss, expected.unigramLoss, 'Training-only unigram CE');
		close(metric.unigramAccuracy, expected.unigramAccuracy, 'Training-only unigram accuracy');
		close(metric.uniformLoss, Math.log(4096), 'Uniform baseline');
	}
	const final = record.metrics.at(-1),
		modelId = record.snapshots[0].atlas.modelId;
	const projections = record.snapshots.map(({ atlas, geometry }) => {
		assert.equal(atlas.modelId, modelId);
		assert.equal(atlas.corpusId, corpus.metadata.id);
		assert.equal(atlas.tokenizerId, corpus.tokenizer.id);
		assert.equal(atlas.seed, record.seed);
		assert.deepEqual(atlas.config, config);
		assert.equal(atlas.unitCount, units);
		assert(record.metrics.some((m) => m.step === atlas.step));
		assert.deepEqual(atlas.tokenIds, expected.calibration);
		assert.deepEqual(atlas.examples, expected.calibration.map(corpus.decode));
		assert.deepEqual(
			atlas.positions,
			Array.from({ length: 16 }, (_, i) => Math.round((i * (config.context - 1)) / 15))
		);
		assert.deepEqual(
			atlas.tokenPositions,
			atlas.tokenIds.map((ids) =>
				Array.from({ length: 16 }, (_, i) => Math.round((i * (ids.length - 1)) / 15))
			)
		);
		assert.equal(geometry.modelId, modelId);
		assert.equal(geometry.corpusId, corpus.metadata.id);
		assert.equal(geometry.step, atlas.step);
		assert.equal(geometry.unitCount, units);
		assert.equal(geometry.layerWidth, config.hidden);
		return verifyGeometry(atlas, geometry);
	});
	if (record.checkpoint) {
		validateTokenStoryCheckpoint(record.checkpoint);
		assert.equal(record.checkpoint.modelId, modelId);
		assert.equal(record.checkpoint.corpusId, corpus.metadata.id);
		assert.equal(record.checkpoint.tokenizerId, corpus.tokenizer.id);
		assert.deepEqual(record.checkpoint.config, config);
		assert.equal(record.checkpoint.step, final.step);
		assert.equal(record.checkpoint.trainedTokens, final.trainedTokens);
	}
	for (const sample of record.samples ?? []) {
		assert.equal(sample.modelId, modelId);
		integer(sample.step, 0, final.step);
		assert.equal(sample.tokenizerId, corpus.tokenizer.id);
		const full = [96, ...corpus.encode(sample.prompt.original)],
			ids = full.slice(-config.context);
		assert.deepEqual(sample.prompt.tokenIds, ids);
		assert.equal(sample.prompt.text, corpus.decode(ids));
		assert.deepEqual(
			sample.prompt.pieces,
			ids.map((id) => corpus.pieces[id])
		);
		assert.equal(sample.prompt.truncatedTokens, full.length - ids.length);
		assert.equal(sample.prompt.includesBos, ids[0] === 96);
		assert.deepEqual(sample.prompt.unsupportedCharacters, []);
		integer(sample.requestedTokens, 1, 256);
		integer(sample.samplingSeed, 0, 0xffffffff);
		finite(sample.temperature, 0.05, 2);
		integer(sample.topK, 1, 4096);
		assert(sample.tokenIds.length <= sample.requestedTokens);
		assert(!sample.tokenIds.includes(96));
		assert(
			sample.tokenIds.indexOf(97) < 0 || sample.tokenIds.indexOf(97) === sample.tokenIds.length - 1
		);
		assert.equal(sample.completion, corpus.decode(sample.tokenIds));
		assert.deepEqual(
			sample.pieces,
			sample.tokenIds.map((id) => corpus.pieces[id])
		);
		assert.equal(sample.stoppedOnEos, sample.tokenIds.at(-1) === 97);
		assert(
			sample.cancelled || sample.stoppedOnEos || sample.tokenIds.length === sample.requestedTokens
		);
	}
	for (const intervention of record.interventions ?? []) {
		assert.equal(intervention.modelId, modelId);
		integer(intervention.step, 0, final.step);
		integer(intervention.neuron, 0, units - 1);
		for (const row of [intervention.probabilities, intervention.lesionedProbabilities]) {
			assert(row instanceof Float32Array);
			assert.equal(row.length, 4096);
			row.forEach((v) => finite(v, 0, 1));
			close(
				row.reduce((sum, v) => sum + v, 0),
				1,
				'Intervention probability mass',
				1e-4
			);
		}
	}
	return { units, count, final, projections };
}

async function auditOne(bytes, reportPath, corpus, reference) {
	const record = decodeArchive(bytes),
		checked = verifyRecord(record, corpus);
	await verifyProvenance(record.provenance);
	if (reference) {
		assert.equal(bytes.length, reference.bytes);
		assert.equal(sha(bytes), reference.sha256);
		for (const key of ['id', 'title', 'presetId', 'seed'])
			assert.equal(reference[key], record[key]);
		assert.equal(reference.step, checked.final.step);
		assert.equal(reference.parameterCount, checked.count);
		assert.equal(reference.unitCount, checked.units);
		assert.equal(reference.validationLoss, checked.final.validationLoss);
		assert.equal(reference.unigramLoss, checked.final.unigramLoss);
	}
	const report = await readJSON(reportPath);
	assert.equal(report.corpusId, corpus.metadata.id);
	assert.equal(report.tokenizerId, corpus.tokenizer.id);
	if (!reference)
		assert.equal(sha(bytes), report.fullArchive.sha256, 'Local full archive checksum');
	for (const key of ['id', 'title', 'presetId', 'seed']) assert.equal(report[key], record[key]);
	assert.deepEqual(report.metrics, record.metrics);
	assert.deepEqual(report.provenance, record.provenance);
	assert.deepEqual(report.generation, record.samples.at(-1));
	assert.deepEqual(report.samples, record.samples);
	if (reference) assert.deepEqual(report.published, reference);
	const last = record.snapshots.at(-1).geometry;
	assert.deepEqual(report.projection, {
		validUnits: last.valid.filter(Boolean).length,
		variance: last.explainedVariance,
		retention: last.audit.neighborRetention,
		audit: last.audit,
		pca: last.pca
	});
	assert.equal(report.timings.length, record.snapshots.length);
	report.timings.forEach((timing, i) => {
		assert.equal(timing.step, record.snapshots[i].atlas.step);
		assert.equal(timing.atlasMs, record.snapshots[i].atlas.elapsedMs);
	});
	const intervention = record.interventions.find(
		(event) => event.step === checked.final.step && event.neuron === report.probe.unit
	);
	assert(intervention, 'Missing raw intervention');
	assert.equal(report.probe.unit, Math.floor(checked.units / 2));
	const difference = Math.max(
		...Array.from(intervention.probabilities, (v, i) =>
			Math.abs(v - intervention.lesionedProbabilities[i])
		)
	);
	assert.equal(report.probe.maxProbabilityChange, difference);
	assert.equal(report.probe.checkpointPreserved, true);
	return {
		id: record.id,
		presetId: record.presetId,
		seed: record.seed,
		step: checked.final.step,
		trainedTokens: checked.final.trainedTokens,
		parameters: checked.count,
		units: checked.units,
		bytes: bytes.length,
		sha256: sha(bytes),
		reportFile: reportPath,
		reportHash: sha(await readFile(reportPath)),
		validationLoss: checked.final.validationLoss,
		unigramLoss: checked.final.unigramLoss,
		samplesDecoded: record.samples.length,
		projections: checked.projections,
		interventionMaximumProbabilityChange: difference,
		resumable: !!record.checkpoint
	};
}

async function audit() {
	const corpus = await verifyCorpus(),
		file = option('--file'),
		report = option('--report'),
		runs = [];
	if (file) {
		assert(report, '--file requires --report');
		runs.push(await auditOne(await readFile(file), report, corpus));
	} else {
		const catalog = await readJSON(
			option('--catalog', 'static/experiments/token-stories-index.json')
		);
		assert.equal(catalog.version, 1);
		assert(
			catalog.references.length > 0,
			'No published BPE references to audit yet; use --file and --report for a local run.'
		);
		assert.equal(new Set(catalog.references.map((r) => r.id)).size, catalog.references.length);
		for (const reference of catalog.references) {
			const paths = reference.parts ?? [reference.file];
			assert(paths.length > 0 && paths.length <= 32);
			paths.forEach((path, i) =>
				assert.equal(path, reference.parts ? `${reference.file}.part-${i}` : reference.file)
			);
			const bytes = Buffer.concat(
				await Promise.all(paths.map((path) => readFile(assetPath(path))))
			);
			runs.push(await auditOne(bytes, assetPath(reference.reportFile), corpus, reference));
		}
	}
	const result = {
		version: 1,
		auditedAt: new Date().toISOString(),
		auditScriptHash: sha(await readFile('scripts/audit-token-stories.mjs')),
		corpusProvenanceHash: corpus.provenanceHash,
		methods:
			'Independent packed binary parsing, corpus/tokenizer/source/report hashes, original story reconstruction, fixed held-out windows and training-only unigram baseline, raw activation normalization, exact original-space focal neighbors and tie-inclusive 3D retention, ordered BPE prompt encoding, generated token decoding and raw intervention arithmetic. No model forward pass or PCA refit. Checkpoint preservation remains a recorded runtime assertion.',
		runs
	};
	if (args.includes('--write') || args.includes('--output'))
		await writeFile(
			option('--output', 'static/experiments/token-stories-audit.json'),
			JSON.stringify(result, null, 2) + '\n'
		);
	console.log(JSON.stringify(result, null, 2));
}
if (args.includes('--self-test')) selfTest();
else await audit();
