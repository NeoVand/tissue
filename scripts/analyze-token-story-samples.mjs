/** Descriptive completion overlap/repetition audit. No coherence or originality score.
 * node scripts/analyze-token-story-samples.mjs [--file run.tissue|report.json] [--output path]
 * Default: audit every sample in the published BPE catalog and write the lab artifact.
 * node scripts/analyze-token-story-samples.mjs --self-test
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const args = process.argv.slice(2);
const option = (name, fallback) => {
	const at = args.indexOf(name);
	if (at < 0) return fallback;
	assert(args[at + 1] && !args[at + 1].startsWith('--'), `${name} requires a value`);
	return args[at + 1];
};
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const readJSON = async (path) => JSON.parse(await readFile(path, 'utf8'));
const magic = Buffer.from('TISSUE-TOKEN-STORY-V1\n');
const special = (id) => id === 96 || id === 97;

/** Every position refers to one story; candidate extension cannot cross its end. */
function trainingIndex(stories) {
	const positions = new Map();
	for (let story = 0; story < stories.length; story++)
		for (let offset = 0; offset < stories[story].length; offset++) {
			const id = stories[story][offset];
			if (special(id)) continue;
			const list = positions.get(id) ?? [];
			list.push({ story, offset });
			positions.set(id, list);
		}
	return { stories, positions };
}

function longestMatch(sample, index) {
	let best = {
		tokenCount: 0,
		sampleTokenOffset: null,
		trainingStoryIndex: null,
		trainingStoryTokenOffset: null
	};
	for (let i = 0; i < sample.length; i++) {
		if (sample.length - i <= best.tokenCount) break;
		if (special(sample[i])) continue;
		for (const candidate of index.positions.get(sample[i]) ?? []) {
			const story = index.stories[candidate.story];
			let count = 0;
			while (
				i + count < sample.length &&
				candidate.offset + count < story.length &&
				!special(sample[i + count]) &&
				!special(story[candidate.offset + count]) &&
				sample[i + count] === story[candidate.offset + count]
			)
				count++;
			if (count > best.tokenCount)
				best = {
					tokenCount: count,
					sampleTokenOffset: i,
					trainingStoryIndex: candidate.story,
					trainingStoryTokenOffset: candidate.offset
				};
		}
	}
	return best;
}

/** Overlapping windows; special-token boundaries terminate a candidate four-gram. */
function fourGramRepetition(sample) {
	const frequencies = new Map();
	let windows = 0;
	for (let i = 0; i + 4 <= sample.length; i++) {
		const ids = sample.slice(i, i + 4);
		if (ids.some(special)) continue;
		windows++;
		const key = ids.join(','),
			existing = frequencies.get(key) ?? { tokenIds: ids, count: 0, sampleTokenOffsets: [] };
		existing.count++;
		existing.sampleTokenOffsets.push(i);
		frequencies.set(key, existing);
	}
	const repeated = [...frequencies.values()]
		.filter((entry) => entry.count > 1)
		.sort((a, b) => b.count - a.count || a.sampleTokenOffsets[0] - b.sampleTokenOffsets[0]);
	return {
		windowCount: windows,
		distinctSequenceCount: frequencies.size,
		repeatedSequenceCount: repeated.length,
		excessOccurrences: repeated.reduce((sum, entry) => sum + entry.count - 1, 0),
		occurrencesInRepeatedSequences: repeated.reduce((sum, entry) => sum + entry.count, 0),
		maxFrequency: Math.max(0, ...Array.from(frequencies.values(), (entry) => entry.count)),
		mostFrequentRepeatedSequences: repeated.slice(0, 6)
	};
}

function selfTest() {
	const index = trainingIndex([
		[96, 1, 2, 97],
		[96, 3, 4, 97],
		[96, 1, 2, 5, 97]
	]);
	assert.equal(longestMatch([1, 2, 3, 4], index).tokenCount, 2, 'Cannot bridge training stories');
	assert.equal(longestMatch([1, 2, 97, 5], index).tokenCount, 2, 'Cannot bridge sample EOS');
	assert.deepEqual(longestMatch([3, 4, 1, 2], index), {
		tokenCount: 2,
		sampleTokenOffset: 0,
		trainingStoryIndex: 1,
		trainingStoryTokenOffset: 1
	});
	assert.deepEqual(longestMatch([1, 2], index), {
		tokenCount: 2,
		sampleTokenOffset: 0,
		trainingStoryIndex: 0,
		trainingStoryTokenOffset: 1
	});
	assert.equal(longestMatch([96, 97], index).tokenCount, 0);
	const repetition = fourGramRepetition([1, 2, 1, 2, 1, 2]);
	assert.equal(repetition.windowCount, 3);
	assert.equal(repetition.repeatedSequenceCount, 1);
	assert.equal(repetition.maxFrequency, 2);
	assert.equal(repetition.excessOccurrences, 1);
	assert.deepEqual(repetition.mostFrequentRepeatedSequences[0].sampleTokenOffsets, [0, 2]);
	assert.equal(fourGramRepetition([1, 2, 97, 1, 2, 3, 4]).windowCount, 1);
	console.log(
		'Sample overlap audit self-test passed: story/EOS boundaries, ties, overlapping repeats.'
	);
}

/** Samples contain ordinary JSON token IDs; float payloads are not read as measurements. */
function metadataFromBytes(bytes) {
	if (!bytes.subarray(0, magic.length).equals(magic)) return JSON.parse(bytes.toString('utf8'));
	assert(bytes.length >= magic.length + 4 && bytes.length <= 512 * 1024 * 1024, 'Archive size');
	const length = bytes.readUInt32LE(magic.length),
		start = magic.length + 4;
	assert(
		length > 0 && length <= 32 * 1024 * 1024 && start + length <= bytes.length,
		'Archive metadata length'
	);
	const record = JSON.parse(
		new TextDecoder('utf-8', { fatal: true }).decode(bytes.subarray(start, start + length))
	);
	assert.equal(record.version, 1);
	assert.equal(record.kind, 'tissue-token-story-run');
	return record;
}

async function corpus() {
	const root = 'static/data/tinystories-bpe',
		metadata = await readJSON(`${root}/corpus.json`);
	const [tokenizerBytes, tokenBytes] = await Promise.all([
		readFile(`${root}/tokenizer.json`),
		readFile(`${root}/tokens.bin`)
	]);
	assert.equal(hash(tokenizerBytes), metadata.tokenizer.sha256);
	assert.equal(hash(tokenBytes), metadata.tokens.sha256);
	const tokenizer = JSON.parse(tokenizerBytes);
	assert.equal(tokenizer.id, metadata.tokenizer.id);
	assert.equal(tokenizer.bosId, 96);
	assert.equal(tokenizer.eosId, 97);
	const split = metadata.splits.train;
	assert.equal(split.stories, 2097);
	assert.equal(split.offset, 0);
	let end = split.offset;
	const stories = split.storyOffsets.map((offset, i) => {
		assert.equal(offset, end);
		const ids = Array.from({ length: split.storyLengths[i] }, (_, j) =>
			tokenBytes.readUInt16LE(2 * (offset + j))
		);
		assert.equal(ids[0], 96);
		assert.equal(ids.at(-1), 97);
		assert(ids.slice(1, -1).every((id) => !special(id)));
		end += ids.length;
		return ids;
	});
	assert.equal(end, split.length);
	const decode = (ids) =>
		ids
			.filter((id) => !special(id))
			.map((id) => tokenizer.pieces[id])
			.join('');
	assert.equal(hash(stories.map(decode).join('\n\n') + '\n\n'), tokenizer.trainingTextSha256);
	return { metadata, tokenizer, index: trainingIndex(stories), decode };
}

function analyze(record, source, data) {
	assert.equal(record.corpusId, data.metadata.id, 'Sample corpus identity');
	assert.equal(
		record.tokenizer?.id ?? record.tokenizerId,
		data.tokenizer.id,
		'Sample tokenizer identity'
	);
	const samples = record.samples ?? (record.generation ? [record.generation] : []);
	assert(samples.length > 0, 'No saved generated samples');
	if (record.generation && record.samples)
		assert.deepEqual(record.generation, record.samples.at(-1));
	return {
		runId: record.id,
		presetId: record.presetId,
		seed: record.seed,
		source,
		samples: samples.map((sample, i) => {
			assert.equal(sample.tokenizerId, data.tokenizer.id);
			assert(Array.isArray(sample.tokenIds) && sample.tokenIds.length <= 256);
			assert(
				sample.tokenIds.every(
					(id) => Number.isInteger(id) && id >= 0 && id < data.tokenizer.vocabularySize
				)
			);
			assert.equal(
				data.decode(sample.tokenIds),
				sample.completion,
				'Completion agrees with saved token IDs'
			);
			const match = longestMatch(sample.tokenIds, data.index),
				repeat = fourGramRepetition(sample.tokenIds);
			const matchedIds = match.tokenCount
				? sample.tokenIds.slice(match.sampleTokenOffset, match.sampleTokenOffset + match.tokenCount)
				: [];
			return {
				sampleIndex: i,
				step: sample.step,
				prompt: sample.prompt.original,
				samplingSeed: sample.samplingSeed,
				temperature: sample.temperature,
				topK: sample.topK,
				requestedTokens: sample.requestedTokens,
				generatedTokens: sample.tokenIds.length,
				generatedTextTokens: sample.tokenIds.filter((id) => !special(id)).length,
				stoppedOnEos: sample.stoppedOnEos,
				cancelled: sample.cancelled,
				completion: sample.completion,
				longestExactTrainingMatch: {
					...match,
					trainingCorpusTokenOffset: match.tokenCount
						? data.metadata.splits.train.storyOffsets[match.trainingStoryIndex] +
							match.trainingStoryTokenOffset
						: null,
					tokenIds: matchedIds,
					text: data.decode(matchedIds)
				},
				repeatedFourTokenSequences: {
					...repeat,
					mostFrequentRepeatedSequences: repeat.mostFrequentRepeatedSequences.map((entry) => ({
						...entry,
						text: data.decode(entry.tokenIds)
					}))
				}
			};
		})
	};
}

async function main() {
	const data = await corpus(),
		file = option('--file'),
		runs = [];
	if (file) {
		const bytes = await readFile(file);
		runs.push(
			analyze(
				metadataFromBytes(bytes),
				{ path: resolve(file), bytes: bytes.length, sha256: hash(bytes) },
				data
			)
		);
	} else {
		const catalog = await readJSON(
			option('--catalog', 'static/experiments/token-stories-index.json')
		);
		assert.equal(catalog.version, 1);
		assert(catalog.references.length > 0, 'No published token-story samples yet');
		for (const reference of catalog.references) {
			const paths = reference.parts ?? [reference.file];
			paths.forEach((path, i) => {
				assert(
					/^\/experiments\/token-stories-[a-zA-Z0-9.-]+$/.test(path),
					'Unexpected archive asset'
				);
				assert.equal(path, reference.parts ? `${reference.file}.part-${i}` : reference.file);
			});
			const bytes = Buffer.concat(
				await Promise.all(paths.map((path) => readFile(`static${path}`)))
			);
			assert.equal(bytes.length, reference.bytes);
			assert.equal(hash(bytes), reference.sha256);
			const record = metadataFromBytes(bytes);
			assert.equal(record.id, reference.id);
			runs.push(
				analyze(
					record,
					{
						path: reference.file,
						parts: reference.parts,
						bytes: bytes.length,
						sha256: hash(bytes)
					},
					data
				)
			);
		}
	}
	const result = {
		version: 1,
		analyzedAt: new Date().toISOString(),
		scriptHash: hash(await readFile('scripts/analyze-token-story-samples.mjs')),
		corpusId: data.metadata.id,
		tokenizerId: data.tokenizer.id,
		trainingStories: data.metadata.splits.train.stories,
		methods:
			'Every declared generated completion is compared token-for-token against training stories separately. Prompt tokens are excluded. Matches never cross a training-story boundary, BOS, or EOS. Longest-match ties choose earliest completion offset, lowest training-story index, then earliest story-token offset. Story-token offsets are zero-based including BOS at zero. Four-token repetition counts overlapping windows excluding BOS/EOS; excess occurrences count all appearances after the first. maxFrequency is zero for no eligible four-grams and one when all are unique.',
		limitations:
			'Descriptive copying and repetition diagnostics, not coherence, originality, or memorization scores. Formulaic TinyStories phrases naturally overlap. Short overlap does not establish originality; exact matches can occur by chance or shared conventions. BPE token spans depend on this tokenizer. No generated samples were selected or discarded by these diagnostics.',
		runs
	};
	const output = option(
		'--output',
		file
			? '/tmp/token-stories-sample-audit.json'
			: 'static/experiments/token-stories-sample-audit.json'
	);
	await writeFile(output, JSON.stringify(result, null, 2) + '\n');
	console.log(
		JSON.stringify(
			{
				output,
				runs: runs.map((run) => ({
					runId: run.runId,
					samples: run.samples.map((sample) => ({
						step: sample.step,
						generatedTextTokens: sample.generatedTextTokens,
						longestMatch: sample.longestExactTrainingMatch.tokenCount,
						matchText: sample.longestExactTrainingMatch.text,
						repeatedFourTokenSequences: sample.repeatedFourTokenSequences.repeatedSequenceCount,
						maximumFourGramFrequency: sample.repeatedFourTokenSequences.maxFrequency
					}))
				}))
			},
			null,
			2
		)
	);
}
if (args.includes('--self-test')) selfTest();
else await main();
