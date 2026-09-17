/** Rebuild the train-only BPE corpus from the existing attributed ASCII subset.
 * Node 24+: node scripts/prepare-token-stories.mjs
 * Original <|endoftext|> boundaries are recovered from a pinned source revision,
 * then verified byte-for-byte against the previously committed ASCII corpus.
 */
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import {
	createTokenStoryTokenizer,
	identifyTokenStoryTokenizer,
	trainTokenStoryBpe,
	TOKEN_STORY_VOCABULARY_SIZE
} from '../src/lib/token-stories/tokenizer.ts';

const SOURCE_REVISION = 'f54c09fd23315a6f9c86f9dc80f725de7d8f9c64';
const SOURCE = 'https://huggingface.co/datasets/roneneldan/TinyStories';
const destination = resolve('static/data/tinystories-bpe');
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const jsonBytes = (value) => Buffer.from(JSON.stringify(value, null, 2) + '\n');
const normalize = (text) =>
	text
		.replace(/[‘’]/g, "'")
		.replace(/[“”]/g, '"')
		.replace(/[–—]/g, '-')
		.replace(/…/g, '...')
		.replace(/[^\n\x20-\x7e]/g, ' ')
		.replace(/[ \t]+/g, ' ')
		.trim();

async function firstOriginalStories(file, count) {
	const response = await fetch(`${SOURCE}/resolve/${SOURCE_REVISION}/${file}`);
	if (!response.ok || !response.body)
		throw new Error(`Original story boundaries: HTTP ${response.status}`);
	const reader = response.body.getReader();
	const decoder = new TextDecoder();
	let pending = '';
	const stories = [];
	try {
		while (stories.length < count) {
			const { value, done } = await reader.read();
			if (done) throw new Error(`Source ended before ${count} complete stories.`);
			pending += decoder.decode(value, { stream: true });
			let boundary;
			while ((boundary = pending.indexOf('<|endoftext|>')) >= 0 && stories.length < count) {
				const story = pending.slice(0, boundary).trim();
				pending = pending.slice(boundary + '<|endoftext|>'.length);
				if (story) stories.push(normalize(story));
			}
		}
	} finally {
		await reader.cancel();
	}
	return stories;
}

const [asciiBytes, asciiMetadataBytes, license] = await Promise.all(
	['tokens.bin', 'corpus.json', 'LICENSE.html'].map((file) =>
		readFile(resolve('static/data/tinystories', file))
	)
);
const asciiMetadata = JSON.parse(asciiMetadataBytes);
const [train, validation] = await Promise.all([
	firstOriginalStories('TinyStories-train.txt', asciiMetadata.trainStories),
	firstOriginalStories('TinyStories-valid.txt', asciiMetadata.validationStories)
]);
const asciiText = Array.from(asciiBytes, (id) => asciiMetadata.chars[id]).join('');
const trainingText = train.join('\n\n') + '\n\n';
const validationText = validation.join('\n\n') + '\n\n';
if (
	trainingText + validationText !== asciiText ||
	trainingText.length !== asciiMetadata.trainTokens
)
	throw new Error('Pinned original stories do not exactly reproduce the existing ASCII corpus.');
if (
	train.length !== 2097 ||
	validation.length !== 184 ||
	new Set(validation).size !== validation.length ||
	validation.some((story) => trainingText.includes(story))
)
	throw new Error('Unexpected story split counts or exact held-out duplicate.');

console.log('Verified original boundaries: 2,097 train / 32 calibration / 152 evaluation stories.');
const definition = trainTokenStoryBpe(train, hash(trainingText));
if (definition.vocabularySize !== TOKEN_STORY_VOCABULARY_SIZE)
	throw new Error(`Only ${definition.vocabularySize} repeatable BPE tokens were found.`);
const tokenizerData = await identifyTokenStoryTokenizer(definition);
const tokenizer = createTokenStoryTokenizer(tokenizerData);
const tokenizerBytes = jsonBytes(tokenizerData);
const flat = [];
const splits = {};
for (const [name, stories] of [
	['train', train],
	['calibration', validation.slice(0, 32)],
	['evaluation', validation.slice(32)]
]) {
	const offset = flat.length;
	const storyOffsets = [];
	const storyLengths = [];
	const storyCharacters = [];
	for (const story of stories) {
		const encoded = tokenizer.encode(story, { bos: true, eos: true });
		if (tokenizer.decode(encoded) !== story) throw new Error('Tokenizer story round-trip failed.');
		storyOffsets.push(flat.length);
		storyLengths.push(encoded.length);
		storyCharacters.push(story.length);
		flat.push(...encoded);
	}
	splits[name] = {
		offset,
		length: flat.length - offset,
		stories: stories.length,
		characters: storyCharacters.reduce((sum, count) => sum + count, 0),
		storyOffsets,
		storyLengths,
		storyCharacters
	};
}
const tokenBytes = Buffer.alloc(flat.length * 2);
flat.forEach((id, i) => tokenBytes.writeUInt16LE(id, i * 2));
const corpusIdentity = hash(
	JSON.stringify({
		tokenizerId: tokenizer.id,
		tokensSha256: hash(tokenBytes),
		splits,
		asciiSourceSha256: hash(asciiBytes),
		sourceRevision: SOURCE_REVISION
	})
);
const metadata = {
	version: 1,
	id: `tinystories-bpe-corpus-v1:${corpusIdentity}`,
	dataset: 'TinyStories',
	authors: asciiMetadata.authors,
	source: SOURCE,
	sourceRevision: SOURCE_REVISION,
	license: asciiMetadata.license,
	licenseUrl: asciiMetadata.licenseUrl,
	tokenizer: {
		file: 'tokenizer.json',
		id: tokenizer.id,
		sha256: hash(tokenizerBytes),
		vocabularySize: tokenizer.vocabularySize,
		bosId: tokenizer.bosId,
		eosId: tokenizer.eosId
	},
	tokens: {
		file: 'tokens.bin',
		encoding: 'uint16-le',
		count: flat.length,
		bytes: tokenBytes.length,
		sha256: hash(tokenBytes)
	},
	splits,
	trainingCharactersPerTextToken:
		splits.train.characters / (splits.train.length - 2 * splits.train.stories),
	evaluationCharactersPerTextToken:
		splits.evaluation.characters / (splits.evaluation.length - 2 * splits.evaluation.stories),
	preparation:
		'Same bounded ASCII TinyStories subset as the character study. Exact original story boundaries recovered from the pinned upstream release. Word-boundary BPE merges fit exclusively on training stories; BOS and EOS bound each story. First 32 original validation stories are geometry calibration; remaining 152 are held-out evaluation.',
	limitations:
		'A normalized ASCII subset of 2,097 training stories, not the full TinyStories benchmark. Token loss is not directly comparable to character loss; report nats per character on the same evaluated text for that comparison.'
};
const metadataBytes = jsonBytes(metadata);
const provenance = {
	version: 1,
	id: metadata.id,
	source: SOURCE,
	sourceRevision: SOURCE_REVISION,
	sourceAsciiCorpusId: `tinystories-ascii-v1:${hash(asciiBytes)}`,
	sourceAsciiFiles: {
		'tokens.bin': { bytes: asciiBytes.length, sha256: hash(asciiBytes) },
		'corpus.json': { bytes: asciiMetadataBytes.length, sha256: hash(asciiMetadataBytes) }
	},
	trainingTextSha256: definition.trainingTextSha256,
	algorithm:
		'Weighted word-piece pair counts over training stories; maximum frequency first, ties lexicographic by integer (leftId,rightId); pairs must repeat. Merge ranks define inference. No merge crosses a word-piece, story, or split boundary.',
	heldOutUsedToFitTokenizer: false,
	exactHeldOutDuplicatesInTraining: 0,
	originalTrainingStoriesContainingInternalBlankLines: train.filter((story) =>
		story.includes('\n\n')
	).length,
	allStoriesRoundTripExactly: true,
	files: Object.fromEntries(
		[
			['tokenizer.json', tokenizerBytes],
			['tokens.bin', tokenBytes],
			['corpus.json', metadataBytes],
			['LICENSE.html', license]
		].map(([name, bytes]) => [name, { bytes: bytes.length, sha256: hash(bytes) }])
	)
};
await mkdir(destination, { recursive: true });
await Promise.all([
	writeFile(resolve(destination, 'tokenizer.json'), tokenizerBytes),
	writeFile(resolve(destination, 'tokens.bin'), tokenBytes),
	writeFile(resolve(destination, 'corpus.json'), metadataBytes),
	writeFile(resolve(destination, 'LICENSE.html'), license),
	writeFile(resolve(destination, 'provenance.json'), jsonBytes(provenance))
]);
console.log(
	JSON.stringify(
		{
			id: metadata.id,
			tokenizer: metadata.tokenizer,
			tokens: metadata.tokens,
			splits: Object.fromEntries(
				Object.entries(splits).map(([name, split]) => [
					name,
					{
						stories: split.stories,
						tokens: split.length,
						characters: split.characters
					}
				])
			),
			trainingCharactersPerTextToken: metadata.trainingCharactersPerTextToken
		},
		null,
		2
	)
);
