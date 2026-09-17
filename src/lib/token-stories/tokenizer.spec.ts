import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { describe, expect, it } from 'vitest';
import {
	createTokenStoryTokenizer,
	identifyTokenStoryTokenizer,
	tokenStoryWordPieces,
	trainTokenStoryBpe,
	validateTokenStoryTokenizerIdentity,
	TOKEN_STORY_BASE_CHARACTERS,
	type TokenStoryCorpusMetadata,
	type TokenStoryTokenizerData
} from './tokenizer';

const data = JSON.parse(
	readFileSync('static/data/tinystories-bpe/tokenizer.json', 'utf8')
) as TokenStoryTokenizerData;
const metadata = JSON.parse(
	readFileSync('static/data/tinystories-bpe/corpus.json', 'utf8')
) as TokenStoryCorpusMetadata;
const tokenizer = createTokenStoryTokenizer(data);
const dummyHash = '1'.repeat(64);
const digest = (bytes: string | Uint8Array) => createHash('sha256').update(bytes).digest('hex');

/** Deliberately simple independent implementation: ordered full merge passes. */
function referenceEncode(text: string): number[] {
	const ids = new Map(TOKEN_STORY_BASE_CHARACTERS.map((c, id) => [c, id]));
	return tokenStoryWordPieces(text).flatMap((word) => {
		let tokens = Array.from(word, (c) => ids.get(c)!);
		data.merges.forEach(([a, b], rank) => {
			const next = [];
			for (let i = 0; i < tokens.length; i++) {
				if (tokens[i] === a && tokens[i + 1] === b) {
					next.push(98 + rank);
					i++;
				} else next.push(tokens[i]);
			}
			tokens = next;
		});
		return tokens;
	});
}

describe('TinyStories train-only word-boundary BPE', () => {
	it('round trips every base symbol, whitespace, punctuation and literal special-token spellings', () => {
		for (const text of [
			TOKEN_STORY_BASE_CHARACTERS.join(''),
			'',
			'  A little girl\n\nnamed Lily said, "Hello!"  ',
			'aaa aaaaaa abcabcabc 123123 !!!! <|bos|> <|eos|>'
		]) {
			expect(tokenizer.decode(tokenizer.encode(text))).toBe(text);
			expect(tokenizer.decode(tokenizer.encode(text, { bos: true, eos: true }))).toBe(text);
		}
		expect(tokenizer.decode([96, 97], { skipSpecial: false })).toBe('<|bos|><|eos|>');
	});

	it('matches independent ordered merge passes, including overlapping repeated pairs', () => {
		for (const text of [
			'Once upon a time, there was a little girl named Lily.',
			'aaa aaaaaa abcabcabc 123123 !!!!',
			'No words may become merged sentences.\n\n New story.'
		])
			expect(tokenizer.encode(text)).toEqual(referenceEncode(text));
	});

	it('does not silently drop unsupported text or invalid token ids', () => {
		for (const text of ['café', 'a\tb', '🙂', 'a\rb'])
			expect(() => tokenizer.encode(text)).toThrow(/unsupported/);
		for (const id of [-1, 4096, NaN, 1.5]) expect(() => tokenizer.decode([id])).toThrow(/Invalid/);
	});

	it('elects equal-frequency pairs deterministically, independently of story ordering', () => {
		const first = trainTokenStoryBpe(['ab', 'ac', 'ab', 'ac'], dummyHash, 101);
		const reordered = trainTokenStoryBpe(['ac', 'ab', 'ac', 'ab'], dummyHash, 101);
		expect(first.merges[0]).toEqual([
			TOKEN_STORY_BASE_CHARACTERS.indexOf('a'),
			TOKEN_STORY_BASE_CHARACTERS.indexOf('b')
		]);
		// Different leading-space frequencies can differ: use exactly permuted complete stories.
		expect(trainTokenStoryBpe(['ab ab', 'ac ac'], dummyHash, 104)).toEqual(
			trainTokenStoryBpe(['ac ac', 'ab ab'], dummyHash, 104)
		);
		expect(reordered.merges[0]).toEqual(first.merges[0]);
	});

	it('cannot merge words, newlines or BOS/EOS into normal vocabulary pieces', () => {
		expect(data.pieces.slice(98).every((piece) => tokenStoryWordPieces(piece).length === 1)).toBe(
			true
		);
		expect(data.merges.flat().some((id) => id === 96 || id === 97)).toBe(false);
		const trained = trainTokenStoryBpe(['a b\na b\na b'], dummyHash, 110);
		expect(trained.pieces.some((piece) => piece === 'a b' || piece === 'b\na')).toBe(false);
	});

	it('authenticates merge identity and rejects forged dependency/piece tables', async () => {
		await expect(validateTokenStoryTokenizerIdentity(data)).resolves.toBeUndefined();
		const forgedIdentity = structuredClone(data);
		forgedIdentity.id = `tinystories-bpe-v1:${'0'.repeat(64)}`;
		await expect(validateTokenStoryTokenizerIdentity(forgedIdentity)).rejects.toThrow(/identity/);
		const forgedPiece = structuredClone(data);
		forgedPiece.pieces[98] = 'forged';
		expect(() => createTokenStoryTokenizer(forgedPiece)).toThrow(/pieces/);
		const forgedMerge = structuredClone(data);
		forgedMerge.merges[0][0] = 99;
		expect(() => createTokenStoryTokenizer(forgedMerge)).toThrow(/dependency/);
	});

	it('verifies shipped corpus hashes, complete story boundaries and exact character round trips', () => {
		const tokenBytes = readFileSync('static/data/tinystories-bpe/tokens.bin');
		expect(digest(tokenBytes)).toBe(metadata.tokens.sha256);
		expect(digest(readFileSync('static/data/tinystories-bpe/tokenizer.json'))).toBe(
			metadata.tokenizer.sha256
		);
		const tokens = Array.from({ length: tokenBytes.length / 2 }, (_, i) =>
			tokenBytes.readUInt16LE(i * 2)
		);
		const texts: Record<string, string[]> = {};
		let expectedOffset = 0;
		for (const [name, split] of Object.entries(metadata.splits)) {
			expect(split.offset).toBe(expectedOffset);
			texts[name] = [];
			for (let i = 0; i < split.stories; i++) {
				expect(split.storyOffsets[i]).toBe(expectedOffset);
				const story = tokens.slice(expectedOffset, expectedOffset + split.storyLengths[i]);
				expect(story[0]).toBe(96);
				expect(story.at(-1)).toBe(97);
				expect(story.slice(1, -1).some((id) => id === 96 || id === 97)).toBe(false);
				const text = tokenizer.decode(story);
				expect(text.length).toBe(split.storyCharacters[i]);
				expect(tokenizer.encode(text, { bos: true, eos: true })).toEqual(story);
				texts[name].push(text);
				expectedOffset += story.length;
			}
			expect(expectedOffset).toBe(split.offset + split.length);
		}
		expect(expectedOffset).toBe(metadata.tokens.count);
		const oldMeta = JSON.parse(readFileSync('static/data/tinystories/corpus.json', 'utf8'));
		const oldBytes = readFileSync('static/data/tinystories/tokens.bin');
		const oldText = Array.from(oldBytes, (id) => oldMeta.chars[id]).join('');
		const reconstructed =
			[...texts.train, ...texts.calibration, ...texts.evaluation].join('\n\n') + '\n\n';
		expect(reconstructed).toBe(oldText);
		expect(texts.train.length).toBe(2097);
		expect(texts.calibration.length).toBe(32);
		expect(texts.evaluation.length).toBe(152);
		expect(new Set([...texts.calibration, ...texts.evaluation]).size).toBe(184);
		expect(
			[...texts.calibration, ...texts.evaluation].some((text) => texts.train.includes(text))
		).toBe(false);
	});

	it('reproduces the shipped vocabulary using training stories alone', async () => {
		const bytes = readFileSync('static/data/tinystories-bpe/tokens.bin');
		const split = metadata.splits.train;
		const trainingStories = split.storyOffsets.map((offset, i) =>
			tokenizer.decode(
				Array.from({ length: split.storyLengths[i] }, (_, j) =>
					bytes.readUInt16LE((offset + j) * 2)
				)
			)
		);
		const hash = digest(trainingStories.join('\n\n') + '\n\n');
		expect(hash).toBe(data.trainingTextSha256);
		const reproduced = await identifyTokenStoryTokenizer(trainTokenStoryBpe(trainingStories, hash));
		expect(reproduced).toEqual(data);
	}, 60000);
});
