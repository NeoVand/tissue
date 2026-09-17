import { publicAsset } from '../deployment/public-assets';
import {
	createTokenStoryTokenizer,
	validateTokenStoryTokenizerIdentity,
	type TokenStoryTokenizer,
	type TokenStoryTokenizerData,
	type TokenStoryCorpusMetadata
} from './tokenizer';
import type { TokenStoryConfig, TokenStoryEncodedPrompt } from './protocol';
export const TOKEN_STORY_CORPUS_ID =
	'tinystories-bpe-corpus-v1:5ecd0780ff18a2c247f647025e89797a756c9e1a34f461298b9c0cebed50595c';
export const TOKEN_STORY_TOKENIZER_ID =
	'tinystories-bpe-v1:eeef1b989796da285c9051cf8fd0cf395071175777616db3f02ccf0c573a2dbd';
export const TOKEN_STORY_TOKENS_SHA256 =
	'1b0473b548c9fefaebb3c13a3bf341cf9abe64c598b00d3c27a1ae693f13aa05';
export interface TokenStoryTrainingWindow {
	offset: number;
	length: number;
	story: number;
}
export interface TokenStoryCorpus {
	info: TokenStoryCorpusMetadata;
	tokenizerData: TokenStoryTokenizerData;
	tokenizer: TokenStoryTokenizer;
	tokens: Uint16Array;
	calibration: Uint16Array[];
	evaluation: Uint16Array[];
	trainingWindows: TokenStoryTrainingWindow[];
	trainingCumulative: Float64Array;
	unigram: Float64Array;
}
export class TokenStoryRandom {
	constructor(public state: number) {
		this.state >>>= 0;
	}
	next() {
		this.state = (this.state + 0x6d2b79f5) >>> 0;
		let t = this.state;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	}
}
export const encodeTokenStoryPrompt = (
	original: string,
	context: number,
	tokenizer: TokenStoryTokenizer
): TokenStoryEncodedPrompt => {
	if (typeof original !== 'string' || original.length > 100000)
		throw new Error('Prompt must be at most100,000 characters');
	const full = tokenizer.encode(original, { bos: true });
	const tokenIds = full.slice(-context);
	return {
		original,
		text: tokenizer.decode(tokenIds),
		tokenIds,
		pieces: tokenIds.map((id) => tokenizer.tokenPiece(id)),
		truncatedTokens: full.length - tokenIds.length,
		unsupportedCharacters: [],
		includesBos: tokenIds[0] === tokenizer.bosId
	};
};
async function hash(bytes: Uint8Array) {
	return Array.from(
		new Uint8Array(await crypto.subtle.digest('SHA-256', Uint8Array.from(bytes))),
		(v) => v.toString(16).padStart(2, '0')
	).join('');
}
export async function prepareTokenStoryCorpus(
	bytes: Uint8Array,
	info: TokenStoryCorpusMetadata,
	tokenizerData: TokenStoryTokenizerData,
	config: TokenStoryConfig
): Promise<TokenStoryCorpus> {
	if ((await hash(bytes)) !== TOKEN_STORY_TOKENS_SHA256 || bytes.length !== 953342)
		throw new Error('BPE TinyStories token checksum mismatch');
	await validateTokenStoryTokenizerIdentity(tokenizerData);
	const tokenizer = createTokenStoryTokenizer(tokenizerData);
	if (
		info.version !== 1 ||
		info.id !== TOKEN_STORY_CORPUS_ID ||
		info.dataset !== 'TinyStories' ||
		info.tokenizer.id !== TOKEN_STORY_TOKENIZER_ID ||
		tokenizer.id !== info.tokenizer.id ||
		tokenizer.vocabularySize !== config.vocabularySize ||
		info.tokens.sha256 !== TOKEN_STORY_TOKENS_SHA256 ||
		info.tokens.encoding !== 'uint16-le' ||
		info.tokens.bytes !== bytes.length ||
		info.tokens.count * 2 !== bytes.length
	)
		throw new Error('BPE TinyStories corpus identity mismatch');
	const identity = await hash(
		new TextEncoder().encode(
			JSON.stringify({
				tokenizerId: tokenizer.id,
				tokensSha256: info.tokens.sha256,
				splits: info.splits,
				asciiSourceSha256: '67fb35357ec45562152b012c60d4a2d1aec5a95ad9d9a4fc6d534bc97fb4c6db',
				sourceRevision: info.sourceRevision
			})
		)
	);
	if (`tinystories-bpe-corpus-v1:${identity}` !== info.id)
		throw new Error('BPE TinyStories story boundaries changed');
	const data = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength),
		tokens = new Uint16Array(bytes.length / 2);
	for (let i = 0; i < tokens.length; i++) {
		tokens[i] = data.getUint16(i * 2, true);
		if (tokens[i] >= tokenizer.vocabularySize) throw new Error('Invalid BPE token');
	}
	let end = 0;
	for (const [name, expected] of [
		['train', 2097],
		['calibration', 32],
		['evaluation', 152]
	] as const) {
		const split = info.splits[name];
		if (
			split.offset !== end ||
			split.stories !== expected ||
			split.storyOffsets.length !== expected ||
			split.storyLengths.length !== expected
		)
			throw new Error('Invalid BPE story split');
		for (let i = 0; i < expected; i++) {
			const start = split.storyOffsets[i],
				length = split.storyLengths[i];
			if (
				start !== end ||
				!Number.isInteger(length) ||
				length < 2 ||
				tokens[start] !== tokenizer.bosId ||
				tokens[start + length - 1] !== tokenizer.eosId
			)
				throw new Error('Invalid BPE story boundary');
			for (let j = start + 1; j < start + length - 1; j++)
				if (tokens[j] === tokenizer.bosId || tokens[j] === tokenizer.eosId)
					throw new Error('Special token inside story');
			end += length;
		}
		if (end !== split.offset + split.length) throw new Error('Invalid BPE split length');
	}
	if (end !== tokens.length) throw new Error('Unassigned BPE corpus tokens');
	const trainingWindows: TokenStoryTrainingWindow[] = [],
		cumulative: number[] = [],
		unigram = new Float64Array(tokenizer.vocabularySize).fill(1);
	let total = 0;
	const training = info.splits.train;
	for (let story = 0; story < training.stories; story++) {
		const offset = training.storyOffsets[story],
			length = training.storyLengths[story];
		for (let t = 1; t < length; t++) unigram[tokens[offset + t]]++;
		for (let start = 0; start < length - 1; start += config.context) {
			const count = Math.min(config.context, length - 1 - start);
			trainingWindows.push({ offset: offset + start, length: count + 1, story });
			total += count;
			cumulative.push(total);
		}
	}
	for (let i = 0; i < unigram.length; i++) unigram[i] /= total + unigram.length;
	const windows = (name: 'calibration' | 'evaluation', count: number, seed: number) => {
		const split = info.splits[name],
			random = new TokenStoryRandom(seed);
		return Array.from({ length: count }, (_, i) => {
			const story = Math.floor((i * split.stories) / count),
				length = split.storyLengths[story],
				size = Math.min(config.context + 1, length),
				start = Math.floor(random.next() * (length - size + 1));
			return tokens.slice(
				split.storyOffsets[story] + start,
				split.storyOffsets[story] + start + size
			);
		});
	};
	return {
		info,
		tokenizerData,
		tokenizer,
		tokens,
		trainingWindows,
		trainingCumulative: new Float64Array(cumulative),
		unigram,
		calibration: windows('calibration', 8, 0x62706361),
		evaluation: windows('evaluation', 16, 0x62706576)
	};
}
/** A window's draw probability is its valid target count. Mean per-window CE then gives an unbiased token objective without overweighting short tails. */
export function sampleTokenStoryTrainingWindow(
	corpus: TokenStoryCorpus,
	random: TokenStoryRandom
): Uint16Array {
	const cumulative = corpus.trainingCumulative,
		draw = random.next() * cumulative[cumulative.length - 1];
	let low = 0,
		high = cumulative.length - 1;
	while (low < high) {
		const mid = (low + high) >>> 1;
		if (draw < cumulative[mid]) high = mid;
		else low = mid + 1;
	}
	const window = corpus.trainingWindows[low];
	return corpus.tokens.subarray(window.offset, window.offset + window.length);
}
export async function loadTokenStoryCorpus(config: TokenStoryConfig): Promise<TokenStoryCorpus> {
	const signal = AbortSignal.timeout(30000),
		root = '/data/tinystories-bpe/';
	const [tokens, metadata, tokenizer] = await Promise.all(
		['tokens.bin', 'corpus.json', 'tokenizer.json'].map((file) =>
			fetch(publicAsset(root + file), { signal })
		)
	);
	if (!tokens.ok || !metadata.ok || !tokenizer.ok)
		throw new Error('BPE TinyStories data could not be loaded');
	return prepareTokenStoryCorpus(
		new Uint8Array(await tokens.arrayBuffer()),
		await metadata.json(),
		await tokenizer.json(),
		config
	);
}
