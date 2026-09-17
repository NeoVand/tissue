/** Deterministic, word-boundary BPE for the attributed TinyStories subset.
 * The merge vocabulary is fitted on training stories only. ASCII is explicit:
 * unsupported input fails rather than disappearing from a prompt. The boundary
 * rule follows Jaxverse's src/lib/data/bpe.ts; training and ranked encoding here
 * are independently implemented, with explicit story tokens and hashed identity. */
export const TOKEN_STORY_BASE_CHARACTERS = [
	'\n',
	...Array.from({ length: 95 }, (_, i) => String.fromCharCode(i + 32))
];
export const TOKEN_STORY_BOS_ID = 96;
export const TOKEN_STORY_EOS_ID = 97;
export const TOKEN_STORY_VOCABULARY_SIZE = 4096;
const SPECIAL_PIECES = ['<|bos|>', '<|eos|>'];
const BASE_SIZE = TOKEN_STORY_BASE_CHARACTERS.length + SPECIAL_PIECES.length;
const PAIR_STRIDE = 65536;

export interface TokenStoryTokenizerDefinition {
	version: 1;
	kind: 'word-boundary-bpe';
	baseCharacters: string[];
	merges: [number, number][];
	pieces: string[];
	vocabularySize: number;
	bosId: number;
	eosId: number;
	trainingTextSha256: string;
}
export interface TokenStoryTokenizerData extends TokenStoryTokenizerDefinition {
	id: string;
}
export interface TokenStoryCorpusSplit {
	/** Absolute offsets into the complete Uint16 token array. */
	offset: number;
	length: number;
	stories: number;
	characters: number;
	storyOffsets: number[];
	/** Includes each story's BOS and EOS. */
	storyLengths: number[];
	storyCharacters: number[];
}
export interface TokenStoryCorpusMetadata {
	version: 1;
	id: string;
	dataset: 'TinyStories';
	authors: string;
	source: string;
	sourceRevision: string;
	license: string;
	licenseUrl: string;
	tokenizer: {
		file: string;
		id: string;
		sha256: string;
		vocabularySize: number;
		bosId: number;
		eosId: number;
	};
	tokens: {
		file: string;
		encoding: 'uint16-le';
		count: number;
		bytes: number;
		sha256: string;
	};
	splits: {
		train: TokenStoryCorpusSplit;
		calibration: TokenStoryCorpusSplit;
		evaluation: TokenStoryCorpusSplit;
	};
	trainingCharactersPerTextToken: number;
	evaluationCharactersPerTextToken: number;
	preparation: string;
	limitations: string;
}
export interface TokenStoryTokenizer {
	id: string;
	vocabularySize: number;
	bosId: number;
	eosId: number;
	pieces: string[];
	encode(text: string, options?: { bos?: boolean; eos?: boolean }): number[];
	decode(ids: ArrayLike<number>, options?: { skipSpecial?: boolean }): string;
	tokenPiece(id: number): string;
}

function characterClass(c: string): number {
	if (c === '\n') return 0;
	if (/[A-Za-z]/.test(c)) return 1;
	if (/[0-9]/.test(c)) return 2;
	return 3;
}

/** A single leading space can join the following run; words never join words. */
export function tokenStoryWordPieces(text: string): string[] {
	const pieces: string[] = [];
	let start = 0;
	for (let i = 0; i < text.length; i++) {
		const c = text[i];
		if (c !== '\n' && (c.charCodeAt(0) < 32 || c.charCodeAt(0) > 126))
			throw new Error(
				`TinyStories tokenizer supports newline and printable ASCII; unsupported ${JSON.stringify(c)} at character ${i}.`
			);
		const previous = text[i - 1];
		if (
			i > 0 &&
			(c === '\n' ||
				previous === '\n' ||
				c === ' ' ||
				(previous !== ' ' && characterClass(previous) !== characterClass(c)))
		) {
			pieces.push(text.slice(start, i));
			start = i;
		}
	}
	if (start < text.length) pieces.push(text.slice(start));
	return pieces;
}

/** Offline trainer. Weighted distinct pieces make full-corpus fitting practical.
 * Equal counts elect the smallest (leftId,rightId); pairs must occur twice. */
export function trainTokenStoryBpe(
	trainingStories: readonly string[],
	trainingTextSha256: string,
	vocabularySize = TOKEN_STORY_VOCABULARY_SIZE
): TokenStoryTokenizerDefinition {
	if (!/^[a-f0-9]{64}$/.test(trainingTextSha256))
		throw new Error('Training text SHA-256 is required.');
	if (!Number.isInteger(vocabularySize) || vocabularySize < BASE_SIZE || vocabularySize > 65535)
		throw new Error('BPE vocabulary size must be an integer between 98 and 65535.');
	const frequencies = new Map<string, number>();
	for (const story of trainingStories)
		for (const piece of tokenStoryWordPieces(story))
			frequencies.set(piece, (frequencies.get(piece) ?? 0) + 1);
	if (frequencies.size === 0) throw new Error('BPE training needs nonempty training stories.');
	const characterIds = new Map(TOKEN_STORY_BASE_CHARACTERS.map((c, i) => [c, i]));
	const words = Array.from(frequencies, ([word, frequency]) => ({
		ids: Array.from(word, (c) => characterIds.get(c)!),
		frequency
	}));
	const pieces = [...TOKEN_STORY_BASE_CHARACTERS, ...SPECIAL_PIECES];
	const merges: [number, number][] = [];
	while (pieces.length < vocabularySize) {
		const counts = new Map<number, number>();
		for (const { ids, frequency } of words)
			for (let i = 0; i + 1 < ids.length; i++) {
				const pair = ids[i] * PAIR_STRIDE + ids[i + 1];
				counts.set(pair, (counts.get(pair) ?? 0) + frequency);
			}
		let bestPair = -1;
		let bestCount = 1;
		for (const [pair, count] of counts)
			if (count > bestCount || (count === bestCount && count > 1 && pair < bestPair)) {
				bestPair = pair;
				bestCount = count;
			}
		if (bestPair < 0) break;
		const a = Math.floor(bestPair / PAIR_STRIDE);
		const b = bestPair % PAIR_STRIDE;
		const newId = pieces.length;
		for (const word of words) {
			let write = 0;
			for (let read = 0; read < word.ids.length; read++) {
				if (word.ids[read] === a && word.ids[read + 1] === b) {
					word.ids[write++] = newId;
					read++;
				} else word.ids[write++] = word.ids[read];
			}
			word.ids.length = write;
		}
		pieces.push(pieces[a] + pieces[b]);
		merges.push([a, b]);
	}
	return {
		version: 1,
		kind: 'word-boundary-bpe',
		baseCharacters: [...TOKEN_STORY_BASE_CHARACTERS],
		merges,
		pieces,
		vocabularySize: pieces.length,
		bosId: TOKEN_STORY_BOS_ID,
		eosId: TOKEN_STORY_EOS_ID,
		trainingTextSha256
	};
}

/** Canonical field order is part of the tokenizer identity contract. */
export function tokenStoryTokenizerCanonicalJSON(data: TokenStoryTokenizerDefinition): string {
	return JSON.stringify({
		version: data.version,
		kind: data.kind,
		baseCharacters: data.baseCharacters,
		merges: data.merges,
		pieces: data.pieces,
		vocabularySize: data.vocabularySize,
		bosId: data.bosId,
		eosId: data.eosId,
		trainingTextSha256: data.trainingTextSha256
	});
}

export async function identifyTokenStoryTokenizer(
	definition: TokenStoryTokenizerDefinition
): Promise<TokenStoryTokenizerData> {
	const bytes = new TextEncoder().encode(tokenStoryTokenizerCanonicalJSON(definition));
	const digest = await crypto.subtle.digest('SHA-256', bytes);
	const hash = Array.from(new Uint8Array(digest), (n) => n.toString(16).padStart(2, '0')).join('');
	return { ...definition, id: `tinystories-bpe-v1:${hash}` };
}

export async function validateTokenStoryTokenizerIdentity(
	data: TokenStoryTokenizerData
): Promise<void> {
	validateDefinition(data);
	const identified = await identifyTokenStoryTokenizer(data);
	if (identified.id !== data.id)
		throw new Error('TinyStories BPE tokenizer identity does not match its vocabulary.');
}

function validateDefinition(data: TokenStoryTokenizerDefinition): void {
	if (
		!data ||
		data.version !== 1 ||
		data.kind !== 'word-boundary-bpe' ||
		JSON.stringify(data.baseCharacters) !== JSON.stringify(TOKEN_STORY_BASE_CHARACTERS) ||
		data.bosId !== TOKEN_STORY_BOS_ID ||
		data.eosId !== TOKEN_STORY_EOS_ID ||
		!Number.isInteger(data.vocabularySize) ||
		data.vocabularySize < BASE_SIZE ||
		data.vocabularySize > 65535 ||
		!Array.isArray(data.merges) ||
		data.merges.length !== data.vocabularySize - BASE_SIZE ||
		!Array.isArray(data.pieces) ||
		data.pieces.length !== data.vocabularySize ||
		!/^[a-f0-9]{64}$/.test(data.trainingTextSha256)
	)
		throw new Error('Malformed TinyStories BPE tokenizer.');
	const table = [...TOKEN_STORY_BASE_CHARACTERS, ...SPECIAL_PIECES];
	const seen = new Set<number>();
	for (let i = 0; i < data.merges.length; i++) {
		const pair = data.merges[i];
		if (
			!Array.isArray(pair) ||
			pair.length !== 2 ||
			pair.some(
				(id) =>
					!Number.isInteger(id) ||
					id < 0 ||
					id >= BASE_SIZE + i ||
					id === data.bosId ||
					id === data.eosId
			)
		)
			throw new Error('Malformed BPE merge dependency.');
		const key = pair[0] * PAIR_STRIDE + pair[1];
		const piece = table[pair[0]] + table[pair[1]];
		if (seen.has(key) || tokenStoryWordPieces(piece).length !== 1)
			throw new Error('BPE merge repeats or crosses a word boundary.');
		seen.add(key);
		table.push(piece);
	}
	if (table.some((piece, id) => piece !== data.pieces[id]))
		throw new Error('BPE pieces do not match their merge dependencies.');
}

export function createTokenStoryTokenizer(data: TokenStoryTokenizerData): TokenStoryTokenizer {
	validateDefinition(data);
	if (!/^tinystories-bpe-v1:[a-f0-9]{64}$/.test(data.id))
		throw new Error('Missing BPE tokenizer identity.');
	const pieces = [...data.pieces];
	const ids = new Map(TOKEN_STORY_BASE_CHARACTERS.map((c, i) => [c, i]));
	const ranks = new Map(data.merges.map(([a, b], rank) => [a * PAIR_STRIDE + b, rank]));
	const cache = new Map<string, number[]>();
	const checkId = (id: number) => {
		if (!Number.isInteger(id) || id < 0 || id >= pieces.length)
			throw new Error(`Invalid TinyStories BPE token id: ${id}.`);
	};
	return {
		id: data.id,
		vocabularySize: pieces.length,
		bosId: data.bosId,
		eosId: data.eosId,
		pieces,
		encode(text, options = {}) {
			const output: number[] = options.bos ? [data.bosId] : [];
			for (const piece of tokenStoryWordPieces(text)) {
				let encoded = cache.get(piece);
				if (!encoded) {
					encoded = Array.from(piece, (c) => ids.get(c)!);
					while (encoded.length > 1) {
						let bestRank = Infinity;
						let bestIndex = -1;
						for (let i = 0; i + 1 < encoded.length; i++) {
							const rank = ranks.get(encoded[i] * PAIR_STRIDE + encoded[i + 1]);
							if (rank !== undefined && rank < bestRank) {
								bestRank = rank;
								bestIndex = i;
							}
						}
						if (bestIndex < 0) break;
						encoded.splice(bestIndex, 2, BASE_SIZE + bestRank);
					}
					if (cache.size < 8192) cache.set(piece, encoded);
				}
				output.push(...encoded);
			}
			if (options.eos) output.push(data.eosId);
			return output;
		},
		decode(tokenIds, options = {}) {
			let output = '';
			for (let i = 0; i < tokenIds.length; i++) {
				const id = tokenIds[i];
				checkId(id);
				if (options.skipSpecial !== false && (id === data.bosId || id === data.eosId)) continue;
				output += pieces[id];
			}
			return output;
		},
		tokenPiece(id) {
			checkId(id);
			return pieces[id];
		}
	};
}
