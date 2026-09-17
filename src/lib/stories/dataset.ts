import {
	STORY_CHARACTERS,
	type StoryConfig,
	type StoryCorpusInfo,
	type StoryEncodedPrompt
} from './protocol';

export const STORY_CORPUS_SHA256 =
	'67fb35357ec45562152b012c60d4a2d1aec5a95ad9d9a4fc6d534bc97fb4c6db';
export const STORY_CORPUS_ID = `tinystories-ascii-v1:${STORY_CORPUS_SHA256}`;
export interface StoryCorpus {
	info: StoryCorpusInfo;
	tokens: Uint8Array;
	calibration: Uint8Array[];
	evaluation: Uint8Array[];
	unigram: Float64Array;
}
export class StoryRandom {
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
export const decodeStory = (ids: ArrayLike<number>) =>
	Array.from(ids, (id) => STORY_CHARACTERS[id] ?? '').join('');
export function encodeStoryPrompt(original: string, context: number): StoryEncodedPrompt {
	if (typeof original !== 'string' || original.length > 100000)
		throw new Error('Prompt must be at most 100,000 characters');
	const characters = [...original];
	const unsupportedCharacters = [
		...new Set(characters.filter((c) => !STORY_CHARACTERS.includes(c)))
	];
	if (unsupportedCharacters.length)
		throw new Error(
			`Unsupported prompt characters: ${unsupportedCharacters.map((c) => JSON.stringify(c)).join(', ')}. Use newline and printable ASCII.`
		);
	if (!characters.length) throw new Error('Enter at least one prompt character');
	const text = characters.slice(-context).join('');
	return {
		original,
		text,
		tokenIds: [...text].map((c) => STORY_CHARACTERS.indexOf(c)),
		truncatedCharacters: Math.max(0, characters.length - context),
		unsupportedCharacters
	};
}
export async function prepareStoryCorpus(
	tokens: Uint8Array,
	metadata: Record<string, unknown>,
	config: StoryConfig
): Promise<StoryCorpus> {
	const digest = await crypto.subtle.digest('SHA-256', Uint8Array.from(tokens));
	const hash = Array.from(new Uint8Array(digest), (v) => v.toString(16).padStart(2, '0')).join('');
	if (hash !== STORY_CORPUS_SHA256)
		throw new Error('TinyStories corpus checksum does not match the recorded source');
	if (
		metadata.trainTokens !== 1773785 ||
		metadata.validationTokens !== 130152 ||
		metadata.trainStories !== 2097 ||
		metadata.validationStories !== 184 ||
		JSON.stringify(metadata.chars) !== JSON.stringify(STORY_CHARACTERS) ||
		tokens.length !== 1903937
	)
		throw new Error('TinyStories corpus metadata does not match the recorded source');
	const validationText = decodeStory(tokens.subarray(1773785));
	const stories = validationText.split('\n\n').filter(Boolean);
	if (stories.length !== 184 || stories.some((s) => s.length < config.context + 1))
		throw new Error('TinyStories held-out story boundaries are inconsistent');
	const calibrationStories = stories.slice(0, 32),
		evaluationStories = stories.slice(32);
	const windows = (source: string[], count: number, seed: number) => {
		const random = new StoryRandom(seed);
		return Array.from({ length: count }, (_, i) => {
			const story = source[Math.floor((i * source.length) / count)];
			const start = Math.floor(random.next() * (story.length - config.context));
			return Uint8Array.from(story.slice(start, start + config.context + 1), (c) =>
				STORY_CHARACTERS.indexOf(c)
			);
		});
	};
	// Add-one smoothing over training characters only keeps unseen characters finite.
	const unigram = new Float64Array(96).fill(1);
	for (const id of tokens.subarray(0, 1773785)) unigram[id]++;
	for (let id = 0; id < unigram.length; id++) unigram[id] /= 1773785 + 96;
	return {
		info: {
			id: STORY_CORPUS_ID,
			dataset: 'TinyStories',
			trainTokens: 1773785,
			calibrationTokens: calibrationStories.reduce((n, s) => n + s.length + 2, 0),
			evaluationTokens: evaluationStories.reduce((n, s) => n + s.length + 2, 0),
			trainStories: 2097,
			calibrationStories: 32,
			evaluationStories: 152,
			vocabulary: [...STORY_CHARACTERS],
			source: String(metadata.source),
			license: String(metadata.license)
		},
		tokens,
		calibration: windows(calibrationStories, 8, 0x73746361),
		evaluation: windows(evaluationStories, 16, 0x73746576),
		unigram
	};
}
export async function loadStoryCorpus(config: StoryConfig): Promise<StoryCorpus> {
	const signal = AbortSignal.timeout(30000);
	const [bytes, metadata] = await Promise.all([
		fetch('/data/tinystories/tokens.bin', { signal }),
		fetch('/data/tinystories/corpus.json', { signal })
	]);
	if (!bytes.ok || !metadata.ok) throw new Error('TinyStories data could not be loaded');
	return prepareStoryCorpus(
		new Uint8Array(await bytes.arrayBuffer()),
		await metadata.json(),
		config
	);
}
