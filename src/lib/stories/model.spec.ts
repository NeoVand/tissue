import { readFile } from 'node:fs/promises';
import { beforeAll, describe, expect, it } from 'vitest';
import { defaultDevice, init, jit, tree } from '@jax-js/jax';
import { decodeStory, encodeStoryPrompt, prepareStoryCorpus, type StoryCorpus } from './dataset';
import {
	initializeStoryParameters,
	serializeStoryTree,
	storyBatch,
	storyForward,
	storyMask
} from './model';
import {
	STORY_PRESETS,
	storyParameterCount,
	storyParameterShapes,
	validateStoryCheckpoint,
	validateStoryConfig,
	type StoryConfig
} from './protocol';
import { StoryRuntime } from './runtime';

const config: StoryConfig = {
	layers: 2,
	width: 8,
	heads: 2,
	hidden: 32,
	context: 16,
	batchSize: 2,
	vocabularySize: 96,
	learningRate: 0.01
};
let corpus: StoryCorpus;
beforeAll(async () => {
	const bytes = new Uint8Array(await readFile('static/data/tinystories/tokens.bin'));
	const metadata = JSON.parse(await readFile('static/data/tinystories/corpus.json', 'utf8'));
	corpus = await prepareStoryCorpus(bytes, metadata, config);
	await init('wasm');
	defaultDevice('wasm');
});
describe('TinyStories scientific contracts', () => {
	it('pins preset parameter counts and validates bounded architecture dimensions', () => {
		expect(Object.values(STORY_PRESETS).map(storyParameterCount)).toEqual([
			827392, 3227648, 10739712
		]);
		for (const preset of Object.values(STORY_PRESETS))
			expect(() => validateStoryConfig(preset)).not.toThrow();
		expect(() => validateStoryConfig({ ...config, heads: 3 })).toThrow();
		expect(() => validateStoryConfig({ ...config, layers: 7 })).toThrow();
	});
	it('holds complete validation stories out of training, then splits calibration from evaluation', () => {
		const train = decodeStory(corpus.tokens.subarray(0, corpus.info.trainTokens));
		const stories = decodeStory(corpus.tokens.subarray(corpus.info.trainTokens))
			.split('\n\n')
			.filter(Boolean);
		expect(stories).toHaveLength(184);
		for (const story of stories) expect(train.includes(story)).toBe(false);
		expect(corpus.calibration).toHaveLength(8);
		expect(corpus.evaluation).toHaveLength(16);
		for (const window of corpus.calibration)
			expect(stories.slice(0, 32).some((story) => story.includes(decodeStory(window)))).toBe(true);
		for (const window of corpus.evaluation)
			expect(stories.slice(32).some((story) => story.includes(decodeStory(window)))).toBe(true);
		expect(corpus.unigram.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 12);
		const trainSpace = corpus.tokens
			.subarray(0, corpus.info.trainTokens)
			.filter((id) => id === 1).length;
		expect(corpus.unigram[1]).toBe((trainSpace + 1) / (corpus.info.trainTokens + 96));
	});
	it('rejects unsupported characters and records context truncation explicitly', () => {
		expect(() => encodeStoryPrompt('café', 128)).toThrow(/Unsupported.*é/);
		expect(() => encodeStoryPrompt('', 128)).toThrow(/at least one/);
		const prompt = encodeStoryPrompt('Once upon a time', 8);
		expect(prompt.text).toBe('n a time');
		expect(prompt.truncatedCharacters).toBe(8);
		expect(decodeStory(prompt.tokenIds)).toBe(prompt.text);
	});
	it('retains exact tensor leaf order, intact capture parity, causal prefixes, and all-position ablation', async () => {
		const p = initializeStoryParameters(config, 42);
		const infer = jit((params, tokens, pos, mask) =>
			storyForward(params, config, tokens, pos, mask)
		);
		const capture = jit((params, tokens, pos, mask) =>
			storyForward(params, config, tokens, pos, mask, true)
		);
		try {
			expect((await serializeStoryTree(p)).map((leaf) => leaf.shape)).toEqual(
				storyParameterShapes(config)
			);
			const run = async (window: ArrayLike<number>, captured: boolean, neuron?: number) => {
				const batch = storyBatch(config, [window]);
				batch.targets.dispose();
				const out = (captured ? capture : infer)(
					tree.ref(p),
					batch.tokens,
					batch.positions,
					storyMask(config, neuron)
				);
				return {
					logprobs: await out.logprobs.data(),
					layers: await Promise.all(
						out.activations.map((a: { data: () => Promise<Float32Array> }) => a.data())
					)
				};
			};
			const intact = await run(corpus.calibration[0], false),
				observed = await run(corpus.calibration[0], true);
			expect(observed.logprobs).toEqual(intact.logprobs);
			const changed = Uint8Array.from(corpus.calibration[0]);
			changed[config.context - 1] = (changed[config.context - 1] + 1) % 96;
			const future = await run(changed, true);
			for (let layer = 0; layer < config.layers; layer++)
				expect(future.layers[layer].subarray(0, (config.context - 1) * config.hidden)).toEqual(
					observed.layers[layer].subarray(0, (config.context - 1) * config.hidden)
				);
			const lesion = await run(corpus.calibration[0], true, 35);
			expect(
				Array.from({ length: config.context }, (_, t) => lesion.layers[1][t * config.hidden + 3])
			).toEqual(Array(config.context).fill(0));
			expect(lesion.layers[0]).toEqual(observed.layers[0]);
			expect(() => storyMask(config, 64)).toThrow();
		} finally {
			tree.dispose(p);
			infer.dispose();
			capture.dispose();
		}
	}, 30000);
	it('learns held-out characters and resumes exact Adam/RNG state after atlas, probe, and sampling', async () => {
		const runtime = new StoryRuntime();
		try {
			const initial = await runtime.initialize(config, 42, 'wasm', corpus);
			expect(initial.metrics.validationLoss).toBeCloseTo(Math.log(96), 1);
			const trained = await runtime.train(20);
			expect(trained.validationLoss).toBeLessThan(initial.metrics.validationLoss - 0.5);
			const checkpoint = await runtime.exportCheckpoint();
			validateStoryCheckpoint(checkpoint);
			await runtime.train(2);
			const reference = await runtime.exportCheckpoint();
			await runtime.loadCheckpoint(checkpoint, 'wasm');
			const atlas = await runtime.captureAtlas();
			expect(atlas.fingerprints).toHaveLength(64 * 128);
			const probe = await runtime.probe(atlas.examples[0]);
			for (let unit = 0; unit < 64; unit++)
				for (let position = 0; position < 16; position++)
					expect(atlas.fingerprints[unit * 128 + position]).toBe(
						probe.activations[atlas.positions[position] * 64 + unit]
					);
			const generated = await runtime.generate('Once', { maxTokens: 4, seed: 25 });
			expect(generated.tokenIds).toHaveLength(4);
			expect(await runtime.exportCheckpoint()).toEqual(checkpoint);
			await runtime.train(2);
			const resumed = await runtime.exportCheckpoint();
			expect(resumed.trainRngState).toBe(reference.trainRngState);
			expect(resumed.parameters).toEqual(reference.parameters);
			expect(resumed.optimizer).toEqual(reference.optimizer);
			const corrupt = structuredClone(checkpoint);
			corrupt.optimizer.v[0].values[0] = -1;
			expect(() => validateStoryCheckpoint(corrupt)).toThrow(/tensor 0/);
		} finally {
			runtime.dispose();
		}
	}, 120000);
});
