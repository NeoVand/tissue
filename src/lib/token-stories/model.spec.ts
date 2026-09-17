import { readFile } from 'node:fs/promises';
import { beforeAll, describe, expect, it } from 'vitest';
import { defaultDevice, init, jit, numpy as np, tree } from '@jax-js/jax';
import {
	prepareTokenStoryCorpus,
	sampleTokenStoryTrainingWindow,
	TokenStoryRandom,
	encodeTokenStoryPrompt,
	type TokenStoryCorpus
} from './dataset';
import {
	TOKEN_STORY_PRESETS,
	tokenStoryParameterCount,
	tokenStoryParameterShapes,
	validateTokenStoryCheckpoint,
	type TokenStoryConfig
} from './protocol';
import {
	initializeTokenStoryParameters,
	tokenStoryBatch,
	tokenStoryForward,
	tokenStoryLoss,
	tokenStoryMask,
	serializeTokenStoryTree
} from './model';
import { TokenStoryRuntime } from './runtime';
const config: TokenStoryConfig = {
	layers: 2,
	width: 8,
	heads: 2,
	hidden: 32,
	context: 16,
	batchSize: 2,
	vocabularySize: 4096,
	learningRate: 0.01
};
let corpus: TokenStoryCorpus;
beforeAll(async () => {
	const bytes = new Uint8Array(await readFile('static/data/tinystories-bpe/tokens.bin'));
	const metadata = JSON.parse(await readFile('static/data/tinystories-bpe/corpus.json', 'utf8'));
	const tokenizer = JSON.parse(
		await readFile('static/data/tinystories-bpe/tokenizer.json', 'utf8')
	);
	corpus = await prepareTokenStoryCorpus(bytes, metadata, tokenizer, config);
	await init('wasm');
	defaultDevice('wasm');
});
describe('BPE story corpus and supervision', () => {
	it('covers every training next-token transition once before weighted sampling; BOS never becomes a target', () => {
		const split = corpus.info.splits.train,
			coverage = new Uint8Array(split.length);
		let previous = 0;
		for (let i = 0; i < corpus.trainingWindows.length; i++) {
			const w = corpus.trainingWindows[i],
				start = split.storyOffsets[w.story],
				end = start + split.storyLengths[w.story];
			expect(w.offset).toBeGreaterThanOrEqual(start);
			expect(w.offset + w.length).toBeLessThanOrEqual(end);
			expect(corpus.trainingCumulative[i] - previous).toBe(w.length - 1);
			previous = corpus.trainingCumulative[i];
			for (let j = 1; j < w.length; j++) coverage[w.offset + j]++;
		}
		for (let story = 0; story < split.stories; story++) {
			const start = split.storyOffsets[story],
				end = start + split.storyLengths[story];
			expect(coverage[start]).toBe(0);
			expect(coverage.subarray(start + 1, end).every((x) => x === 1)).toBe(true);
		}
		expect(previous).toBe(split.length - split.stories);
		const random = new TokenStoryRandom(9);
		for (let i = 0; i < 100; i++) {
			const window = sampleTokenStoryTrainingWindow(corpus, random);
			expect(window.length).toBeLessThanOrEqual(config.context + 1);
			expect(Array.from(window).slice(1)).not.toContain(96);
		}
		expect(corpus.unigram[96]).toBe(1 / (previous + 4096));
		expect(corpus.unigram[97]).toBe((2097 + 1) / (previous + 4096));
	});
	it('pins parameter totals and retains BOS/truncation and exact visible BPE pieces', () => {
		expect(Object.values(TOKEN_STORY_PRESETS).map(tokenStoryParameterCount)).toEqual([
			1851392, 5308416, 13860864
		]);
		const prompt = encodeTokenStoryPrompt(
			'Once upon a time, there was a little fox.',
			4,
			corpus.tokenizer
		);
		expect(prompt.tokenIds).toHaveLength(4);
		expect(prompt.truncatedTokens).toBeGreaterThan(0);
		expect(prompt.includesBos).toBe(false);
		expect(prompt.pieces).toEqual(prompt.tokenIds.map((id) => corpus.tokenizer.tokenPiece(id)));
		expect(encodeTokenStoryPrompt('', 16, corpus.tokenizer).tokenIds).toEqual([96]);
		expect(() => encodeTokenStoryPrompt('café', 16, corpus.tokenizer)).toThrow(/unsupported/);
	});
});
describe('BPE transformer numerical contracts', () => {
	it('masks padded loss and averages windows equally, matching the token-weighted sampling objective', async () => {
		const c = { ...config, context: 8, vocabularySize: 98 },
			p = initializeTokenStoryParameters(c, 3);
		p.lmHead = p.lmHead.mul(200);
		const windows = [
			[96, 4, 97],
			[96, 5, 6, 7, 8, 9, 10, 11, 97]
		];
		const predict = jit((params, tokens, positions, mask) =>
			tokenStoryForward(params, c, tokens, positions, mask)
		);
		const loss = jit((params, tokens, positions, targets, weights, mask) =>
			tokenStoryLoss(params, c, tokens, positions, targets, weights, mask)
		);
		try {
			const input = tokenStoryBatch(c, windows);
			input.targets.dispose();
			input.lossMask.dispose();
			const output = predict(tree.ref(p), input.tokens, input.positions, tokenStoryMask(c));
			const lp = await output.logprobs.data();
			let expected = 0;
			windows.forEach((window, b) => {
				let total = 0;
				for (let t = 0; t < window.length - 1; t++) total -= lp[(b * 8 + t) * 98 + window[t + 1]];
				expected += total / (window.length - 1) / 2;
			});
			const batch = tokenStoryBatch(c, windows);
			const measured = await loss(
				tree.ref(p),
				batch.tokens,
				batch.positions,
				batch.targets,
				batch.lossMask,
				tokenStoryMask(c)
			).data();
			expect(measured[0]).toBeCloseTo(expected, 5);
			// Changing only future padding cannot change the two supervised positions in row0.
			const changed = tokenStoryBatch(c, [[96, 4, 97, 7, 8, 9, 10, 11, 12], windows[1]]);
			changed.lossMask.dispose();
			const masked = np
				.array(new Float32Array([1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]))
				.reshape([2, 8]);
			const altered = await loss(
				tree.ref(p),
				changed.tokens,
				changed.positions,
				changed.targets,
				masked,
				tokenStoryMask(c)
			).data();
			expect(altered[0]).toBeCloseTo(measured[0], 6);
		} finally {
			predict.dispose();
			loss.dispose();
			tree.dispose(p);
		}
	});
	it('capture agrees with inference, future input leaves prefixes unchanged, and lesions zero only their exact layer/channel', async () => {
		const p = initializeTokenStoryParameters(config, 42),
			c = config;
		const infer = jit((params, tokens, pos, mask) =>
			tokenStoryForward(params, c, tokens, pos, mask)
		);
		const capture = jit((params, tokens, pos, mask) =>
			tokenStoryForward(params, c, tokens, pos, mask, true)
		);
		const run = async (window: ArrayLike<number>, captured: boolean, neuron?: number) => {
			const b = tokenStoryBatch(c, [window]);
			b.targets.dispose();
			b.lossMask.dispose();
			const result = (captured ? capture : infer)(
				tree.ref(p),
				b.tokens,
				b.positions,
				tokenStoryMask(c, neuron)
			);
			return {
				logprobs: await result.logprobs.data(),
				layers: await Promise.all(
					result.activations.map((a: { data: () => Promise<Float32Array> }) => a.data())
				)
			};
		};
		try {
			expect((await serializeTokenStoryTree(p)).map((x) => x.shape)).toEqual(
				tokenStoryParameterShapes(c)
			);
			const original = await run(corpus.calibration[0], true),
				uncaptured = await run(corpus.calibration[0], false);
			expect(original.logprobs).toEqual(uncaptured.logprobs);
			const changed = corpus.calibration[0].slice();
			changed[c.context - 1] = (changed[c.context - 1] + 1) % 4096;
			const later = await run(changed, true);
			for (let layer = 0; layer < c.layers; layer++)
				expect(later.layers[layer].subarray(0, (c.context - 1) * c.hidden)).toEqual(
					original.layers[layer].subarray(0, (c.context - 1) * c.hidden)
				);
			const lesion = await run(corpus.calibration[0], true, 35);
			expect(lesion.layers[0]).toEqual(original.layers[0]);
			expect(
				Array.from({ length: c.context }, (_, t) => lesion.layers[1][t * c.hidden + 3])
			).toEqual(Array(c.context).fill(0));
		} finally {
			tree.dispose(p);
			infer.dispose();
			capture.dispose();
		}
	});
	it('learns held-out token loss and resumes exact optimizer/RNG after probe, atlas, and generation', async () => {
		const runtime = new TokenStoryRuntime();
		try {
			const initial = await runtime.initialize(config, 42, 'wasm', corpus);
			expect(initial.metrics.validationLoss).toBeCloseTo(Math.log(4096), 1);
			const trained = await runtime.train(20);
			expect(trained.validationLoss).toBeLessThan(initial.metrics.validationLoss - 0.4);
			const checkpoint = await runtime.exportCheckpoint();
			validateTokenStoryCheckpoint(checkpoint);
			expect(checkpoint.trainedTokens).toBe(trained.trainedTokens);
			await runtime.train(2);
			const reference = await runtime.exportCheckpoint();
			await runtime.loadCheckpoint(checkpoint, 'wasm');
			const atlas = await runtime.captureAtlas();
			expect(atlas.fingerprints).toHaveLength(64 * 128);
			expect(
				atlas.tokenPositions.every((positions, i) =>
					positions.every((position) => position < atlas.tokenIds[i].length)
				)
			).toBe(true);
			const probe = await runtime.probe('Once upon a time');
			expect(probe.probabilities).toHaveLength(4096);
			expect(probe.unitAddresses[35]).toMatchObject({
				layer: 1,
				channel: 3,
				incomingTensor: 'layers.1.mlpFc1',
				outgoingRow: 3
			});
			const generated = await runtime.generate('Once', { maxTokens: 4, seed: 71 });
			expect(generated.tokenIds).not.toContain(96);
			expect(await runtime.exportCheckpoint()).toEqual(checkpoint);
			await runtime.train(2);
			const resumed = await runtime.exportCheckpoint();
			expect(resumed.parameters).toEqual(reference.parameters);
			expect(resumed.optimizer).toEqual(reference.optimizer);
			expect(resumed.trainRngState).toBe(reference.trainRngState);
			expect(resumed.trainedTokens).toBe(reference.trainedTokens);
			// An intentionally controlled checkpoint makes BOS strongest and EOS second.
			// Sampling must exclude BOS, emit EOS once, and finish without claiming cancellation.
			const controlled = structuredClone(checkpoint);
			controlled.parameters.forEach((leaf) => leaf.values.fill(0));
			controlled.parameters[0].values.fill(1);
			controlled.parameters[2].values[96] = 1000;
			controlled.parameters[2].values[97] = 100;
			await runtime.loadCheckpoint(controlled, 'wasm');
			const eos = await runtime.generate('', { maxTokens: 4, topK: 1 });
			expect(eos.tokenIds).toEqual([97]);
			expect(eos.stoppedOnEos).toBe(true);
			expect(eos.cancelled).toBe(false);
			expect(eos.completion).toBe('');
		} finally {
			runtime.dispose();
		}
	}, 120000);
});
