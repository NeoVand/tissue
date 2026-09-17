import { readFile } from 'node:fs/promises';
import { beforeAll, describe, expect, it } from 'vitest';
import { jit, tree } from '@jax-js/jax';
import { prepareTokenStoryCorpus, type TokenStoryCorpus } from './dataset';
import { LiveTokenStoryRuntime } from './live-runtime';
import { TokenStoryRuntime } from './runtime';
import type { LiveTokenStoryFrame } from './live-protocol';
import type { TokenStoryCheckpoint, TokenStoryConfig } from './protocol';
import {
	deserializeTokenStoryTree,
	tokenStoryBatch,
	tokenStoryForward,
	tokenStoryMask
} from './model';

const config: TokenStoryConfig = {
	layers: 2,
	width: 8,
	heads: 2,
	hidden: 32,
	context: 8,
	batchSize: 2,
	vocabularySize: 4096,
	learningRate: 0.01
};
let corpus: TokenStoryCorpus;
beforeAll(async () => {
	corpus = await prepareTokenStoryCorpus(
		new Uint8Array(await readFile('static/data/tinystories-bpe/tokens.bin')),
		JSON.parse(await readFile('static/data/tinystories-bpe/corpus.json', 'utf8')),
		JSON.parse(await readFile('static/data/tinystories-bpe/tokenizer.json', 'utf8')),
		config
	);
});
function forceToken(checkpoint: TokenStoryCheckpoint, token: number) {
	const controlled = structuredClone(checkpoint);
	controlled.parameters.forEach((leaf) => leaf.values.fill(0));
	controlled.parameters[0].values.fill(1);
	controlled.parameters[2].values[token] = 100;
	return controlled;
}
function deferred<T>() {
	let resolve!: (value: T) => void;
	const promise = new Promise<T>((done) => {
		resolve = done;
	});
	return { promise, resolve };
}

describe('live token generation numerical contracts', () => {
	it('measures the actual final input position and agrees with ordinary sampling without changing the checkpoint', async () => {
		const runtime = new LiveTokenStoryRuntime();
		await runtime.initialize(config, 42, 'wasm', corpus);
		const checkpoint = await runtime.exportCheckpoint();
		const parameters = deserializeTokenStoryTree(config, checkpoint.parameters);
		const forward = jit((p, tokens, positions, mask) =>
			tokenStoryForward(p, config, tokens, positions, mask, true)
		);
		try {
			const options = { maxTokens: 4, seed: 71, topK: 40 },
				frames: LiveTokenStoryFrame[] = [];
			const ordinary = await runtime.generate('Once upon a time', options);
			const live = await runtime.generateLive('Once upon a time', options, (frame) => {
				frames.push(frame);
			});
			expect(live).toEqual(ordinary);
			expect(frames).toHaveLength(live.tokenIds.length);
			expect(frames.length).toBeGreaterThan(1);
			for (const frame of frames) {
				expect(frame.modelId).toBe(checkpoint.modelId);
				expect(frame.step).toBe(checkpoint.step);
				expect(frame.trainedTokens).toBe(checkpoint.trainedTokens);
				expect(frame.tokenizerId).toBe(checkpoint.tokenizerId);
				expect(frame.corpusId).toBe(checkpoint.corpusId);
				expect(frame.sampledToken).toBe(live.tokenIds[frame.index]);
				expect(frame.position).toBe(frame.context.tokenIds.length - 1);
				const batch = tokenStoryBatch(config, [frame.context.tokenIds]);
				batch.targets.dispose();
				batch.lossMask.dispose();
				const result = forward(
					tree.ref(parameters),
					batch.tokens,
					batch.positions,
					tokenStoryMask(config)
				);
				const logp = await result.logprobs.data();
				const layers = await Promise.all(
					result.activations.map((array: { data(): Promise<Float32Array> }) => array.data())
				);
				const expected = Float32Array.from(
					logp.slice(frame.position * 4096, (frame.position + 1) * 4096),
					Math.exp
				);
				expect(
					Math.max(...frame.probabilities.map((p, i) => Math.abs(p - expected[i])))
				).toBeLessThan(1e-7);
				for (let unit = 0; unit < frame.unitCount; unit++)
					expect(frame.activations[unit]).toBe(
						layers[Math.floor(unit / config.hidden)][
							frame.position * config.hidden + (unit % config.hidden)
						]
					);
				expect(frame.activations.some((value) => value > 0)).toBe(true);
			}
			expect(await runtime.exportCheckpoint()).toEqual(checkpoint);
		} finally {
			forward.dispose();
			tree.dispose(parameters);
			runtime.dispose();
		}
	});
	it('retains noncanonical generated BPE IDs exactly after initial truncation and context sliding', async () => {
		const runtime = new LiveTokenStoryRuntime();
		await runtime.initialize(config, 42, 'wasm', corpus);
		try {
			const repeated = corpus.tokenizer.pieces.indexOf('l');
			await runtime.loadCheckpoint(forceToken(await runtime.exportCheckpoint(), repeated), 'wasm');
			const frames: LiveTokenStoryFrame[] = [];
			const result = await runtime.generateLive(
				'Once upon a time, there was a little fox in a forest.',
				{ maxTokens: 4, topK: 1 },
				(frame) => {
					frames.push(frame);
				}
			);
			expect(result.tokenIds).toEqual([repeated, repeated, repeated, repeated]);
			expect(result.prompt.truncatedTokens).toBeGreaterThan(0);
			expect(corpus.tokenizer.encode('ll')).not.toEqual([repeated, repeated]);
			frames.forEach((frame, index) => {
				const exact = [...result.prompt.tokenIds, ...result.tokenIds.slice(0, index)].slice(
					-config.context
				);
				expect(frame.index).toBe(index);
				expect(frame.context.tokenIds).toEqual(exact);
				expect(frame.context.pieces).toEqual(exact.map((id) => corpus.tokenizer.tokenPiece(id)));
				expect(frame.context.text).toBe(corpus.tokenizer.decode(exact));
				expect(frame.context.truncatedTokens).toBe(result.prompt.truncatedTokens + index);
				expect(frame.position).toBe(config.context - 1);
			});
			expect(frames[3].context.tokenIds).not.toEqual(
				corpus.tokenizer.encode(frames[3].context.text)
			);
		} finally {
			runtime.dispose();
		}
	});
	it('cancels a stalled callback immediately and returns only acknowledged tokens', async () => {
		const runtime = new LiveTokenStoryRuntime();
		await runtime.initialize(config, 42, 'wasm', corpus);
		try {
			await runtime.loadCheckpoint(forceToken(await runtime.exportCheckpoint(), 77), 'wasm');
			const before = await runtime.exportCheckpoint(),
				reached = deferred<void>(),
				release = deferred<void>();
			let frameCount = 0;
			const generating = runtime.generateLive('Once', { maxTokens: 4, topK: 1 }, async () => {
				frameCount++;
				if (frameCount === 2) {
					reached.resolve();
					await release.promise;
				}
			});
			await reached.promise;
			expect(frameCount).toBe(2);
			runtime.pause();
			const cancelled = await generating;
			expect(cancelled.tokenIds).toEqual([77]);
			expect(cancelled.cancelled).toBe(true);
			expect(cancelled.stoppedOnEos).toBe(false);
			expect(await runtime.exportCheckpoint()).toEqual(before);
			const next = await runtime.generateLive('Once', { maxTokens: 1, topK: 1 }, () => {});
			release.resolve();
			expect(next.tokenIds).toEqual([77]);
			expect(frameCount).toBe(2);
		} finally {
			runtime.dispose();
		}
	});
	it('does not commit an unacknowledged EOS, and can acknowledge EOS or propagate a callback error safely', async () => {
		const runtime = new LiveTokenStoryRuntime();
		await runtime.initialize(config, 42, 'wasm', corpus);
		try {
			await runtime.loadCheckpoint(forceToken(await runtime.exportCheckpoint(), 97), 'wasm');
			const before = await runtime.exportCheckpoint();
			const cancelled = await runtime.generateLive('', { maxTokens: 4, topK: 1 }, (frame) => {
				expect(frame.isEos).toBe(true);
				runtime.pause();
			});
			expect(cancelled.tokenIds).toEqual([]);
			expect(cancelled.cancelled).toBe(true);
			expect(cancelled.stoppedOnEos).toBe(false);
			const complete = await runtime.generateLive('', { maxTokens: 4, topK: 1 }, () => {});
			expect(complete.tokenIds).toEqual([97]);
			expect(complete.stoppedOnEos).toBe(true);
			expect(complete.cancelled).toBe(false);
			await expect(
				runtime.generateLive('', {}, () => {
					throw new Error('animation failed');
				})
			).rejects.toThrow('animation failed');
			expect(await runtime.exportCheckpoint()).toEqual(before);
		} finally {
			runtime.dispose();
		}
	});
	it('preserves frozen training math and resumes the exact optimizer trajectory after live sampling', async () => {
		const runtime = new LiveTokenStoryRuntime(),
			frozen = new TokenStoryRuntime();
		await runtime.initialize(config, 9, 'wasm', corpus);
		await frozen.initialize(config, 9, 'wasm', corpus);
		try {
			await runtime.train(1);
			await frozen.train(1);
			const before = await runtime.exportCheckpoint(),
				reference = await frozen.exportCheckpoint();
			expect(before.parameters).toEqual(reference.parameters);
			expect(before.optimizer).toEqual(reference.optimizer);
			await runtime.generateLive('Once', { maxTokens: 3 }, () => {});
			await runtime.train(1);
			await frozen.train(1);
			const after = await runtime.exportCheckpoint(),
				expected = await frozen.exportCheckpoint();
			expect(after.parameters).toEqual(expected.parameters);
			expect(after.optimizer).toEqual(expected.optimizer);
			expect(after.trainRngState).toBe(expected.trainRngState);
			expect(after.trainedTokens).toBe(expected.trainedTokens);
		} finally {
			runtime.dispose();
			frozen.dispose();
		}
	});
});
