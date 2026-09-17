import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
	decodeTokenStoryRun,
	encodeTokenStoryRun,
	importTokenStoryRun,
	validateTokenStoryRun,
	type TokenStoryLiveRunRecord
} from './live-archive';
import { encodeTokenStoryRun as encodeHistorical } from './archive';
import { TOKEN_STORY_PRESETS, tokenStoryUnitCount, type TokenStoryGeneration } from './protocol';
import { createTokenStoryTokenizer, type TokenStoryTokenizerData } from './tokenizer';
import { encodeTokenStoryPrompt } from './dataset';
import type { LiveTokenStoryFrame } from './live-protocol';

const tokenizerData = JSON.parse(
	readFileSync('static/data/tinystories-bpe/tokenizer.json', 'utf8')
) as TokenStoryTokenizerData;
const corpusId = JSON.parse(readFileSync('static/data/tinystories-bpe/corpus.json', 'utf8')).id;
const tokenizer = createTokenStoryTokenizer(tokenizerData);
const config = TOKEN_STORY_PRESETS.small;

function fixture(original = 'Once upon a time'): TokenStoryLiveRunRecord {
	// Deliberately noncanonical BPE continuation: these separate pieces decode to
	// "ab", which a tokenizer can merge. Inference must retain the generated IDs.
	const tokenIds = [66, 67, tokenizer.eosId];
	const sample: TokenStoryGeneration = {
		modelId: 'live-fixture',
		step: 0,
		tokenizerId: tokenizer.id,
		prompt: encodeTokenStoryPrompt(original, config.context, tokenizer),
		completion: tokenizer.decode(tokenIds),
		tokenIds,
		pieces: tokenIds.map((id) => tokenizer.tokenPiece(id)),
		samplingSeed: 71,
		temperature: 0.8,
		topK: 40,
		requestedTokens: 8,
		cancelled: false,
		stoppedOnEos: true
	};
	let context = [...sample.prompt.tokenIds];
	const frames: LiveTokenStoryFrame[] = tokenIds.map((sampledToken, index) => {
		const activations = new Float32Array(tokenStoryUnitCount(config));
		activations[0] = index + 0.125;
		activations[activations.length - 1] = Math.PI;
		const frame: LiveTokenStoryFrame = {
			version: 1,
			modelId: sample.modelId,
			step: 0,
			trainedTokens: 0,
			seed: 42,
			backend: 'wasm',
			config: { ...config },
			tokenizerId: tokenizer.id,
			corpusId,
			index,
			prompt: original,
			context: {
				tokenIds: [...context],
				pieces: context.map((id) => tokenizer.tokenPiece(id)),
				text: tokenizer.decode(context),
				truncatedTokens:
					sample.prompt.truncatedTokens +
					Math.max(0, sample.prompt.tokenIds.length + index - config.context),
				includesBos: context[0] === tokenizer.bosId
			},
			position: context.length - 1,
			unitCount: tokenStoryUnitCount(config),
			activations,
			probabilities: new Float32Array(config.vocabularySize).fill(1 / config.vocabularySize),
			sampledToken,
			sampledPiece: tokenizer.tokenPiece(sampledToken),
			isEos: sampledToken === tokenizer.eosId
		};
		context = [...context, sampledToken].slice(-config.context);
		return frame;
	});
	return {
		version: 1,
		kind: 'tissue-token-story-run',
		tokenizer: structuredClone(tokenizerData),
		corpusId,
		id: 'live-archive',
		title: 'Live generation trace fixture',
		seed: 42,
		presetId: 'small',
		createdAt: '2026-09-17T01:00:00Z',
		updatedAt: '2026-09-17T01:00:00Z',
		checkpoint: null,
		observations: [],
		snapshots: [],
		metrics: [
			{
				step: 0,
				trainLoss: null,
				validationLoss: 8.4,
				validationAccuracy: 0.001,
				unigramLoss: 6.5,
				unigramAccuracy: 0.04,
				uniformLoss: Math.log(4096),
				evaluationTokens: 2048,
				trainedTokens: 0,
				elapsedMs: 0,
				stepMs: 0,
				backend: 'wasm'
			}
		],
		samples: [sample],
		liveTraces: [{ generationIndex: 0, frames }]
	};
}

function frame(record: TokenStoryLiveRunRecord, index = 0): LiveTokenStoryFrame {
	return record.liveTraces![0].frames[index];
}

describe('optional live token-story evidence archives', () => {
	it('loads old archives without manufacturing live measurements', async () => {
		const record = fixture();
		delete record.liveTraces;
		const decoded = decodeTokenStoryRun(await encodeHistorical(record).arrayBuffer());
		expect(decoded.liveTraces).toBeUndefined();
		expect(decoded.samples).toEqual(record.samples);
	});

	it('round trips every exact float32 vector, input ID and sampled output through the frozen codec', async () => {
		const record = fixture();
		const result = await importTokenStoryRun(
			new File([encodeTokenStoryRun(record)], 'live.tissue')
		);
		expect(result.liveTraces).toEqual(record.liveTraces);
		for (let i = 0; i < 3; i++) {
			expect(frame(result, i).activations).toBeInstanceOf(Float32Array);
			expect(frame(result, i).probabilities).toBeInstanceOf(Float32Array);
			expect(frame(result, i).probabilities[tokenizer.bosId]).toBeGreaterThan(0);
		}
		expect(frame(result, 2).context.tokenIds.slice(-2)).toEqual([66, 67]);
		expect(frame(result, 2).isEos).toBe(true);
	});

	it('preserves initial truncation and exact sliding-window transitions without retokenizing generated text', () => {
		const record = fixture('\n'.repeat(config.context + 10));
		expect(record.samples![0].prompt.truncatedTokens).toBe(11);
		expect(validateTokenStoryRun(record)).toBe(record);
		expect(frame(record, 2).context.tokenIds.slice(-2)).toEqual([66, 67]);
		expect(frame(record, 2).context.truncatedTokens).toBe(13);
		expect(frame(record, 2).position).toBe(config.context - 1);
		frame(record, 1).context.truncatedTokens--;
		expect(() => validateTokenStoryRun(record)).toThrow(/input context/);
	});

	it('rejects missing committed frames, extra unacknowledged frames and duplicate generation records', () => {
		const record = fixture();
		const removed = record.liveTraces![0].frames.pop()!;
		expect(() => validateTokenStoryRun(record)).toThrow(/every committed/);
		record.liveTraces![0].frames.push(removed, structuredClone(removed));
		expect(() => validateTokenStoryRun(record)).toThrow(/every committed/);
		record.liveTraces![0].frames.pop();
		record.samples!.push(structuredClone(record.samples![0]));
		record.liveTraces!.push(structuredClone(record.liveTraces![0]));
		expect(() => validateTokenStoryRun(record)).toThrow(/duplicate/);
	});

	it('accepts complete evidence for cancelled partial generations, including cancellation before the first ACK', () => {
		const record = fixture();
		for (const count of [2, 0]) {
			const sample = record.samples![0];
			sample.tokenIds = sample.tokenIds.slice(0, count);
			sample.pieces = sample.tokenIds.map((id) => tokenizer.tokenPiece(id));
			sample.completion = tokenizer.decode(sample.tokenIds);
			sample.cancelled = true;
			sample.stoppedOnEos = false;
			record.liveTraces![0].frames = record.liveTraces![0].frames.slice(0, count);
			expect(validateTokenStoryRun(record).liveTraces![0].frames).toHaveLength(count);
		}
	});

	it('rejects checkpoint/architecture/index mismatches while allowing a different inference backend', () => {
		const changes: Array<[string, (value: LiveTokenStoryFrame) => void]> = [
			['checkpoint', (value) => (value.modelId = 'different')],
			['checkpoint', (value) => (value.trainedTokens = 1)],
			['checkpoint', (value) => (value.seed = 7)],
			['checkpoint', (value) => (value.tokenizerId = 'different')],
			['architecture', (value) => (value.config = { ...value.config, context: 256 })],
			['index', (value) => (value.index = 1)]
		];
		for (const [message, change] of changes) {
			const record = fixture();
			change(frame(record));
			expect(() => validateTokenStoryRun(record)).toThrow(new RegExp(message));
		}
		const record = fixture();
		frame(record).backend = 'webgpu';
		expect(validateTokenStoryRun(record)).toBe(record);
	});

	it('rejects token/piece/context changes, wrong input-row position and mismatched sampled output', () => {
		const changes: Array<[string, (value: LiveTokenStoryFrame) => void]> = [
			['input context', (value) => (value.context.tokenIds[0] = 1)],
			['input context', (value) => (value.context.pieces[0] = 'different')],
			['input context', (value) => (value.context.text += 'different')],
			['input context', (value) => (value.context.includesBos = false)],
			['final real input', (value) => value.position--],
			['sampled output', (value) => (value.sampledToken = 1)],
			['sampled output', (value) => (value.sampledPiece = 'different')],
			['sampled output', (value) => (value.isEos = true)]
		];
		for (const [message, change] of changes) {
			const record = fixture();
			change(frame(record));
			expect(() => validateTokenStoryRun(record)).toThrow(new RegExp(message));
		}
	});

	it('rejects wrong shapes, nonfinite/negative ReLU values and invalid raw distributions even on trusted saves', () => {
		const changes: Array<[string, (value: LiveTokenStoryFrame) => void]> = [
			['activation vector', (value) => (value.activations = new Float32Array(1))],
			['activation vector', (value) => (value.activations[0] = NaN)],
			['activation vector', (value) => (value.activations[0] = -1)],
			['probability distribution', (value) => (value.probabilities = new Float32Array(1))],
			['probability distribution', (value) => (value.probabilities[0] = Infinity)],
			['probability distribution', (value) => (value.probabilities[0] = -0.1)],
			['probability distribution', (value) => value.probabilities.fill(0)]
		];
		for (const [message, change] of changes) {
			const record = fixture();
			change(frame(record));
			expect(() => validateTokenStoryRun(record, false)).toThrow(new RegExp(message));
		}
	});
});
