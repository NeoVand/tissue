import { describe, expect, it } from 'vitest';
import { encodeStoryPrompt } from './dataset';
import { decodeStoryRun, encodeStoryRun, validateStoryRun, type StoryRunRecord } from './archive';
import {
	STORY_PRESETS,
	STORY_CHARACTERS,
	storyParameterShapes,
	type StoryCorpusInfo
} from './protocol';

const corpus: StoryCorpusInfo = {
	id: 'tinystories-ascii-v1:67fb35357ec45562152b012c60d4a2d1aec5a95ad9d9a4fc6d534bc97fb4c6db',
	dataset: 'TinyStories',
	trainTokens: 1773785,
	calibrationTokens: 21719,
	evaluationTokens: 108433,
	trainStories: 2097,
	calibrationStories: 32,
	evaluationStories: 152,
	vocabulary: STORY_CHARACTERS,
	source: 'https://huggingface.co/datasets/roneneldan/TinyStories',
	license: 'CDLA-Sharing-1.0'
};

function fixture(): StoryRunRecord {
	const tensors = () =>
		storyParameterShapes(STORY_PRESETS.small).map((shape) => ({
			shape,
			values: new Float32Array(shape[0] * shape[1])
		}));
	return {
		version: 1,
		kind: 'tissue-story-run',
		id: 'round-trip',
		title: 'Archive contract',
		createdAt: '2026-09-17T01:00:00Z',
		updatedAt: '2026-09-17T01:00:00Z',
		seed: 42,
		presetId: 'small',
		metrics: [
			{
				step: 0,
				trainLoss: null,
				validationLoss: 4.6,
				validationAccuracy: 0.02,
				unigramLoss: 3,
				unigramAccuracy: 0.15,
				uniformLoss: Math.log(96),
				evaluationTokens: 2048,
				trainedTokens: 0,
				elapsedMs: 0,
				stepMs: 0,
				backend: 'wasm'
			}
		],
		snapshots: [],
		observations: [],
		checkpoint: {
			version: 1,
			architecture: 'stories-transformer-v1',
			modelId: 'fixture-model',
			config: STORY_PRESETS.small,
			seed: 42,
			step: 0,
			elapsedMs: 0,
			trainLoss: null,
			trainRngState: 42,
			corpus,
			parameters: tensors(),
			optimizer: { m: tensors(), v: tensors(), t: 0 }
		}
	};
}

describe('binary language-model archives', () => {
	it('preserves typed weights, optimizer state and exact float32 values in a binary round trip', async () => {
		const record = fixture();
		record.checkpoint!.parameters[0].values.set([Math.PI, -0.125, 1e-20]);
		record.checkpoint!.optimizer.m[0].values[0] = 0.375;
		record.checkpoint!.optimizer.v[0].values[0] = 0.03125;
		const blob = encodeStoryRun(record);
		expect(blob.size).toBeLessThan(10_100_000);
		const restored = decodeStoryRun(await blob.arrayBuffer());
		expect(restored.checkpoint!.parameters[0].values).toBeInstanceOf(Float32Array);
		for (const [expected, actual] of [
			[record.checkpoint!.parameters, restored.checkpoint!.parameters],
			[record.checkpoint!.optimizer.m, restored.checkpoint!.optimizer.m],
			[record.checkpoint!.optimizer.v, restored.checkpoint!.optimizer.v]
		]) {
			expect(actual.length).toBe(expected.length);
			for (let i = 0; i < expected.length; i++) {
				expect(actual[i].shape).toEqual(expected[i].shape);
				expect(actual[i].values.every((value, j) => Object.is(value, expected[i].values[j]))).toBe(
					true
				);
			}
		}
		expect(restored.checkpoint!.trainRngState).toBe(record.checkpoint!.trainRngState);
		expect(restored.checkpoint!.optimizer.t).toBe(record.checkpoint!.optimizer.t);
		expect(restored.metrics).toEqual(record.metrics);
	});
	it('rejects truncated payloads, extra bytes and a nonfinite checkpoint tensor', async () => {
		const record = fixture();
		const bytes = await encodeStoryRun(record).arrayBuffer();
		expect(() => decodeStoryRun(bytes.slice(0, -1))).toThrow('packed tensor');
		const extra = new Uint8Array(bytes.byteLength + 1);
		extra.set(new Uint8Array(bytes));
		expect(() => decodeStoryRun(extra.buffer)).toThrow('trailing bytes');
		const corrupt = bytes.slice(0);
		new DataView(corrupt).setFloat32(corrupt.byteLength - 4, Number.NaN, true);
		expect(() => decodeStoryRun(corrupt)).toThrow(/nonfinite|Invalid story/);
	});
	it('rejects mismatched architectures and negative second moments before worker allocation', () => {
		const record = fixture();
		record.checkpoint!.config = { ...STORY_PRESETS.medium };
		expect(() => validateStoryRun(record)).toThrow('architecture');
		record.checkpoint!.config = { ...STORY_PRESETS.small };
		record.checkpoint!.optimizer.v[0].values[0] = -1;
		expect(() => validateStoryRun(record)).toThrow('Invalid story v tensor');
	});
	it('preserves later observations when the last durable weights precede the observed step', () => {
		const record = fixture();
		record.metrics.push({ ...record.metrics[0], step: 25, trainedTokens: 12800 });
		record.observations.push({
			time: record.updatedAt,
			step: 25,
			title: 'Capture interrupted',
			detail: 'Weights remain at step zero.'
		});
		expect(validateStoryRun(record).checkpoint!.step).toBe(0);
		expect(validateStoryRun(record).metrics.at(-1)!.step).toBe(25);
		record.checkpoint!.step = 26;
		expect(() => validateStoryRun(record)).toThrow('checkpoint state');
	});
	it('accepts explicitly nonresumable evidence instead of manufacturing optimizer state', async () => {
		const record = fixture();
		record.checkpoint = null;
		const result = decodeStoryRun(await encodeStoryRun(record).arrayBuffer());
		expect(result.checkpoint).toBeNull();
	});
	it('keeps generated characters consistent with tokenizer, sampling settings, and cancellation', () => {
		const record = fixture();
		record.samples = [
			{
				modelId: 'fixture-model',
				step: 0,
				prompt: encodeStoryPrompt('Once upon a time', 128),
				completion: 'a',
				tokenIds: [66],
				samplingSeed: 71,
				temperature: 0.8,
				topK: 20,
				requestedTokens: 1,
				cancelled: false
			}
		];
		expect(validateStoryRun(record).samples![0].completion).toBe('a');
		record.samples[0].completion = 'b';
		expect(() => validateStoryRun(record)).toThrow('generated tokens');
		record.samples[0].completion = 'a';
		record.samples[0].prompt.text = 'Different';
		expect(() => validateStoryRun(record)).toThrow('tokenization');
	});
	it('round-trips paired intervention probabilities and rejects invalid probability mass', async () => {
		const record = fixture();
		record.interventions = [
			{
				step: 0,
				modelId: 'fixture-model',
				neuron: 2047,
				prompt: 'Once upon a time',
				capturedAt: record.updatedAt,
				probabilities: new Float32Array(96).fill(1 / 96),
				lesionedProbabilities: new Float32Array(96).fill(1 / 96)
			}
		];
		const decoded = decodeStoryRun(await encodeStoryRun(record).arrayBuffer());
		expect(decoded.interventions).toEqual(record.interventions);
		record.interventions[0].lesionedProbabilities[0] = 0.5;
		expect(() => validateStoryRun(record)).toThrow('intervention probabilities');
	});
});
