import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
	decodeTokenStoryRun,
	encodeTokenStoryRun,
	importTokenStoryRun,
	validateTokenStoryRun,
	type TokenStoryRunRecord
} from './archive';
import { TOKEN_STORY_PRESETS, tokenStoryParameterShapes, type TokenStoryAtlas } from './protocol';
import { createTokenStoryTokenizer, type TokenStoryTokenizerData } from './tokenizer';
import { encodeTokenStoryPrompt } from './dataset';
import { geometryAtlas } from './geometry';
import { buildStoryGeometry } from '../stories/geometry';

const tokenizerData = JSON.parse(
	readFileSync('static/data/tinystories-bpe/tokenizer.json', 'utf8')
) as TokenStoryTokenizerData;
const corpus = JSON.parse(readFileSync('static/data/tinystories-bpe/corpus.json', 'utf8'));
const tokenizer = createTokenStoryTokenizer(tokenizerData);

function fixture(): TokenStoryRunRecord {
	const tensors = () =>
		tokenStoryParameterShapes(TOKEN_STORY_PRESETS.small).map((shape) => ({
			shape,
			values: new Float32Array(shape[0] * shape[1])
		}));
	return {
		version: 1,
		kind: 'tissue-token-story-run',
		tokenizer: structuredClone(tokenizerData),
		corpusId: corpus.id,
		id: 'token-archive-fixture',
		title: 'BPE archive validation fixture',
		createdAt: '2026-09-17T01:00:00Z',
		updatedAt: '2026-09-17T01:00:00Z',
		seed: 42,
		presetId: 'small',
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
		snapshots: [],
		observations: [],
		checkpoint: {
			version: 1,
			architecture: 'token-stories-transformer-v1',
			modelId: 'fixture-model',
			config: { ...TOKEN_STORY_PRESETS.small },
			seed: 42,
			step: 0,
			trainedTokens: 0,
			elapsedMs: 0,
			trainLoss: null,
			trainRngState: 42,
			corpusId: corpus.id,
			tokenizerId: tokenizer.id,
			parameters: tensors(),
			optimizer: { m: tensors(), v: tensors(), t: 0 }
		}
	};
}

function addSample(record: TokenStoryRunRecord): void {
	const tokenIds = [...tokenizer.encode(' there was a little cat.'), tokenizer.eosId];
	record.samples = [
		{
			modelId: 'fixture-model',
			step: 0,
			tokenizerId: tokenizer.id,
			prompt: encodeTokenStoryPrompt(
				'Once upon a time',
				TOKEN_STORY_PRESETS.small.context,
				tokenizer
			),
			completion: tokenizer.decode(tokenIds),
			tokenIds,
			pieces: tokenIds.map((id) => tokenizer.tokenPiece(id)),
			samplingSeed: 71,
			temperature: 0.8,
			topK: 40,
			requestedTokens: 32,
			cancelled: false,
			stoppedOnEos: true
		}
	];
}

function addSnapshot(record: TokenStoryRunRecord): void {
	const config = TOKEN_STORY_PRESETS.small;
	const tokenIds = Array.from({ length: 8 }, () =>
		tokenizer.encode('A little girl smiled at her friend.', { bos: true })
	);
	const atlas: TokenStoryAtlas = {
		version: 1,
		modelId: 'fixture-model',
		config,
		seed: 42,
		step: 0,
		backend: 'wasm',
		capturedAt: record.createdAt,
		elapsedMs: 1,
		corpusId: corpus.id,
		tokenizerId: tokenizer.id,
		unitCount: config.layers * config.hidden,
		dimensions: 128,
		fingerprints: new Float32Array(config.layers * config.hidden * 128),
		positions: Array.from({ length: 16 }, (_, i) => Math.round((i * (config.context - 1)) / 15)),
		tokenPositions: tokenIds.map((ids) =>
			Array.from({ length: 16 }, (_, i) => Math.round((i * (ids.length - 1)) / 15))
		),
		tokenIds,
		examples: tokenIds.map((ids) => tokenizer.decode(ids))
	};
	record.snapshots = [
		{
			capturedAt: record.createdAt,
			atlas,
			geometry: buildStoryGeometry(geometryAtlas(atlas)).geometry
		}
	];
}

describe('BPE model evidence archives', () => {
	it('round trips exact typed weights, Adam state, RNG, tokenizer and readable generated pieces', async () => {
		const record = fixture();
		addSample(record);
		record.checkpoint!.parameters[0].values.set([Math.PI, -0.125, 1e-20]);
		record.checkpoint!.optimizer.m[0].values[0] = -0.375;
		record.checkpoint!.optimizer.v[0].values[0] = 0.03125;
		const restored = decodeTokenStoryRun(await encodeTokenStoryRun(record).arrayBuffer());
		for (const key of ['parameters', 'm', 'v'] as const) {
			const original =
				key === 'parameters' ? record.checkpoint!.parameters : record.checkpoint!.optimizer[key];
			const copied =
				key === 'parameters'
					? restored.checkpoint!.parameters
					: restored.checkpoint!.optimizer[key];
			expect(copied.length).toBe(original.length);
			for (let i = 0; i < copied.length; i++) {
				expect(copied[i].values).toBeInstanceOf(Float32Array);
				expect(copied[i].shape).toEqual(original[i].shape);
				expect(copied[i].values.every((value, j) => Object.is(value, original[i].values[j]))).toBe(
					true
				);
			}
		}
		expect(restored.tokenizer).toEqual(record.tokenizer);
		expect(restored.checkpoint!.trainRngState).toBe(42);
		expect(restored.samples).toEqual(record.samples);
	});

	it('rejects truncated bytes, trailing bytes, bad magic and unsupported record versions', async () => {
		const record = fixture();
		const bytes = await encodeTokenStoryRun(record).arrayBuffer();
		expect(() => decodeTokenStoryRun(bytes.slice(0, -1))).toThrow(/packed tensor/);
		const trailing = new Uint8Array(bytes.byteLength + 1);
		trailing.set(new Uint8Array(bytes));
		expect(() => decodeTokenStoryRun(trailing.buffer)).toThrow(/trailing/);
		const magic = bytes.slice(0);
		new Uint8Array(magic)[0] = 0;
		expect(() => decodeTokenStoryRun(magic)).toThrow(/archive/);
		Object.assign(record, { version: 2 });
		expect(() => validateTokenStoryRun(record)).toThrow(/version/);
	});

	it('rejects swapped tokenizer/corpus identities and cryptographically forged tokenizer definitions on import', async () => {
		const record = fixture();
		record.checkpoint!.tokenizerId = `tinystories-bpe-v1:${'0'.repeat(64)}`;
		expect(() => validateTokenStoryRun(record)).toThrow(/corpus or tokenizer/);
		record.checkpoint!.tokenizerId = tokenizer.id;
		record.checkpoint!.corpusId = `tinystories-bpe-corpus-v1:${'0'.repeat(64)}`;
		expect(() => validateTokenStoryRun(record)).toThrow(/corpus or tokenizer/);
		record.checkpoint!.corpusId = record.corpusId;
		record.tokenizer.trainingTextSha256 = '0'.repeat(64);
		const file = new File([encodeTokenStoryRun(record)], 'forged-tokenizer.tissue');
		await expect(importTokenStoryRun(file)).rejects.toThrow(/identity/);
	});

	it('rejects different architectures, nonfinite parameters/moments and negative second moments', () => {
		const record = fixture();
		record.checkpoint!.config = { ...TOKEN_STORY_PRESETS.medium };
		expect(() => validateTokenStoryRun(record)).toThrow(/architecture/);
		record.checkpoint!.config = { ...TOKEN_STORY_PRESETS.small };
		for (const row of [
			record.checkpoint!.parameters[0],
			record.checkpoint!.optimizer.m[0],
			record.checkpoint!.optimizer.v[0]
		]) {
			row.values[0] = Infinity;
			expect(() => validateTokenStoryRun(record)).toThrow(/nonfinite/);
			row.values[0] = 0;
		}
		record.checkpoint!.optimizer.v[0].values[0] = -1;
		expect(() => validateTokenStoryRun(record)).toThrow(/v tensor/);
	});

	it('rejects prompt token or piece changes and completion text that disagrees with generated ids', () => {
		const record = fixture();
		addSample(record);
		const saved = structuredClone(record.samples![0]);
		record.samples![0].prompt.tokenIds[1] = 1;
		expect(() => validateTokenStoryRun(record)).toThrow(/tokenization/);
		record.samples![0] = structuredClone(saved);
		record.samples![0].prompt.pieces[1] = 'invented';
		expect(() => validateTokenStoryRun(record)).toThrow(/tokenization/);
		record.samples![0] = structuredClone(saved);
		record.samples![0].completion += ' not in tokens';
		expect(() => validateTokenStoryRun(record)).toThrow(/generated tokens/);
	});

	it('allows genuine EOS termination and rejects generated BOS or tokens after EOS', () => {
		const record = fixture();
		addSample(record);
		expect(validateTokenStoryRun(record).samples![0].stoppedOnEos).toBe(true);
		for (const ids of [
			[tokenizer.bosId, tokenizer.eosId],
			[tokenizer.eosId, 66, tokenizer.eosId]
		]) {
			record.samples![0].tokenIds = ids;
			record.samples![0].pieces = ids.map((id) => tokenizer.tokenPiece(id));
			record.samples![0].completion = tokenizer.decode(ids);
			expect(() => validateTokenStoryRun(record)).toThrow(/generated tokens/);
		}
	});

	it('rejects altered calibration token positions and decoded example text before fitting a map', () => {
		const record = fixture();
		record.checkpoint = null;
		addSnapshot(record);
		expect(validateTokenStoryRun(record).snapshots[0].atlas.tokenPositions).toHaveLength(8);
		const atlas = record.snapshots[0].atlas;
		atlas.tokenPositions[0][1] += 1;
		expect(() => validateTokenStoryRun(record)).toThrow(/calibration/);
		atlas.tokenPositions[0][1] -= 1;
		atlas.examples[0] += 'invented';
		expect(() => validateTokenStoryRun(record)).toThrow(/calibration/);
	});

	it('retains later observations with earlier durable weights but rejects impossible token counters', () => {
		const record = fixture();
		record.metrics.push({ ...record.metrics[0], step: 2, trainedTokens: 600 });
		expect(validateTokenStoryRun(record).checkpoint!.step).toBe(0);
		record.checkpoint!.trainedTokens = 1;
		expect(() => validateTokenStoryRun(record)).toThrow(/token/);
		record.checkpoint!.trainedTokens = 0;
		record.metrics.push({ ...record.metrics[1], step: 3, trainedTokens: 500 });
		expect(() => validateTokenStoryRun(record)).toThrow(/token/);
	});
});
