/** Optional live-inference evidence around the frozen Study 004 binary codec.
 * Old archives remain valid. Live frames cover every committed generated token;
 * an unacknowledged provisional frame is not part of a completed/cancelled sample.
 */
import * as archive from './archive';
import type { TokenStoryRunRecord as HistoricalRunRecord, TokenStoryReference } from './archive';
import type { LiveTokenStoryFrame } from './live-protocol';
import { TOKEN_STORY_PRESETS, tokenStoryUnitCount } from './protocol';
import { createTokenStoryTokenizer } from './tokenizer';

export type { TokenStoryReference, TokenStoryRunSummary } from './archive';
export {
	getTokenStoryArchiveWarnings,
	listTokenStoryRuns,
	summarizeTokenStoryRun
} from './archive';

export interface TokenStoryLiveTrace {
	generationIndex: number;
	/** Exactly one final-input-position measurement per committed output token. */
	frames: LiveTokenStoryFrame[];
}

export interface TokenStoryLiveRunRecord extends HistoricalRunRecord {
	liveTraces?: TokenStoryLiveTrace[];
}

/** Drop-in name for callers moving from the historical archive module. */
export type TokenStoryRunRecord = TokenStoryLiveRunRecord;

const check = (condition: unknown, message: string): void => {
	if (!condition) throw new Error(`Invalid live token-story trace: ${message}`);
};
const integer = (value: unknown, maximum = Number.MAX_SAFE_INTEGER): value is number =>
	typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 && value <= maximum;
const finite = (value: unknown, minimum: number, maximum = Infinity): value is number =>
	typeof value === 'number' && Number.isFinite(value) && value >= minimum && value <= maximum;
const sameArray = (actual: unknown, expected: readonly unknown[]): boolean =>
	Array.isArray(actual) &&
	actual.length === expected.length &&
	actual.every((value, i) => value === expected[i]);

export function validateTokenStoryRun(value: unknown, deep = true): TokenStoryLiveRunRecord {
	const record = archive.validateTokenStoryRun(value, deep) as TokenStoryLiveRunRecord;
	if (record.liveTraces === undefined) return record;
	const samples = record.samples ?? [];
	check(
		Array.isArray(record.liveTraces) && record.liveTraces.length <= samples.length,
		'invalid trace history'
	);
	const tokenizer = createTokenStoryTokenizer(record.tokenizer);
	const config = TOKEN_STORY_PRESETS[record.presetId];
	const units = tokenStoryUnitCount(config);
	const tracedSamples = new Set<number>();
	for (const trace of record.liveTraces) {
		check(trace && integer(trace.generationIndex, samples.length - 1), 'invalid generation index');
		check(!tracedSamples.has(trace.generationIndex), 'duplicate generation trace');
		tracedSamples.add(trace.generationIndex);
		const generation = samples[trace.generationIndex];
		check(
			Array.isArray(trace.frames) &&
				trace.frames.length <= 256 &&
				trace.frames.length === generation.tokenIds.length,
			'trace must contain every committed generation token exactly once'
		);
		const metric = record.metrics.find((entry) => entry.step === generation.step);
		check(metric, 'traced checkpoint has no recorded metric');
		let context = [...generation.prompt.tokenIds];
		for (let index = 0; index < trace.frames.length; index++) {
			const frame = trace.frames[index];
			check(
				frame && frame.version === 1 && frame.index === index,
				'invalid frame version or index'
			);
			check(
				frame.modelId === generation.modelId &&
					frame.step === generation.step &&
					frame.seed === record.seed &&
					frame.tokenizerId === tokenizer.id &&
					frame.corpusId === record.corpusId &&
					integer(frame.trainedTokens) &&
					frame.trainedTokens === metric!.trainedTokens &&
					['webgpu', 'wasm', 'cpu'].includes(frame.backend),
				'frame checkpoint identity mismatch'
			);
			check(
				frame.config &&
					typeof frame.config === 'object' &&
					Object.entries(config).every(
						([key, expected]) => frame.config[key as keyof typeof config] === expected
					),
				'frame architecture mismatch'
			);
			check(
				frame.prompt === generation.prompt.original &&
					frame.context &&
					sameArray(frame.context.tokenIds, context) &&
					sameArray(
						frame.context.pieces,
						context.map((id) => tokenizer.tokenPiece(id))
					) &&
					frame.context.text === tokenizer.decode(context) &&
					frame.context.includesBos === (context[0] === tokenizer.bosId) &&
					frame.context.truncatedTokens ===
						generation.prompt.truncatedTokens +
							Math.max(0, generation.prompt.tokenIds.length + index - config.context),
				'frame input context or tokenization mismatch'
			);
			check(
				frame.position === context.length - 1 && frame.unitCount === units,
				'frame must measure the final real input position and every model channel'
			);
			check(
				frame.activations instanceof Float32Array &&
					frame.activations.length === units &&
					frame.activations.every((value) => finite(value, 0)),
				'invalid post-ReLU activation vector'
			);
			check(
				frame.probabilities instanceof Float32Array &&
					frame.probabilities.length === tokenizer.vocabularySize &&
					frame.probabilities.every((value) => finite(value, 0, 1)) &&
					Math.abs(frame.probabilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-3,
				'invalid raw next-token probability distribution'
			);
			const sampled = generation.tokenIds[index];
			check(
				frame.sampledToken === sampled &&
					frame.sampledPiece === tokenizer.tokenPiece(sampled) &&
					frame.isEos === (sampled === tokenizer.eosId),
				'frame sampled output mismatch'
			);
			context = [...context, sampled].slice(-config.context);
		}
	}
	return record;
}

export function encodeTokenStoryRun(record: TokenStoryLiveRunRecord): Blob {
	validateTokenStoryRun(record, false);
	return archive.encodeTokenStoryRun(record);
}

export function decodeTokenStoryRun(buffer: ArrayBuffer): TokenStoryLiveRunRecord {
	return validateTokenStoryRun(archive.decodeTokenStoryRun(buffer));
}

export async function importTokenStoryRun(file: File): Promise<TokenStoryLiveRunRecord> {
	return validateTokenStoryRun(await archive.importTokenStoryRun(file));
}

export async function loadTokenStoryRun(id: string): Promise<TokenStoryLiveRunRecord | null> {
	const record = await archive.loadTokenStoryRun(id);
	return record ? validateTokenStoryRun(record) : null;
}

export async function loadTokenStoryReference(
	reference: TokenStoryReference
): Promise<TokenStoryLiveRunRecord> {
	return validateTokenStoryRun(await archive.loadTokenStoryReference(reference));
}

export async function saveTokenStoryRun(record: TokenStoryLiveRunRecord): Promise<void> {
	validateTokenStoryRun(record, false);
	await archive.saveTokenStoryRun(record);
}

export function exportTokenStoryRun(record: TokenStoryLiveRunRecord): void {
	validateTokenStoryRun(record, false);
	archive.exportTokenStoryRun(record);
}
