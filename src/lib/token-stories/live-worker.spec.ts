import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const flush = async () => {
	for (let i = 0; i < 20; i++) await Promise.resolve();
};
function deferred() {
	let resolve!: () => void;
	const promise = new Promise<void>((done) => {
		resolve = done;
	});
	return { promise, resolve };
}
interface WireMessage {
	id: number;
	op?: string;
	index?: number;
	frame?: { index: number };
	result?: { cancelled?: boolean; tokenIds?: number[] };
	[key: string]: unknown;
}
let scope: {
	onmessage: ((event: MessageEvent<WireMessage>) => void) | null;
	postMessage: ReturnType<typeof vi.fn>;
};
let initialization: ReturnType<typeof deferred>;
let generationCalls: boolean[];
const send = (data: WireMessage) => scope.onmessage!({ data } as MessageEvent<WireMessage>);
const received = (): WireMessage[] => scope.postMessage.mock.calls.map(([message]) => message);
beforeEach(async () => {
	vi.resetModules();
	initialization = deferred();
	generationCalls = [];
	scope = { onmessage: null, postMessage: vi.fn() };
	vi.stubGlobal('self', scope);
	vi.doMock('./live-runtime', () => ({
		LiveTokenStoryRuntime: class {
			async initialize() {
				await initialization.promise;
				return {};
			}
			pause() {}
			async generateLive(
				_text: string,
				_options: unknown,
				onFrame: (frame: unknown) => Promise<void>,
				cancelled: boolean
			) {
				generationCalls.push(cancelled);
				if (!cancelled) await onFrame({ index: 0 });
				return { tokenIds: cancelled ? [] : [77], cancelled };
			}
		}
	}));
	await import('./live-worker');
});
afterEach(() => {
	vi.doUnmock('./live-runtime');
	vi.unstubAllGlobals();
});

describe('live worker request routing', () => {
	it('retains cancellation for live generation queued behind initialization, without blocking later requests', async () => {
		send({ id: 1, op: 'initialize' });
		send({ id: 2, op: 'generate-live', prompt: 'Once', options: {} });
		send({ id: 3, op: 'pause' });
		await flush();
		expect(received().some((message) => message.id === 3 && message.ok)).toBe(true);
		expect(generationCalls).toEqual([]);
		initialization.resolve();
		await flush();
		expect(generationCalls).toEqual([true]);
		expect(received().find((message) => message.id === 2)?.result).toEqual({
			tokenIds: [],
			cancelled: true
		});
		expect(received().some((message) => message.frame)).toBe(false);
		send({ id: 4, op: 'generate-live', prompt: 'Once', options: {} });
		await flush();
		expect(generationCalls).toEqual([true, false]);
		expect(received().find((message) => message.id === 4)?.frame).toEqual({ index: 0 });
		send({ id: 4, op: 'live-ack', index: 0 });
		await flush();
		expect(received().find((message) => message.id === 4 && message.ok)?.result).toEqual({
			tokenIds: [77],
			cancelled: false
		});
	});
	it('ignores ACKs with the wrong request or frame index and unblocks callback failure', async () => {
		send({ id: 10, op: 'generate-live', prompt: 'Once', options: {} });
		await flush();
		expect(received()).toHaveLength(1);
		send({ id: 9, op: 'live-ack', index: 0 });
		send({ id: 10, op: 'live-ack', index: 1 });
		await flush();
		expect(received()).toHaveLength(1);
		send({ id: 10, op: 'live-error', index: 0, error: 'animation failed' });
		await flush();
		expect(received().at(-1)).toMatchObject({
			id: 10,
			ok: false,
			error: 'Error: animation failed'
		});
	});
});
