import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { LiveTokenStoryEngine } from './live-engine';
import type { LiveTokenStoryFrame } from './live-protocol';

class FakeWorker {
	static current: FakeWorker;
	onmessage: ((event: MessageEvent) => void) | null = null;
	onerror = null;
	onmessageerror = null;
	sent: Array<{ id: number; op: string; index?: number; error?: string }> = [];
	terminated = false;
	constructor() {
		FakeWorker.current = this;
	}
	postMessage(message: (typeof this.sent)[number]) {
		this.sent.push(message);
	}
	terminate() {
		this.terminated = true;
	}
	receive(data: unknown) {
		this.onmessage?.({ data } as MessageEvent);
	}
}
function deferred() {
	let resolve!: () => void;
	const promise = new Promise<void>((done) => {
		resolve = done;
	});
	return { promise, resolve };
}
// Transport tests only require sequence identity. Numerical frame contents are checked in live-runtime.spec.ts.
const frame = (index: number) => ({ index }) as LiveTokenStoryFrame;
beforeEach(() => {
	vi.useFakeTimers();
	vi.stubGlobal('Worker', FakeWorker);
});
afterEach(() => {
	vi.unstubAllGlobals();
	vi.useRealTimers();
});

describe('live worker client backpressure', () => {
	it('waits indefinitely for user playback, then restarts the compute watchdog after ACK', async () => {
		const engine = new LiveTokenStoryEngine(),
			playback = deferred();
		const completed = engine
			.generateLive('Once', {}, () => playback.promise)
			.catch((error: Error) => error);
		const worker = FakeWorker.current,
			id = worker.sent[0].id;
		worker.receive({ id, frame: frame(0) });
		await vi.advanceTimersByTimeAsync(10 * 60 * 1000);
		expect(worker.terminated).toBe(false);
		expect(worker.sent.filter((message) => message.op === 'live-ack')).toEqual([]);
		playback.resolve();
		await vi.advanceTimersByTimeAsync(0);
		expect(worker.sent.at(-1)).toEqual({ id, op: 'live-ack', index: 0 });
		await vi.advanceTimersByTimeAsync(180000);
		expect(worker.terminated).toBe(true);
		expect(await completed).toBeInstanceOf(Error);
		expect(String(await completed)).toMatch(/timed out/);
	});
	it('cancels a stalled callback without waiting for it and ignores its late resolution', async () => {
		const engine = new LiveTokenStoryEngine(),
			playback = deferred();
		const generated = engine.generateLive('Once', {}, () => playback.promise);
		const worker = FakeWorker.current,
			id = worker.sent[0].id;
		worker.receive({ id, frame: frame(0) });
		await vi.advanceTimersByTimeAsync(0);
		const paused = engine.pause(),
			pause = worker.sent.at(-1)!;
		expect(pause.op).toBe('pause');
		worker.receive({ id: pause.id, ok: true, result: null });
		worker.receive({ id, ok: true, result: { tokenIds: [], cancelled: true } });
		await paused;
		expect(await generated).toMatchObject({ tokenIds: [], cancelled: true });
		playback.resolve();
		await vi.advanceTimersByTimeAsync(0);
		expect(worker.sent.some((message) => message.op === 'live-ack')).toBe(false);
		const disposed = engine.dispose();
		worker.receive({ id: worker.sent.at(-1)!.id, ok: true, result: null });
		await disposed;
		expect(worker.terminated).toBe(true);
	});
	it('rejects broken frame sequences and forwards callback failures to release the worker', async () => {
		const engine = new LiveTokenStoryEngine();
		const generated = engine
			.generateLive('Once', {}, () => {
				throw new Error('animation failed');
			})
			.catch((error: Error) => error);
		const worker = FakeWorker.current,
			id = worker.sent[0].id;
		worker.receive({ id, frame: frame(0) });
		await vi.advanceTimersByTimeAsync(0);
		expect(worker.sent.at(-1)).toEqual({
			id,
			op: 'live-error',
			index: 0,
			error: 'animation failed'
		});
		worker.receive({ id, ok: false, error: 'animation failed' });
		expect(String(await generated)).toMatch(/animation failed/);
		const malformed = engine.generateLive('Once', {}, () => {}).catch((error: Error) => error);
		const next = worker.sent.at(-1)!.id;
		worker.receive({ id: next, frame: frame(2) });
		expect(String(await malformed)).toMatch(/frame sequence/);
		expect(worker.terminated).toBe(true);
	});
	it('disposal terminates a paused worker promptly even if neither callback nor worker responds', async () => {
		const engine = new LiveTokenStoryEngine(),
			playback = deferred();
		const generated = engine
			.generateLive('Once', {}, () => playback.promise)
			.catch((error: Error) => error);
		const worker = FakeWorker.current;
		worker.receive({ id: worker.sent[0].id, frame: frame(0) });
		await vi.advanceTimersByTimeAsync(0);
		const disposed = engine.dispose();
		await vi.advanceTimersByTimeAsync(500);
		await disposed;
		expect(worker.terminated).toBe(true);
		expect(String(await generated)).toMatch(/disposed/);
		playback.resolve();
		await vi.advanceTimersByTimeAsync(0);
		expect(worker.sent.some((message) => message.op === 'live-ack')).toBe(false);
	});
});
