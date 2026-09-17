import { afterEach, describe, expect, it, vi } from 'vitest';
import { LayerPlaybackClock } from './layer-playback-clock';

afterEach(() => vi.useRealTimers());

describe('measured-layer presentation clock', () => {
	it('freezes a pending layer until exactly one manual advance', async () => {
		vi.useFakeTimers();
		const clock = new LayerPlaybackClock();
		clock.start();
		let advanced = false;
		const layer = clock.waitLayer().then((result) => (advanced = result));
		clock.pause();
		await vi.advanceTimersByTimeAsync(10_000);
		expect(advanced).toBe(false);
		clock.nextLayer();
		await layer;
		expect(advanced).toBe(true);
		let second = false;
		const pending = clock.waitLayer().then((result) => (second = result));
		await vi.advanceTimersByTimeAsync(10_000);
		expect(second).toBe(false);
		clock.stop();
		await pending;
	});

	it('finishes one token but gates the next measured frame while paused', async () => {
		vi.useFakeTimers();
		const clock = new LayerPlaybackClock();
		clock.start(true);
		clock.nextToken();
		expect(await clock.waitForFrame()).toBe(true);
		for (let i = 0; i < 4; i++) {
			const layer = clock.waitLayer();
			await vi.advanceTimersByTimeAsync(32);
			expect(await layer).toBe(true);
		}
		clock.finishToken();
		let opened = false;
		const nextFrame = clock.waitForFrame().then((result) => (opened = result));
		await vi.advanceTimersByTimeAsync(10_000);
		expect(opened).toBe(false);
		clock.nextLayer();
		await nextFrame;
		expect(opened).toBe(true);
		let skippedLayer = false;
		const nextLayer = clock.waitLayer().then((result) => (skippedLayer = result));
		await vi.advanceTimersByTimeAsync(10_000);
		expect(skippedLayer).toBe(false);
		clock.stop();
		await nextLayer;
	});

	it.each(['layer', 'frame'] as const)(
		'never revives a cancelled %s gate on immediate restart',
		async (kind) => {
			const clock = new LayerPlaybackClock();
			clock.start(true);
			const pending = kind === 'layer' ? clock.waitLayer() : clock.waitForFrame();
			clock.stop();
			clock.start();
			expect(await pending).toBe(false);
			clock.stop();
		}
	);
});
