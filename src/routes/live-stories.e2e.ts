import { tools, disclose } from './workspace-test-helpers';
import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';

function header(bytes: Buffer) {
	const magic = 'TISSUE-TOKEN-STORY-V1\n';
	expect(bytes.subarray(0, magic.length).toString()).toBe(magic);
	const length = bytes.readUInt32LE(magic.length);
	return JSON.parse(bytes.subarray(magic.length + 4, magic.length + 4 + length).toString());
}

test('live generation pauses the worker, steps measured layers and tokens, and replays saved evidence', async ({
	page
}, testInfo) => {
	test.setTimeout(240_000);
	const errors: string[] = [];
	page.on('pageerror', (error) => errors.push(error.message));
	await page.setViewportSize({ width: 1360, height: 1000 });
	await page.goto('/?view=tinystories');
	const lab = page.locator('.token-story-lab');
	await lab
		.locator('.archive-item.reference')
		.filter({ hasText: 'TinyStories BPE small · seed 42' })
		.click();
	const resume = lab.getByRole('button', { name: 'Resume step 4096', exact: true });
	await tools(lab, 'model');
	await disclose(lab, 'Training');
	await expect(resume).toBeEnabled({ timeout: 60_000 });
	await resume.click();
	const live = lab.getByRole('region', { name: 'Live token generation', exact: true });
	const generate = lab.getByRole('button', { name: 'Generate live', exact: true });
	const primary = live.locator('.live-prefix > button');
	await expect(generate).toBeEnabled({ timeout: 90_000 });
	const primaryElement = await primary.elementHandle();
	const functional = lab.getByRole('button', { name: 'Functional', exact: true });
	await expect(functional).toHaveAttribute('aria-pressed', 'true');
	await expect(
		lab
			.getByRole('group', { name: 'Activation encoding' })
			.getByRole('button', { name: 'Brightness', exact: true })
	).toHaveAttribute('aria-pressed', 'true');
	await tools(lab, 'inspector');
	await lab.getByLabel('Subword unit ID', { exact: true }).fill('1024');
	await disclose(live, 'Playback controls');
	await live.getByRole('button', { name: 'Slow · 400 ms', exact: true }).click();
	await generate.click();
	await expect(live).toHaveAttribute('data-frame', '0', { timeout: 60_000 });
	await expect(functional).toHaveAttribute('aria-pressed', 'true');
	await expect(primary).toHaveText('Pause generation');
	expect(await primary.evaluate((element, original) => element === original, primaryElement)).toBe(
		true
	);
	await primary.click();
	await expect(primary).toHaveText('Resume generation');
	await expect(live).toHaveAttribute('data-state', 'paused');
	const before = {
		frame: await live.getAttribute('data-frame'),
		layer: Number(await live.getAttribute('data-layer')),
		emitted: Number(await live.getAttribute('data-emitted'))
	};
	await page.waitForTimeout(700);
	await expect(live).toHaveAttribute('data-frame', before.frame!);
	await expect(live).toHaveAttribute('data-layer', String(before.layer));
	await expect(live).toHaveAttribute('data-emitted', String(before.emitted));
	expect(before.layer).toBeLessThan(3);
	await live.getByRole('button', { name: 'Next layer', exact: true }).click();
	await expect(live).toHaveAttribute('data-layer', String(before.layer + 1));
	await expect(live).toHaveAttribute('data-emitted', String(before.emitted));
	await page.screenshot({ path: testInfo.outputPath('live-desktop.png'), fullPage: true });
	await live.getByRole('button', { name: 'Next token', exact: true }).click();
	await expect(live).toHaveAttribute('data-emitted', String(before.emitted + 1), {
		timeout: 30_000
	});
	await expect(live).toHaveAttribute('data-state', 'paused');
	await page.waitForTimeout(500);
	await expect(live).toHaveAttribute('data-emitted', String(before.emitted + 1));
	await live.getByRole('button', { name: 'Stop generation', exact: true }).click();
	await expect(generate).toBeEnabled({ timeout: 30_000 });
	const download = page.waitForEvent('download');
	await lab.getByRole('button', { name: 'Export subword run', exact: true }).click();
	const path = testInfo.outputPath('live-stopped.tissue');
	await (await download).saveAs(path);
	const record = header(await readFile(path));
	const trace = record.liveTraces.at(-1);
	const sample = record.samples[trace.generationIndex];
	expect(sample.cancelled).toBe(true);
	expect(sample.tokenIds).toHaveLength(before.emitted + 1);
	expect(trace.frames).toHaveLength(sample.tokenIds.length);
	expect(trace.frames[0].position).toBe(trace.frames[0].context.tokenIds.length - 1);
	expect(trace.frames[0].activations.length).toBe(2048);
	expect(trace.frames[0].probabilities.length).toBe(4096);
	expect(trace.frames[0].sampledToken).toBe(sample.tokenIds[0]);
	expect(record.checkpoint.step).toBe(4096);
	// A second start retains the other layout too; the same action resumes the paused worker.
	const modelLayout = lab.getByRole('button', { name: 'Model layout', exact: true });
	await modelLayout.click();
	await generate.click();
	await expect(live).toHaveAttribute('data-frame', '0', { timeout: 30_000 });
	await expect(modelLayout).toHaveAttribute('aria-pressed', 'true');
	await expect(primary).toHaveText('Pause generation');
	await primary.click();
	const pausedLayer = await live.getAttribute('data-layer');
	await expect(primary).toHaveText('Resume generation');
	await primary.click();
	await expect(primary).toHaveText('Pause generation');
	await expect(live).not.toHaveAttribute('data-layer', pausedLayer!);
	await primary.click();
	await expect(live).toHaveAttribute('data-state', 'paused');
	const resumedLayer = await live.getAttribute('data-layer');
	await page.waitForTimeout(700);
	await expect(live).toHaveAttribute('data-layer', resumedLayer!);
	await live.getByRole('button', { name: 'Next token', exact: true }).click();
	await expect
		.poll(async () => Number(await live.getAttribute('data-emitted')))
		.toBeGreaterThanOrEqual(1);
	await live.getByRole('button', { name: 'Stop generation', exact: true }).click();
	await expect(generate).toBeEnabled({ timeout: 30_000 });
	await page.reload();
	await lab.locator('.archive-item:not(.reference)').first().click();
	await tools(lab, 'model');
	await disclose(lab, 'Training');
	await expect(lab.getByRole('button', { name: 'Resume step 4096', exact: true })).toBeEnabled({
		timeout: 60_000
	});
	await expect(live.getByRole('button', { name: 'Replay trace', exact: true })).toBeEnabled();
	await live.getByRole('button', { name: 'Replay trace', exact: true }).click();
	await expect(live).toHaveAttribute('data-frame', '0');
	await expect(live).toContainText('4096');
	await tools(lab, 'model');
	await disclose(lab, 'Training');
	await expect(lab.getByRole('button', { name: 'Resume step 4096', exact: true })).toBeVisible();
	await page.setViewportSize({ width: 390, height: 844 });
	expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
	await page.screenshot({ path: testInfo.outputPath('live-mobile.png'), fullPage: true });
	expect(errors).toEqual([]);
});

test('recorded playback preserves visual controls and uses its primary pause action without a model', async ({
	page
}, testInfo) => {
	const errors: string[] = [];
	page.on('pageerror', (error) => errors.push(error.message));
	await page.setViewportSize({ width: 1360, height: 1000 });
	await page.goto('/?view=tinystories');
	const lab = page.locator('.token-story-lab');
	await lab
		.locator('.archive-item.reference')
		.filter({ hasText: 'Live replay · TinyStories BPE small at 4096' })
		.click();
	const live = lab.getByRole('region', { name: 'Live token generation', exact: true });
	const replay = live.getByRole('button', { name: 'Replay trace', exact: true });
	await expect(replay).toBeEnabled({ timeout: 60_000 });
	const primary = live.locator('.live-prefix > button');
	const original = await primary.elementHandle();
	await expect(primary).toHaveText('Load & generate live');
	await expect(primary).toBeEnabled();
	const encoding = lab.getByRole('group', { name: 'Activation encoding' });
	const brightness = encoding.getByRole('button', { name: 'Brightness', exact: true });
	const size = encoding.getByRole('button', { name: 'Size', exact: true });
	await expect(brightness).toHaveAttribute('aria-pressed', 'true');
	await size.click();
	await expect(size).toHaveAttribute('aria-pressed', 'true');
	await brightness.click();
	const functional = lab.getByRole('button', { name: 'Functional', exact: true });
	await disclose(live, 'Playback controls');
	await live.getByRole('button', { name: 'Slow · 400 ms', exact: true }).click();
	await replay.click();
	await expect(live).toHaveAttribute('data-frame', '0');
	await expect(primary).toHaveText('Pause replay');
	expect(await primary.evaluate((element, previous) => element === previous, original)).toBe(true);
	await primary.click();
	await expect(primary).toHaveText('Resume replay');
	await expect(functional).toHaveAttribute('aria-pressed', 'true');
	const frame = await live.getAttribute('data-frame');
	const layer = await live.getAttribute('data-layer');
	await page.waitForTimeout(700);
	await expect(live).toHaveAttribute('data-frame', frame!);
	await expect(live).toHaveAttribute('data-layer', layer!);
	await primary.click();
	await expect(live).not.toHaveAttribute('data-layer', layer!);
	await primary.click();
	await expect(primary).toHaveText('Resume replay');
	await live.getByRole('button', { name: 'Stop replay', exact: true }).click();
	await expect(primary).toHaveText('Load & generate live');
	await expect(primary).toBeEnabled();
	const modelLayout = lab.getByRole('button', { name: 'Model layout', exact: true });
	await modelLayout.click();
	const l4 = lab.locator('.layer-options').getByRole('button', { name: 'L4', exact: true });
	await l4.click();
	await replay.click();
	await expect(live).toHaveAttribute('data-frame', '0');
	await primary.click();
	await expect(modelLayout).toHaveAttribute('aria-pressed', 'true');
	await expect(l4).toHaveAttribute('aria-pressed', 'true');
	await expect(lab.locator('.layer-filter-note')).toContainText('outside the L4 filter');
	await lab.getByRole('button', { name: 'Show all layers', exact: true }).click();
	await expect(lab.locator('.layer-filter-note')).toHaveCount(0);
	await expect(brightness).toHaveAttribute('aria-pressed', 'true');
	const canvas = lab.locator('.token-story-field canvas');
	// Offscreen WebGL rendering is intentionally suspended. Inspect the actual viewport,
	// and let the visibility observer and scheduled draw settle before recording it.
	await canvas.scrollIntoViewIfNeeded();
	await expect(canvas).toBeInViewport({ ratio: 0.5 });
	await page.evaluate(
		() =>
			new Promise<void>((resolve) =>
				requestAnimationFrame(() => requestAnimationFrame(() => resolve()))
			)
	);
	await page.screenshot({
		path: testInfo.outputPath('brightness-replay-desktop.png')
	});
	await page.setViewportSize({ width: 390, height: 844 });
	expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
	await canvas.scrollIntoViewIfNeeded();
	await expect(canvas).toBeInViewport({ ratio: 0.5 });
	await page.evaluate(
		() =>
			new Promise<void>((resolve) =>
				requestAnimationFrame(() => requestAnimationFrame(() => resolve()))
			)
	);
	await page.screenshot({
		path: testInfo.outputPath('brightness-replay-mobile.png')
	});
	expect(errors).toEqual([]);
});

test('continuous training crosses the old default burst and saves its actual paused step', async ({
	page
}, testInfo) => {
	test.setTimeout(240_000);
	const errors: string[] = [];
	page.on('pageerror', (error) => errors.push(error.message));
	await page.goto('/?view=tinystories');
	const lab = page.locator('.token-story-lab');
	await tools(lab, 'model');
	await disclose(lab, 'New model');
	await lab.getByRole('button', { name: 'Initialize model', exact: true }).click();
	const train = lab.getByRole('button', { name: 'Train', exact: true });
	await tools(lab, 'model');
	await disclose(lab, 'Training');
	await expect(train).toBeEnabled({ timeout: 90_000 });
	await expect(
		lab.locator('.transport').getByRole('button', { name: 'Continuous', exact: true })
	).toHaveAttribute('aria-pressed', 'true');
	await tools(lab, 'model');
	await disclose(lab, 'Training');
	await train.click();
	await disclose(lab, 'Measurement details');
	const updates = async () => {
		const text = await lab.locator('.metric-strip').first().innerText();
		return Number(text.match(/Updates\s+([\d,]+)/)?.[1].replaceAll(',', '') ?? 0);
	};
	await expect.poll(updates, { timeout: 150_000 }).toBeGreaterThanOrEqual(125);
	await tools(lab, 'model');
	await disclose(lab, 'Training');
	await lab.locator('.transport').getByRole('button', { name: 'Pause', exact: true }).click();
	await tools(lab, 'model');
	await disclose(lab, 'Training');
	await expect(train).toBeEnabled({ timeout: 60_000 });
	const actualStep = await updates();
	expect(actualStep).toBeGreaterThanOrEqual(125);
	await expect(lab.locator('.timeline')).toContainText(`Durable weights ${actualStep}`);
	const download = page.waitForEvent('download');
	await lab.getByRole('button', { name: 'Export subword run', exact: true }).click();
	const path = testInfo.outputPath('continuous-paused.tissue');
	await (await download).saveAs(path);
	const record = header(await readFile(path));
	expect(record.checkpoint.step).toBe(actualStep);
	expect(record.metrics.at(-1).step).toBe(actualStep);
	expect(record.snapshots.at(-1).atlas.step).toBe(actualStep);
	expect(
		record.snapshots.some((snapshot: { atlas: { step: number } }) => snapshot.atlas.step === 100)
	).toBe(true);
	expect(errors).toEqual([]);
});
