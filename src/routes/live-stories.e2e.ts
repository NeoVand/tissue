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
	await expect(resume).toBeEnabled({ timeout: 60_000 });
	await resume.click();
	const live = lab.getByRole('region', { name: 'Live token generation', exact: true });
	const generate = lab.getByRole('button', { name: 'Generate live', exact: true });
	await expect(generate).toBeEnabled({ timeout: 90_000 });
	await lab.getByLabel('Subword unit ID', { exact: true }).fill('1024');
	await live.getByRole('button', { name: 'Slow · 400 ms', exact: true }).click();
	await generate.click();
	await expect(live).toHaveAttribute('data-frame', '0', { timeout: 60_000 });
	await live.getByRole('button', { name: 'Pause playback', exact: true }).click();
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
	await page.reload();
	await lab.locator('.archive-item:not(.reference)').first().click();
	await expect(lab.getByRole('button', { name: 'Resume step 4096', exact: true })).toBeEnabled({
		timeout: 60_000
	});
	await expect(live.getByRole('button', { name: 'Replay trace', exact: true })).toBeEnabled();
	await live.getByRole('button', { name: 'Replay trace', exact: true }).click();
	await expect(live).toHaveAttribute('data-frame', '0');
	await expect(live).toContainText('4096');
	await expect(lab.getByRole('button', { name: 'Resume step 4096', exact: true })).toBeVisible();
	await page.setViewportSize({ width: 390, height: 844 });
	expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
	await page.screenshot({ path: testInfo.outputPath('live-mobile.png'), fullPage: true });
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
	await lab.getByRole('button', { name: 'Initialize model', exact: true }).click();
	const train = lab.getByRole('button', { name: 'Train', exact: true });
	await expect(train).toBeEnabled({ timeout: 90_000 });
	await expect(
		lab.locator('.transport').getByRole('button', { name: 'Continuous', exact: true })
	).toHaveAttribute('aria-pressed', 'true');
	await train.click();
	const updates = async () => {
		const text = await lab.locator('.metric-strip').first().innerText();
		return Number(text.match(/Updates\s+([\d,]+)/)?.[1].replaceAll(',', '') ?? 0);
	};
	await expect.poll(updates, { timeout: 150_000 }).toBeGreaterThanOrEqual(125);
	await lab.locator('.transport').getByRole('button', { name: 'Pause', exact: true }).click();
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
