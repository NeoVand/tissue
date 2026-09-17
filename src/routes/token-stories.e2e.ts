import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';

test('subword lab trains, probes token positions, records exact ablation, samples, and restores binary evidence', async ({
	page
}, testInfo) => {
	const errors: string[] = [];
	const workers: string[] = [];
	page.on('pageerror', (error) => errors.push(error.message));
	page.on('worker', (worker) => workers.push(worker.url()));
	await page.goto('/');
	await expect(page.getByRole('button', { name: 'Start training', exact: true })).toBeEnabled({
		timeout: 90_000
	});
	const existingWorkerCount = workers.length;
	await page.getByRole('button', { name: 'TinyStories', exact: true }).click();
	const lab = page.locator('.token-story-lab');
	await expect(page.getByRole('button', { name: 'Subword', exact: true })).toHaveAttribute(
		'aria-pressed',
		'true'
	);
	await lab.locator('.prompt-probe > summary').click();
	await expect(lab.getByText('All text fits.', { exact: false }).first()).toBeVisible({
		timeout: 30_000
	});
	// Production worker URLs are hashed. Count allocations rather than matching source paths.
	expect(workers).toHaveLength(existingWorkerCount);
	await expect(lab.locator('.tokenization .pieces span')).not.toHaveCount(0);
	await lab.getByRole('button', { name: 'Initialize model', exact: true }).click();
	const train = lab.getByRole('button', { name: 'Train', exact: true });
	await expect(train).toBeEnabled({ timeout: 90_000 });
	expect(workers.length).toBeGreaterThan(existingWorkerCount);
	await expect(lab.locator('.metric-strip').first()).toContainText('nats / token');
	await lab.getByLabel('Subword unit ID', { exact: true }).fill('2047');
	await expect(lab.getByText('layers.3.mlpFc1[:, 511]', { exact: true })).toBeVisible();
	await expect(lab.getByText('layers.3.mlpFc2[511, :]', { exact: true })).toBeVisible();
	await lab.locator('.transport').getByRole('button', { name: '25', exact: true }).click();
	await train.click();
	await page.getByRole('button', { name: 'Characters', exact: true }).click();
	const characters = page.locator('.story-lab');
	await expect(
		characters.getByRole('button', { name: 'Initialize model', exact: true })
	).toBeDisabled();
	await expect(
		characters.getByText('Another lab worker is active.', { exact: false })
	).toBeVisible();
	await page.getByRole('button', { name: 'Subword', exact: true }).click();
	await expect(train).toBeEnabled({ timeout: 90_000 });
	await expect(lab.locator('.timeline')).toContainText('Durable weights 25');

	const prompt = 'Lily put the red ball in the box. Then she opened the box and saw';
	await lab.getByLabel('Probe context', { exact: true }).fill(prompt);
	await lab.getByRole('button', { name: 'Run prompt', exact: true }).click();
	const silence = lab.getByRole('button', { name: 'Silence selected unit', exact: true });
	await expect(silence).toBeEnabled({ timeout: 30_000 });
	await expect(lab.getByText('Activity measured at step 25', { exact: true })).toBeVisible();
	const originalPrompt = Array.from({ length: 12 }, () => prompt).join(' ');
	await lab.getByLabel('Probe context', { exact: true }).fill(originalPrompt);
	await expect(silence).toBeDisabled();
	await expect(lab.getByText('Activity measured at step 25', { exact: true })).toHaveCount(0);
	await expect(lab.getByText('This input is unmeasured.', { exact: false })).toBeVisible();
	await lab.getByRole('button', { name: 'Run prompt', exact: true }).click();
	await expect(silence).toBeEnabled({ timeout: 30_000 });
	const positions = lab.locator('.token-grid button');
	await expect(positions).toHaveCount(128);
	await positions.nth(3).click();
	await expect(lab.getByText('At token 4', { exact: true })).toBeVisible();
	await expect(lab.locator('.view-note')).toContainText('prompt token 4');
	await silence.click();
	await expect(lab.getByText('Largest probability changes', { exact: true })).toBeVisible({
		timeout: 30_000
	});
	await expect(train).toBeEnabled();
	await lab
		.locator('.sample-controls .segmented')
		.getByRole('button', { name: '32', exact: true })
		.click();
	await lab.getByRole('button', { name: 'Sample', exact: true }).click();
	await expect(train).toBeEnabled({ timeout: 60_000 });
	await expect(lab.locator('.generated-tokens summary')).toContainText(
		'Inspect generated token boundaries'
	);
	const download = page.waitForEvent('download');
	await lab.getByRole('button', { name: 'Export subword run', exact: true }).click();
	const path = testInfo.outputPath('subword.tissue');
	await (await download).saveAs(path);
	const bytes = await readFile(path);
	const magic = 'TISSUE-TOKEN-STORY-V1\n';
	expect(bytes.subarray(0, magic.length).toString()).toBe(magic);
	const length = bytes.readUInt32LE(magic.length);
	const record = JSON.parse(bytes.subarray(magic.length + 4, magic.length + 4 + length).toString());
	expect(record.kind).toBe('tissue-token-story-run');
	expect(record.tokenizer.vocabularySize).toBe(4096);
	expect(record.checkpoint.step).toBe(25);
	expect(record.checkpoint.tokenizerId).toBe(record.tokenizer.id);
	expect(record.snapshots.at(-1).atlas.unitCount).toBe(2048);
	expect(record.interventions.at(-1).probabilities.length).toBe(4096);
	expect(record.interventions.at(-1).lesionedProbabilities.length).toBe(4096);
	expect(record.interventions.at(-1).step).toBe(25);
	expect(record.interventions.at(-1).prompt).toBe(originalPrompt);
	expect(record.samples.at(-1).requestedTokens).toBe(32);
	expect(record.samples.at(-1).tokenIds.length).toBeLessThanOrEqual(32);
	expect(record.samples.at(-1).pieces).toHaveLength(record.samples.at(-1).tokenIds.length);
	await page.getByLabel('Import subword specimen', { exact: true }).setInputFiles(path);
	await expect(lab.getByRole('button', { name: 'Resume step 25', exact: true })).toBeEnabled({
		timeout: 60_000
	});
	await expect(train).toBeDisabled();
	await expect(lab.locator('.archive-item:not(.reference)')).toHaveCount(2);
	await lab.getByRole('button', { name: 'Resume step 25', exact: true }).click();
	await expect(train).toBeEnabled({ timeout: 90_000 });
	await expect(lab.getByRole('button', { name: 'Run prompt', exact: true })).toBeEnabled();
	await lab.getByLabel('Subword unit ID', { exact: true }).fill('2047');
	await expect(lab.getByText('Activity measured at step 25', { exact: true })).toBeVisible();
	await page.reload();
	await page.getByRole('button', { name: 'TinyStories', exact: true }).click();
	await expect(lab.locator('.archive-item:not(.reference)')).toHaveCount(2, { timeout: 30_000 });
	await lab.locator('.archive-item:not(.reference)').first().click();
	await expect(lab.getByRole('button', { name: 'Resume step 25', exact: true })).toBeEnabled({
		timeout: 30_000
	});
	await lab.getByLabel('Subword unit ID', { exact: true }).fill('2047');
	await expect(
		lab.getByRole('img', {
			name: 'Unit 2047: raw responses at all 128 calibration coordinates',
			exact: true
		})
	).toBeVisible();
	await page.setViewportSize({ width: 390, height: 844 });
	expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
	await page.screenshot({ path: testInfo.outputPath('subword-mobile.png'), fullPage: true });
	expect(errors).toEqual([]);
});
