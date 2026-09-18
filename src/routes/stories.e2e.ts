import { tools, disclose } from './workspace-test-helpers';
import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';

async function metadata(path: string) {
	const bytes = await readFile(path);
	const magic = 'TISSUE-STORY-V1\n';
	expect(bytes.subarray(0, magic.length).toString()).toBe(magic);
	const length = bytes.readUInt32LE(magic.length);
	return JSON.parse(bytes.subarray(magic.length + 4, magic.length + 4 + length).toString());
}

test('TinyStories reference resumes real weights, records interventions, trains, and survives import', async ({
	page
}, testInfo) => {
	const errors: string[] = [];
	page.on('pageerror', (error) => errors.push(error.message));
	await page.goto('/');
	await tools(page, 'model');
	await disclose(page, 'Training settings');
	await expect(page.getByRole('button', { name: 'Start training', exact: true })).toBeEnabled({
		timeout: 90000
	});
	await page.getByRole('button', { name: 'TinyStories', exact: true }).click();
	await page.getByRole('button', { name: 'Characters', exact: true }).click();
	const lab = page.locator('.story-lab');
	await lab.getByRole('button', { name: /TinyStories small.*Reference/ }).click();
	await tools(lab, 'model');
	await disclose(lab, 'Training');
	await expect(lab.getByRole('button', { name: 'Resume step 100', exact: true })).toBeEnabled({
		timeout: 60000
	});
	await tools(lab, 'model');
	await disclose(lab, 'Training');
	await expect(lab.getByRole('button', { name: 'Train', exact: true })).toBeDisabled();
	await tools(lab, 'inspector');
	await lab.getByLabel('Story unit ID', { exact: true }).fill('2047');
	await tools(lab, 'inspector');
	await expect(lab.getByText('layers[3] · channel 511', { exact: true })).toBeVisible();
	await tools(lab, 'model');
	await lab.getByRole('button', { name: 'Resume step 100', exact: true }).click();
	await tools(lab, 'inspector');
	await disclose(lab, 'Prompt probe');
	await expect(lab.getByRole('button', { name: 'Run prompt', exact: true })).toBeEnabled({
		timeout: 90000
	});
	await tools(lab, 'inspector');
	await lab.getByRole('button', { name: 'Silence selected unit', exact: true }).click();
	await expect(lab.getByText('Largest probability changes', { exact: true })).toBeVisible({
		timeout: 30000
	});
	await tools(lab, 'model');
	await disclose(lab, 'Training');
	await expect(lab.getByRole('button', { name: 'Train', exact: true })).toBeEnabled();
	await tools(lab, 'model');
	await disclose(lab, 'Training');
	await lab.locator('.transport').getByRole('button', { name: '25', exact: true }).click();
	await lab.getByRole('button', { name: 'Train', exact: true }).click();
	await page.getByRole('button', { name: 'Workbench', exact: true }).click();
	await tools(page, 'model');
	await disclose(page, 'Training settings');
	await expect(page.getByRole('button', { name: 'Start training', exact: true })).toBeDisabled();
	await page.getByRole('button', { name: 'TinyStories', exact: true }).click();
	await tools(lab, 'model');
	await disclose(lab, 'Training');
	await expect(lab.getByRole('button', { name: 'Train', exact: true })).toBeEnabled({
		timeout: 90000
	});
	await expect(lab.locator('.timeline')).toContainText('Durable weights 125');
	await disclose(lab, 'Measurement details');
	await lab.locator('.timeline').getByRole('button', { name: '0', exact: true }).click();
	await tools(lab, 'inspector');
	await disclose(lab, 'Prompt probe');
	await expect(lab.getByRole('button', { name: 'Run prompt', exact: true })).toBeDisabled();
	await lab.getByRole('button', { name: 'Latest', exact: true }).click();
	await tools(lab, 'inspector');
	await disclose(lab, 'Prompt probe');
	await expect(lab.getByRole('button', { name: 'Run prompt', exact: true })).toBeEnabled();
	await disclose(lab, 'Samples & evidence');
	await lab.locator('.sample-controls').getByRole('button', { name: '32', exact: true }).click();
	await lab.getByRole('button', { name: 'Sample', exact: true }).click();
	await expect(lab.getByText('Sample recorded at step 125', { exact: true })).toBeVisible({
		timeout: 30000
	});
	const receipt = page.waitForEvent('download');
	await lab.getByRole('button', { name: 'Export story run', exact: true }).click();
	const path = testInfo.outputPath('stories.tissue');
	await (await receipt).saveAs(path);
	const record = await metadata(path);
	expect(record.checkpoint.step).toBe(125);
	expect(record.metrics.at(-1).step).toBe(125);
	expect(record.interventions.at(-1).neuron).toBe(2047);
	expect(record.samples.at(-1).step).toBe(125);
	expect(record.samples.at(-1).tokenIds).toHaveLength(32);
	expect(record.interventions.at(-1).probabilities.length).toBe(96);
	await page.getByLabel('Import TinyStories specimen', { exact: true }).setInputFiles(path);
	await tools(lab, 'model');
	await disclose(lab, 'Training');
	await expect(lab.getByRole('button', { name: 'Resume step 125', exact: true })).toBeEnabled({
		timeout: 60000
	});
	await expect(lab.locator('.archive-item:not(.reference)')).toHaveCount(2);
	await page.reload();
	await page.getByRole('button', { name: 'TinyStories', exact: true }).click();
	await page.getByRole('button', { name: 'Characters', exact: true }).click();
	await expect(lab.locator('.archive-item:not(.reference)')).toHaveCount(2, { timeout: 30000 });
	expect(errors).toEqual([]);
});

test('large model reference exposes every unit and remains explicitly inspection-only on mobile', async ({
	page
}, testInfo) => {
	await page.goto('/');
	await tools(page, 'model');
	await disclose(page, 'Training settings');
	await expect(page.getByRole('button', { name: 'Start training', exact: true })).toBeEnabled({
		timeout: 90000
	});
	await page.getByRole('button', { name: 'TinyStories', exact: true }).click();
	await page.getByRole('button', { name: 'Characters', exact: true }).click();
	const lab = page.locator('.story-lab');
	await lab.getByRole('button', { name: /TinyStories large.*Reference/ }).click();
	await disclose(lab, 'Measurement details');
	await expect(
		lab.getByText('Saved measurements open · resume explicitly to train or probe', { exact: true })
	).toBeVisible({ timeout: 60000 });
	await tools(lab, 'inspector');
	await lab.getByLabel('Story unit ID', { exact: true }).fill('9215');
	await tools(lab, 'inspector');
	await expect(lab.getByText('layers[5] · channel 1535', { exact: true })).toBeVisible();
	await tools(lab, 'model');
	await disclose(lab, 'Training');
	await expect(lab.getByRole('button', { name: 'Train', exact: true })).toBeDisabled();
	await expect(lab.getByRole('button', { name: /^Resume step/ })).toHaveCount(0);
	await tools(lab, 'model');
	await disclose(lab, 'New model');
	await expect(lab.getByText(/This specimen contains measurements only/)).toBeVisible();
	await lab.getByRole('button', { name: 'Model layout', exact: true }).click();
	await expect(lab.locator('canvas')).toHaveAttribute('aria-label', /Layer \/ channel coordinates/);
	await lab.getByRole('button', { name: 'Functional', exact: true }).click();
	const receipt = page.waitForEvent('download');
	await lab.getByRole('button', { name: 'Export story run', exact: true }).click();
	const path = testInfo.outputPath('large-inspection.tissue');
	await (await receipt).saveAs(path);
	const record = await metadata(path);
	expect(record.checkpoint).toBeNull();
	expect(record.snapshots.at(-1).atlas.unitCount).toBe(9216);
	expect(record.snapshots.at(-1).atlas.fingerprints.length).toBe(9216 * 128);
	expect(record.snapshots).toHaveLength(4);
	await page.setViewportSize({ width: 390, height: 844 });
	expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});
