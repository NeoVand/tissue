import { tools, disclose } from './workspace-test-helpers';
import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';

for (const context of [
	{ name: 'empty', text: '' },
	{ name: 'unsupported', text: 'Once upon a time 🧪' }
]) {
	test(`TinyStories rejects stale ablations after training with ${context.name} probe context`, async ({
		page
	}, testInfo) => {
		const errors: string[] = [];
		page.on('pageerror', (error) => errors.push(error.message));
		await page.goto('/');
		await tools(page, 'model');
		await disclose(page, 'Training settings');
		await expect(page.getByRole('button', { name: 'Start training', exact: true })).toBeEnabled({
			timeout: 90_000
		});
		await page.getByRole('button', { name: 'TinyStories', exact: true }).click();
		await page.getByRole('button', { name: 'Characters', exact: true }).click();
		const lab = page.locator('.story-lab');
		await lab.getByRole('button', { name: /TinyStories small.*Reference/ }).click();
		const resume = lab.getByRole('button', { name: 'Resume step 100', exact: true });
		await tools(lab, 'model');
		await disclose(lab, 'Training');
		await expect(resume).toBeEnabled({ timeout: 60_000 });
		await resume.click();
		const runPrompt = lab.getByRole('button', { name: 'Run prompt', exact: true });
		await tools(lab, 'inspector');
		await disclose(lab, 'Prompt probe');
		await expect(runPrompt).toBeEnabled({ timeout: 90_000 });
		await tools(lab, 'inspector');
		await lab.getByLabel('Story unit ID', { exact: true }).fill('2047');
		const lesion = lab.getByRole('button', { name: 'Silence selected unit', exact: true });
		await tools(lab, 'inspector');
		await expect(lesion).toBeEnabled();
		await tools(lab, 'inspector');
		await expect(lab.getByText('Activity measured at step 100', { exact: true })).toBeVisible();

		await tools(lab, 'inspector');
		await disclose(lab, 'Prompt probe');
		await lab.getByLabel('Probe context', { exact: true }).fill(context.text);
		await tools(lab, 'model');
		await disclose(lab, 'Training');
		await lab.locator('.transport').getByRole('button', { name: '25', exact: true }).click();
		await lab.getByRole('button', { name: 'Train', exact: true }).click();
		await tools(lab, 'model');
		await disclose(lab, 'Training');
		await expect(lab.getByRole('button', { name: 'Train', exact: true })).toBeEnabled({
			timeout: 90_000
		});
		await expect(lab.locator('.timeline')).toContainText('Durable weights 125');
		await tools(lab, 'inspector');
		await expect(lesion).toBeDisabled();
		await tools(lab, 'inspector');
		await expect(lab.getByText('Activity measured at step 100', { exact: true })).toHaveCount(0);
		await tools(lab, 'inspector');
		await expect(lab.getByText('Run a prompt to measure activity', { exact: true })).toBeVisible();
		if (context.name === 'unsupported')
			await expect(lab.getByRole('alert')).toContainText('Unsupported prompt characters');

		const validPrompt = 'Once upon a time, a little bird';
		await tools(lab, 'inspector');
		await disclose(lab, 'Prompt probe');
		await lab.getByLabel('Probe context', { exact: true }).fill(validPrompt);
		await runPrompt.click();
		await tools(lab, 'inspector');
		await expect(lesion).toBeEnabled({ timeout: 30_000 });
		await tools(lab, 'inspector');
		await expect(lab.getByText('Activity measured at step 125', { exact: true })).toBeVisible();
		await lesion.click();
		await expect(lab.getByText('Largest probability changes', { exact: true })).toBeVisible({
			timeout: 30_000
		});
		await tools(lab, 'model');
		await disclose(lab, 'Training');
		await expect(lab.getByRole('button', { name: 'Train', exact: true })).toBeEnabled();
		const receipt = page.waitForEvent('download');
		await lab.getByRole('button', { name: 'Export story run', exact: true }).click();
		const path = testInfo.outputPath(`stories-${context.name}-probe.tissue`);
		await (await receipt).saveAs(path);
		const bytes = await readFile(path);
		const magic = 'TISSUE-STORY-V1\n';
		expect(bytes.subarray(0, magic.length).toString()).toBe(magic);
		const length = bytes.readUInt32LE(magic.length);
		const record = JSON.parse(
			bytes.subarray(magic.length + 4, magic.length + 4 + length).toString()
		);
		const intervention = record.interventions.at(-1);
		expect(intervention.step).toBe(125);
		expect(intervention.modelId).toBe(record.checkpoint.modelId);
		expect(intervention.prompt).toBe(validPrompt);
		expect(intervention.neuron).toBe(2047);
		expect(intervention.probabilities.length).toBe(96);
		expect(intervention.lesionedProbabilities.length).toBe(96);
		expect(errors).toEqual([]);
	});
}
