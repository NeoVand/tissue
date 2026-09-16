import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';

test('measured specimen survives inspection, export, and reload', async ({ page }, testInfo) => {
	const errors: string[] = [];
	page.on('pageerror', (error) => errors.push(error.message));
	await page.goto('/');
	await expect(page.getByRole('button', { name: 'Explore that specimen' })).toBeEnabled({
		timeout: 90000
	});
	await page.getByRole('button', { name: 'Explore that specimen' }).click();
	await expect(page.getByTestId('step')).toHaveText('2000', { timeout: 90000 });
	await expect(page.getByRole('button', { name: 'Continue training' })).toBeEnabled({
		timeout: 30000
	});
	expect(Number((await page.getByTestId('accuracy').innerText()).replace('%', ''))).toBeGreaterThan(
		90
	);
	await page.getByRole('button', { name: 'Intervention', exact: true }).click();
	await page.getByRole('combobox', { name: 'Select neuron' }).selectOption('172');
	await page.getByRole('button', { name: 'Silence selected unit' }).click();
	await expect(page.getByRole('button', { name: 'Restore intact view' })).toBeVisible();
	await page.getByRole('button', { name: 'Restore intact view' }).click();
	await page.getByRole('button', { name: 'Compare repair neighborhoods' }).click();
	await expect(page.getByText('Repair comparison recorded; source model preserved')).toBeVisible({
		timeout: 90000
	});
	await expect(page.getByTestId('step')).toHaveText('2000');
	await page.getByRole('button', { name: 'View checkpoint 500', exact: true }).click();
	await expect(page.getByTestId('step')).toHaveText('500');
	await expect(page.getByRole('button', { name: 'Silence selected unit' })).toBeDisabled();
	await page.getByRole('button', { name: 'Return to live' }).click();
	await expect(page.getByTestId('step')).toHaveText('2000');
	await page.getByRole('button', { name: /Field journal/ }).click();
	await page
		.getByLabel('Leave a field note')
		.fill('E2E: an observation retained with this specimen.');
	await page.getByRole('button', { name: 'Keep this observation' }).click();
	await expect(page.getByText('E2E: an observation retained with this specimen.')).toBeVisible();
	await page.getByRole('button', { name: 'Observatory', exact: true }).click();
	const downloadPromise = page.waitForEvent('download');
	await page.getByRole('button', { name: 'Export experiment' }).click();
	const download = await downloadPromise;
	const artifact = testInfo.outputPath('experiment.json');
	await download.saveAs(artifact);
	const record = JSON.parse(await readFile(artifact, 'utf8'));
	expect(record.checkpoint.step).toBe(2000);
	expect(record.atlas.effectFingerprints).toHaveLength(256);
	expect(record.repairs).toHaveLength(2);
	expect(record.observations.some((item: { kind: string }) => item.kind === 'intervention')).toBe(
		true
	);
	expect(
		record.observations.some((item: { detail: string }) => item.detail.startsWith('E2E:'))
	).toBe(true);
	await page.reload();
	await expect(page.getByTestId('step')).toHaveText('2000', { timeout: 90000 });
	await expect(page.getByRole('button', { name: 'Continue training' })).toBeEnabled({
		timeout: 30000
	});
	await page.getByRole('button', { name: /Field journal/ }).click();
	await expect(page.getByText('E2E: an observation retained with this specimen.')).toBeVisible();
	expect(errors).toEqual([]);
});

test('training, pause, checkpoint capture and mobile layout stay usable', async ({ page }) => {
	await page.setViewportSize({ width: 390, height: 844 });
	await page.goto('/');
	await expect(page.getByRole('button', { name: 'Start training' })).toBeEnabled({
		timeout: 90000
	});
	await page.getByRole('button', { name: 'Start training' }).click();
	await expect(page.getByRole('button', { name: 'Pause training' })).toBeVisible();
	await expect
		.poll(async () => Number(await page.getByTestId('step').innerText()), { timeout: 90000 })
		.toBeGreaterThan(0);
	await page.getByRole('button', { name: 'Pause training' }).click();
	await expect(page.getByRole('button', { name: 'Continue training' })).toBeEnabled({
		timeout: 90000
	});
	expect(await page.getByRole('button', { name: /View checkpoint/ }).count()).toBeGreaterThan(1);
	expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
	await page.getByRole('button', { name: 'Methods', exact: true }).click();
	await expect(page.getByRole('heading', { name: 'Every coordinate has a source.' })).toBeVisible();
});

test('a failed capture preserves evidence and requires checkpoint recovery', async ({
	page
}, testInfo) => {
	await page.addInitScript(() => {
		const original = Worker.prototype.postMessage;
		let trained = false;
		let injected = false;
		Worker.prototype.postMessage = function (message, options) {
			if (message?.op === 'train') trained = true;
			if (message?.op === 'atlas' && trained && !injected) {
				injected = true;
				queueMicrotask(() =>
					this.dispatchEvent(
						new MessageEvent('message', {
							data: { id: message.id, ok: false, error: 'Injected capture failure' }
						})
					)
				);
				return;
			}
			Reflect.apply(original, this, [message, options]);
		};
	});
	await page.goto('/');
	await expect(page.getByRole('button', { name: 'Start training' })).toBeEnabled({
		timeout: 90000
	});
	await page.getByRole('button', { name: 'Start training' }).click();
	await expect(page.getByRole('alert')).toContainText('Injected capture failure', {
		timeout: 90000
	});
	await expect(page.getByTestId('step')).toHaveText('50');
	await expect(page.getByRole('button', { name: 'Continue training' })).toBeDisabled();
	await expect(page.getByRole('button', { name: 'Measure all effects' })).toBeDisabled();
	await page.getByRole('combobox', { name: 'Select neuron' }).selectOption('172');
	await expect(page.getByRole('button', { name: 'Silence selected unit' })).toBeDisabled();
	const receipt = page.waitForEvent('download');
	await page.getByRole('button', { name: 'Export experiment' }).click();
	const failedPath = testInfo.outputPath('interrupted-evidence.json');
	await (await receipt).saveAs(failedPath);
	const interrupted = JSON.parse(await readFile(failedPath, 'utf8'));
	expect(interrupted.checkpoint.step).toBe(0);
	expect(interrupted.metrics.at(-1).step).toBe(50);
	await page.getByRole('button', { name: 'Restore saved checkpoint' }).click();
	await expect(page.getByRole('button', { name: 'Start training' })).toBeEnabled({
		timeout: 90000
	});
	await expect(page.getByTestId('step')).toHaveText('0');
	const recoveredReceipt = page.waitForEvent('download');
	await page.getByRole('button', { name: 'Export experiment' }).click();
	const recoveredPath = testInfo.outputPath('recovered-specimen.json');
	await (await recoveredReceipt).saveAs(recoveredPath);
	const recovered = JSON.parse(await readFile(recoveredPath, 'utf8'));
	expect(recovered.id).not.toBe(interrupted.id);
	expect(recovered.checkpoint.step).toBe(0);
	expect(recovered.metrics.at(-1).step).toBe(0);
	expect(recovered.observations.at(-1).detail).toContain(interrupted.id);
	await page.getByRole('button', { name: /Field journal/ }).click();
	await expect(page.getByRole('heading', { name: /recovered/ })).toBeVisible();
	await expect(page.getByText('Recovered from durable weights')).toBeVisible();
});
