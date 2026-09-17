import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';

test('query finalization stays locked after progress completes and preserves the finished raw study', async ({
	page
}, testInfo) => {
	await page.addInitScript(() => {
		const NativeWorker = window.Worker;
		const control = {
			held: false,
			progress: 0,
			modelMutations: [] as string[],
			release: () => {}
		};
		Object.defineProperty(window, '__queryFinalization', { value: control });
		window.Worker = class extends NativeWorker {
			constructor(url: string | URL, options?: WorkerOptions) {
				super(url, options);
				const queryRequests = new Set<number>();
				const post = this.postMessage;
				this.postMessage = (message, options) => {
					if (message?.op === 'query-shifts') queryRequests.add(message.id);
					if (control.held && ['initialize', 'load', 'train', 'repair'].includes(message?.op))
						control.modelMutations.push(message.op);
					Reflect.apply(post, this, [message, options]);
				};
				// Register before the application assigns onmessage. Only the final success is held;
				// all real progress events, including 512/512, continue to reach the application.
				this.addEventListener('message', (event: MessageEvent) => {
					const response = event.data;
					if (!queryRequests.has(response?.id)) return;
					if (response.event?.type === 'measurement') control.progress = response.event.completed;
					if (response.ok === true && !response.event) {
						event.stopImmediatePropagation();
						queryRequests.delete(response.id);
						control.held = true;
						control.release = () => {
							control.held = false;
							this.dispatchEvent(new MessageEvent('message', { data: response }));
						};
					}
				});
			}
		};
	});
	await page.goto('/');
	await expect(page.getByRole('button', { name: 'Start training' })).toBeEnabled({
		timeout: 90000
	});
	await page.getByRole('button', { name: 'Query shifts', exact: true }).click();
	await page.getByRole('button', { name: 'Measure current checkpoint', exact: true }).click();
	await expect
		.poll(
			() =>
				page.evaluate(() => {
					const state = (
						window as typeof window & { __queryFinalization: { held: boolean; progress: number } }
					).__queryFinalization;
					return { held: state.held, progress: state.progress };
				}),
			{ timeout: 90000 }
		)
		.toEqual({ held: true, progress: 512 });
	await page.getByRole('button', { name: 'Cancel measurement', exact: true }).click();
	await expect(page.getByRole('button', { name: 'Cancel measurement', exact: true })).toBeVisible();
	await expect(page.getByRole('button', { name: 'Reference 42', exact: true })).toBeDisabled();
	await page.getByRole('button', { name: 'Workbench', exact: true }).click();
	await expect(page.getByRole('button', { name: 'Start training' })).toBeDisabled();
	await expect(page.getByRole('button', { name: 'New run', exact: true })).toBeDisabled();
	await expect(
		page.getByRole('button', { name: 'Measure all effects', exact: true })
	).toBeDisabled();
	expect(
		await page.evaluate(
			() =>
				(
					window as typeof window & {
						__queryFinalization: { modelMutations: string[] };
					}
				).__queryFinalization.modelMutations
		)
	).toEqual([]);
	await page.getByRole('button', { name: 'Query shifts', exact: true }).click();
	await page.evaluate(() =>
		(
			window as typeof window & {
				__queryFinalization: { release: () => void };
			}
		).__queryFinalization.release()
	);
	await expect(page.getByText('Study ready · seed 42 · step 0', { exact: true })).toBeVisible({
		timeout: 60000
	});
	await expect(
		page.getByRole('button', { name: 'Measure current checkpoint', exact: true })
	).toBeEnabled();
	const download = page.waitForEvent('download');
	await page.getByRole('button', { name: 'Export query study', exact: true }).click();
	const path = testInfo.outputPath('finalized-query-study.json');
	await (await download).saveAs(path);
	const record = JSON.parse(await readFile(path, 'utf8'));
	expect(record.measurement.step).toBe(0);
	expect(record.measurement.checkpointPreserved).toBe(true);
	expect(record.measurement.splits.test.effects).toHaveLength(256);
	expect(
		await page.evaluate(async (id) => {
			const database = await new Promise<IDBDatabase>((resolve, reject) => {
				const request = indexedDB.open('tissue-query-studies', 1);
				request.onsuccess = () => resolve(request.result);
				request.onerror = () => reject(request.error);
			});
			try {
				return await new Promise<string | null>((resolve, reject) => {
					const request = database.transaction('studies').objectStore('studies').get(id);
					request.onsuccess = () => resolve(request.result?.measurement.checkpointHash ?? null);
					request.onerror = () => reject(request.error);
				});
			} finally {
				database.close();
			}
		}, record.id)
	).toBe(record.measurement.checkpointHash);
	await page.getByRole('button', { name: 'Workbench', exact: true }).click();
	await expect(page.getByTestId('step')).toHaveText('0');
	await expect(page.getByRole('button', { name: 'Start training' })).toBeEnabled();
});

test('measured specimen survives inspection, export, and reload', async ({ page }, testInfo) => {
	const errors: string[] = [];
	page.on('pageerror', (error) => errors.push(error.message));
	await page.goto('/');
	await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
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
	await page.getByRole('button', { name: 'Findings', exact: true }).click();
	await expect(page.getByRole('region', { name: 'Recorded fingerprint comparison' })).toContainText(
		'18.5%'
	);
	await page.getByRole('button', { name: 'Activation matrix', exact: true }).click();
	await page.getByRole('button', { name: 'Intervention', exact: true }).click();
	await page.getByLabel('Neuron ID', { exact: true }).fill('172');
	await expect(page.getByText('layers[1].mlpFc1', { exact: true })).toBeVisible();
	await expect(page.getByText('[:, 44]', { exact: true })).toBeVisible();
	await page.getByLabel('Inspect neuron activations.', { exact: false }).focus();
	await page.keyboard.press('ArrowRight');
	await expect(page.getByLabel('Neuron ID', { exact: true })).toHaveValue('173');
	await page.getByLabel('Neuron ID', { exact: true }).fill('172');
	await page.getByRole('button', { name: 'Model layout', exact: true }).click();
	await expect(page.locator('canvas').first()).toHaveAttribute('aria-label', /MODEL LAYOUT/);
	await page.getByRole('button', { name: 'Functional', exact: true }).click();
	await page.getByRole('button', { name: 'Switch to light mode' }).click();
	await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
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
	await page.getByRole('button', { name: 'Workbench', exact: true }).click();
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
	await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
	await expect(page.getByTestId('step')).toHaveText('2000', { timeout: 90000 });
	await expect(page.getByRole('button', { name: 'Continue training' })).toBeEnabled({
		timeout: 30000
	});
	await page.getByRole('button', { name: /Field journal/ }).click();
	await expect(page.getByText('E2E: an observation retained with this specimen.')).toBeVisible();
	await page.getByRole('button', { name: 'Workbench', exact: true }).click();
	await page.getByRole('button', { name: '100', exact: true }).click();
	await page.getByRole('button', { name: 'Continue training' }).click();
	await page.getByRole('button', { name: 'Intervention', exact: true }).click();
	await expect(page.getByTestId('step')).toHaveText('2100', { timeout: 90000 });
	await expect(page.getByRole('button', { name: 'Continue training' })).toBeEnabled({
		timeout: 30000
	});
	await expect(page.getByRole('button', { name: 'Activation', exact: true })).toHaveClass(/chosen/);
	await expect(page.locator('canvas').first()).toHaveAttribute('aria-label', /ATLAS STEP 2100/);
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
	await page.getByLabel('Neuron ID', { exact: true }).fill('172');
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

test('paired-query studies link raw probes, survive export and reload, and preserve the model', async ({
	page
}, testInfo) => {
	const errors: string[] = [];
	page.on('pageerror', (error) => errors.push(error.message));
	await page.goto('/');
	await expect(page.getByRole('button', { name: 'Start training' })).toBeEnabled({
		timeout: 90000
	});
	await page.getByRole('button', { name: 'Query shifts', exact: true }).click();
	await page.getByRole('button', { name: 'Reference 42', exact: true }).click();
	await expect(page.getByRole('heading', { name: 'Does similarity transfer?' })).toBeVisible({
		timeout: 60000
	});
	await page.getByLabel('Query-study unit ID', { exact: true }).fill('172');
	await page.getByRole('button', { name: 'Held out', exact: true }).click();
	await expect(
		page.getByRole('img', { name: /Unit 172 intervention effects.*test assignment 1/ })
	).toBeVisible();
	await page.getByRole('button', { name: 'Next assignment', exact: true }).click();
	await expect(
		page.getByRole('img', { name: /Unit 172 intervention effects.*test assignment 2/ })
	).toBeVisible();
	const receipt = page.waitForEvent('download');
	await page.getByRole('button', { name: 'Export query study', exact: true }).click();
	const path = testInfo.outputPath('query-study.json');
	await (await receipt).saveAs(path);
	const reference = JSON.parse(await readFile(path, 'utf8'));
	expect(reference.measurement.step).toBe(2000);
	expect(reference.measurement.splits.test.effects).toHaveLength(256);
	expect(reference.measurement.splits.test.effects[172]).toHaveLength(384);
	expect(reference.measurement.checkpointPreserved).toBe(true);
	await page.getByLabel('Import query study JSON', { exact: true }).setInputFiles(path);
	await expect(page.getByRole('button', { name: 'Export query study', exact: true })).toBeEnabled();
	await expect(page.locator('[aria-label="Saved query studies"] button')).toHaveCount(2);
	const importedReceipt = page.waitForEvent('download');
	await page.getByRole('button', { name: 'Export query study', exact: true }).click();
	const importedPath = testInfo.outputPath('query-imported.json');
	await (await importedReceipt).saveAs(importedPath);
	const imported = JSON.parse(await readFile(importedPath, 'utf8'));
	expect(imported.id).not.toBe(reference.id);
	expect(imported.measurement).toEqual(reference.measurement);
	expect(imported.provenance).toEqual(reference.provenance);
	await page.reload();
	await page.getByRole('button', { name: 'Query shifts', exact: true }).click();
	await expect(page.getByRole('heading', { name: 'Does similarity transfer?' })).toBeVisible({
		timeout: 60000
	});
	await expect(
		page.getByRole('button', { name: 'Measure current checkpoint', exact: true })
	).toBeEnabled({ timeout: 90000 });
	await page.getByRole('button', { name: 'Measure current checkpoint', exact: true }).click();
	await page.getByRole('button', { name: 'Cancel measurement', exact: true }).click();
	await expect(
		page.getByRole('button', { name: 'Measure current checkpoint', exact: true })
	).toBeEnabled({ timeout: 60000 });
	await expect(
		page.getByText('Measurement cancelled; no partial study was recorded.', { exact: true })
	).toBeVisible();
	await page.getByRole('button', { name: 'Measure current checkpoint', exact: true }).click();
	await expect(
		page.getByRole('button', { name: 'Measure current checkpoint', exact: true })
	).toBeEnabled({ timeout: 90000 });
	await expect(page.getByText('Study ready · seed 42 · step 0', { exact: true })).toBeVisible();
	const measuredReceipt = page.waitForEvent('download');
	await page.getByRole('button', { name: 'Export query study', exact: true }).click();
	const measuredPath = testInfo.outputPath('query-current.json');
	await (await measuredReceipt).saveAs(measuredPath);
	const measured = JSON.parse(await readFile(measuredPath, 'utf8'));
	expect(measured.measurement.step).toBe(0);
	expect(measured.measurement.splits.calibration.prefixMaxDifference).toBeLessThanOrEqual(1e-6);
	expect(measured.measurement.checkpointPreserved).toBe(true);
	await page.getByRole('button', { name: 'Workbench', exact: true }).click();
	await expect(page.getByTestId('step')).toHaveText('0');
	await page.getByRole('button', { name: /Field journal/ }).click();
	await expect(
		page.getByRole('heading', { name: 'Paired-query intervention study recorded', exact: true })
	).toBeVisible();
	await expect(
		page.getByRole('heading', { name: 'Paired-query studies', exact: true })
	).toBeVisible();
	expect(errors).toEqual([]);
});
