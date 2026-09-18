import { expect, test } from '@playwright/test';
import { tools, disclose } from './workspace-test-helpers';

test('binding evidence follows the canvas and intervention setup is actionable', async ({
	page
}, testInfo) => {
	await page.setViewportSize({ width: 1280, height: 1200 });
	await page.goto('/');
	await tools(page, 'model');
	await disclose(page, 'Training settings');
	await expect(page.getByRole('button', { name: 'Start training', exact: true })).toBeEnabled({
		timeout: 90_000
	});
	const content = page.locator('.workspace-content');
	const evidence = page.locator('.evidence-drawer');
	for (const panel of ['model', 'inspector'] as const) {
		await tools(page, panel);
		const bounds = await content.boundingBox();
		const after = await evidence.boundingBox();
		expect(after!.y - bounds!.y - bounds!.height).toBeGreaterThanOrEqual(0);
		expect(after!.y - bounds!.y - bounds!.height).toBeLessThanOrEqual(20);
	}
	await page.getByRole('button', { name: 'Intervention', exact: true }).click();
	const setup = page.getByRole('region', { name: 'Intervention map setup' });
	await expect(setup).toContainText('silencing each of the 256 units');
	await setup.getByRole('button', { name: 'Measure 256 interventions', exact: true }).click();
	await expect(page.getByRole('button', { name: 'Intervention', exact: true })).toHaveAttribute(
		'aria-pressed',
		'true',
		{ timeout: 90_000 }
	);
	await expect(setup).toBeHidden();
	await page.evaluate(() => window.scrollTo(0, 0));
	await page.screenshot({ path: testInfo.outputPath('binding-evidence-flow.png'), fullPage: true });
	await page.getByRole('button', { name: 'Generate stories', exact: true }).click();
	await expect(
		page.getByRole('button', { name: 'Load & generate live', exact: true })
	).toBeEnabled();
});

test('one action loads trained weights and generates, with a clear path to intervention', async ({
	page
}, testInfo) => {
	const errors: string[] = [];
	page.on('pageerror', (error) => errors.push(error.message));
	await page.setViewportSize({ width: 1360, height: 1000 });
	await page.goto('/?view=tinystories');
	const lab = page.locator('.token-story-lab');
	const live = lab.getByRole('region', { name: 'Live token generation', exact: true });
	await disclose(live, 'Playback controls');
	await live.getByRole('button', { name: 'Slow · 400 ms', exact: true }).click();
	await lab.getByRole('button', { name: 'Load & generate live', exact: true }).click();
	await expect(live).toHaveAttribute('data-frame', '0', { timeout: 90_000 });
	await lab.getByRole('button', { name: 'Pause generation', exact: true }).click();
	await expect(live).toHaveAttribute('data-state', 'paused');
	await lab.getByRole('button', { name: 'Intervene', exact: true }).click();
	await lab.getByLabel('Subword unit ID', { exact: true }).fill('1024');
	await expect(lab.locator('.intervention-hint')).toContainText('Stop generation or replay');
	await live.getByRole('button', { name: 'Next token', exact: true }).click();
	await expect
		.poll(async () => Number(await live.getAttribute('data-emitted')))
		.toBeGreaterThanOrEqual(1);
	await live.getByRole('button', { name: 'Stop generation', exact: true }).click();
	const prepare = lab.getByRole('button', { name: 'Measure prompt for intervention', exact: true });
	await expect(prepare).toBeEnabled({ timeout: 30_000 });
	await prepare.click();
	const silence = lab.getByRole('button', { name: 'Silence selected unit', exact: true });
	await expect(silence).toBeEnabled({ timeout: 30_000 });
	await expect(silence).toBeInViewport();
	await silence.click();
	await expect(lab.getByText('Largest probability changes', { exact: true })).toBeVisible({
		timeout: 30_000
	});
	await page.screenshot({
		path: testInfo.outputPath('generated-and-intervened.png'),
		fullPage: true
	});
	expect(errors).toEqual([]);
});

test('workspace tools resize, collapse and preserve a measured replay across view changes', async ({
	page
}, testInfo) => {
	const errors: string[] = [];
	page.on('pageerror', (error) => errors.push(error.message));
	await page.setViewportSize({ width: 1440, height: 1000 });
	await page.goto('/?view=tinystories');
	const lab = page.locator('.token-story-lab');
	await tools(lab, 'model');
	await lab.locator('.archive-item.reference').filter({ hasText: 'Live replay' }).click();
	const live = lab.getByRole('region', { name: 'Live token generation', exact: true });
	await expect(live.locator('.trace-tokens button')).toHaveCount(24);
	await expect(lab.locator('.evidence-drawer')).not.toHaveAttribute('open');
	await tools(lab, 'inspector');
	await lab.getByLabel('Subword unit ID', { exact: true }).fill('1403');
	await disclose(live, 'Recorded tokens');
	await live.getByRole('button', { name: 'Inspect recorded token 11', exact: true }).click();
	await live.getByRole('button', { name: 'Inspect recorded layer 3', exact: true }).click();
	await expect(live.getByLabel('Selected channel activation')).toHaveText('1.8640');
	const resize = lab.getByRole('slider', { name: 'Resize tools panel' });
	await resize.press('ArrowLeft');
	await expect(resize).toHaveAttribute('aria-valuenow', '384');
	const box = await resize.boundingBox();
	expect(box).not.toBeNull();
	await page.mouse.move(box!.x + box!.width / 2, box!.y + 90);
	await page.mouse.down();
	await page.mouse.move(box!.x - 74, box!.y + 90);
	await page.mouse.up();
	const width = Number(await resize.getAttribute('aria-valuenow'));
	expect(width).toBeGreaterThan(450);
	expect(width).toBeLessThan(480);
	const canvas = lab.locator('.token-story-field canvas');
	const before = (await canvas.boundingBox())!.width;
	await lab.getByRole('button', { name: 'Collapse tools', exact: true }).click();
	await expect(lab.getByLabel('Subword unit ID')).toBeHidden();
	await expect.poll(async () => (await canvas.boundingBox())!.width).toBeGreaterThan(before + 350);
	await tools(lab, 'inspector');
	await expect(lab.getByLabel('Subword unit ID')).toHaveValue('1403');
	await expect(live).toHaveAttribute('data-frame', '10');
	await expect(resize).toHaveAttribute('aria-valuenow', String(width));
	await page.reload();
	await tools(lab, 'model');
	await expect(resize).toHaveAttribute('aria-valuenow', String(width));
	await tools(lab, 'model');
	await lab.locator('.archive-item.reference').filter({ hasText: 'Live replay' }).click();
	await tools(lab, 'inspector');
	await lab.getByLabel('Subword unit ID').fill('1403');
	await disclose(live, 'Recorded tokens');
	await live.getByRole('button', { name: 'Inspect recorded token 11', exact: true }).click();
	await live.getByRole('button', { name: 'Inspect recorded layer 3', exact: true }).click();
	await page.getByRole('link', { name: 'Tissue home' }).focus();
	await page.evaluate(() => window.scrollTo(0, 0));
	await page.screenshot({ path: testInfo.outputPath('workspace-desktop.png') });
	await page.getByRole('button', { name: 'Switch to light mode' }).click();
	await page.screenshot({ path: testInfo.outputPath('workspace-light.png') });
	await page.setViewportSize({ width: 390, height: 844 });
	await expect(resize).toBeHidden();
	await lab.getByRole('button', { name: 'Collapse tools', exact: true }).click();
	await expect(lab.getByRole('button', { name: 'Model', exact: true })).toBeVisible();
	await expect(lab.locator('.tools')).toBeHidden();
	expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
	await disclose(lab, 'Measurement details');
	await disclose(lab, 'Samples & evidence');
	expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
	await page.screenshot({ path: testInfo.outputPath('workspace-mobile.png'), fullPage: true });
	expect(errors).toEqual([]);
});

test('top actions keep replay, the full network and intervention controls in reach on a laptop', async ({
	page
}, testInfo) => {
	await page.setViewportSize({ width: 1280, height: 850 });
	await page.goto('/?view=tinystories');
	const lab = page.locator('.token-story-lab');
	const actions = lab.getByRole('navigation', { name: 'TinyStories actions' });
	for (const name of [
		'Load & generate live',
		'Replay example',
		'Train',
		'Probe',
		'Intervene',
		'Model'
	]) {
		await expect(actions.getByRole('button', { name, exact: true })).toBeInViewport();
	}
	await expect(lab.locator('.tools')).toBeHidden();
	await actions.getByRole('button', { name: 'Replay example', exact: true }).click();
	const live = lab.getByRole('region', { name: 'Live token generation' });
	await expect(live).toHaveAttribute('data-state', 'replaying', { timeout: 60_000 });
	await actions.getByRole('button', { name: 'Pause replay', exact: true }).click();
	const frame = await live.getAttribute('data-frame');
	const network = lab.locator('.token-story-field canvas');
	await expect(network).toBeInViewport({ ratio: 0.98 });
	await expect(live.getByRole('button', { name: 'Next layer', exact: true })).toBeInViewport();
	await actions.getByRole('button', { name: 'Intervene', exact: true }).click();
	await expect(
		lab.getByRole('button', { name: 'Silence selected unit', exact: true })
	).toBeInViewport();
	await expect(lab.locator('.intervention-hint')).toContainText('Stop generation or replay');
	await expect(live).toHaveAttribute('data-frame', frame!);
	await lab.getByRole('button', { name: 'Collapse tools' }).click();
	await page.mouse.move(10, 10);
	await page.screenshot({ path: testInfo.outputPath('top-actions-laptop.png') });
	await page.getByRole('button', { name: 'Switch to light mode' }).click();
	await page.screenshot({ path: testInfo.outputPath('top-actions-light.png') });
	await page.setViewportSize({ width: 390, height: 844 });
	expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
	await page.screenshot({ path: testInfo.outputPath('top-actions-mobile.png'), fullPage: true });
});
