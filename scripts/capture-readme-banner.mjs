/** Capture the real lab and its published trained specimen; no synthetic UI or image editing. */
import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, resolve } from 'node:path';
import { chromium } from 'playwright';

const args = process.argv.slice(2);
const option = (name, fallback) => {
	const index = args.indexOf(name);
	if (index < 0) return fallback;
	assert(args[index + 1] && !args[index + 1].startsWith('--'), `${name} needs a value`);
	return args[index + 1];
};
const url = new URL(option('--url', 'http://127.0.0.1:5173/'));
url.searchParams.set('view', 'tinystories');
const output = resolve(option('--output', 'docs/assets/tissue-banner.png'));
const viewport = { width: 1600, height: 1120 };
await mkdir(dirname(output), { recursive: true });
const browser = await chromium.launch({ headless: true });
try {
	const page = await browser.newPage({ viewport, deviceScaleFactor: 2, colorScheme: 'dark' });
	const errors = [];
	page.on('pageerror', (error) => errors.push(error.message));
	await page.goto(url.href);
	const lab = page.locator('.token-story-lab');
	await lab
		.locator('.archive-item.reference')
		.filter({ hasText: 'Live replay · TinyStories BPE small at 4096' })
		.click();
	const live = lab.getByRole('region', { name: 'Live token generation', exact: true });
	await live
		.getByRole('button', { name: 'Replay trace', exact: true })
		.waitFor({ state: 'visible', timeout: 60_000 });
	await lab.getByLabel('Subword unit ID', { exact: true }).fill('1403');
	await live.getByRole('button', { name: 'Replay trace', exact: true }).click();
	await page.waitForFunction(
		() =>
			Number(
				document.querySelector('[aria-label="Live token generation"]')?.getAttribute('data-frame')
			) >= 10
	);
	await live.getByRole('button', { name: 'Pause replay', exact: true }).click();
	// Use the playback controls to focus a measured layer; never edit the canvas or data.
	while (Number(await live.getAttribute('data-layer')) < 2) {
		const before = await live.getAttribute('data-layer');
		await live.getByRole('button', { name: 'Next layer', exact: true }).click();
		await page.waitForFunction(
			(previous) =>
				document
					.querySelector('[aria-label="Live token generation"]')
					?.getAttribute('data-layer') !== previous,
			before
		);
	}
	await page.evaluate(() => document.fonts.ready);
	const canvas = lab.locator('.token-story-field canvas');
	await canvas.scrollIntoViewIfNeeded();
	const bounds = await canvas.boundingBox();
	assert(bounds);
	await page.mouse.move(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
	await page.mouse.wheel(0, -90);
	await page.waitForTimeout(500); // Let the existing camera damping settle.
	await page.mouse.move(1500, 80);
	await page.evaluate(
		() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)))
	);
	assert.equal(
		await lab.getByRole('button', { name: 'Functional', exact: true }).getAttribute('aria-pressed'),
		'true'
	);
	assert.equal(await live.getAttribute('data-state'), 'paused');
	assert.deepEqual(errors, []);
	await page.screenshot({ path: output, fullPage: false });
	const catalog = JSON.parse(await readFile('static/experiments/token-stories-index.json', 'utf8'));
	const references = Array.isArray(catalog) ? catalog : catalog.references;
	const reference = references.find(
		(entry) => entry.title === 'Live replay · TinyStories BPE small at 4096'
	);
	assert(reference, 'Published replay reference missing');
	const receipt = {
		capturedAt: new Date().toISOString(),
		image: 'tissue-banner.png',
		imageSha256: createHash('sha256')
			.update(await readFile(output))
			.digest('hex'),
		viewport,
		deviceScaleFactor: 2,
		pixelDimensions: { width: viewport.width * 2, height: viewport.height * 2 },
		source:
			'Actual browser screenshot, with the published measured replay paused using the lab controls. No UI injection or image compositing.',
		reference: {
			id: reference.id,
			file: reference.file,
			sha256: reference.sha256,
			step: reference.step,
			parameterCount: reference.parameterCount,
			unitCount: reference.unitCount
		},
		view: {
			layout: 'functional',
			encoding: 'brightness',
			displayedUnits: 2018,
			frameIndex: Number(await live.getAttribute('data-frame')),
			layerIndex: Number(await live.getAttribute('data-layer')),
			selectedUnit: 1403
		},
		url: url.href
	};
	await writeFile(output.replace(/\.png$/, '.json'), JSON.stringify(receipt, null, 2) + '\n');
	console.log(JSON.stringify(receipt, null, 2));
} finally {
	await browser.close();
}
