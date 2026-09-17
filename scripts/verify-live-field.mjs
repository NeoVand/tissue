/** Actual WebGL framebuffer regression; no production test route or mocked renderer.
 * With pnpm dev running:
 * node scripts/verify-live-field.mjs [--url http://127.0.0.1:5173] [--output receipt.json]
 */
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
const args = process.argv.slice(2);
const option = (name, fallback) => {
	const index = args.indexOf(name);
	if (index < 0) return fallback;
	assert(args[index + 1] && !args[index + 1].startsWith('--'), `${name} requires a value`);
	return args[index + 1];
};
const baseUrl = new URL(option('--url', 'http://127.0.0.1:5173'));
assert(['http:', 'https:'].includes(baseUrl.protocol), '--url must use HTTP or HTTPS');
const output = resolve(option('--output', '.local-experiments/live-renderer-depth/receipt.json'));
await mkdir(dirname(output), { recursive: true });
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const files = ['src/lib/scene/neural-field.ts', 'src/lib/scene/field-shaders.ts'];
const sourceHashes = Object.fromEntries(
	await Promise.all(files.map(async (file) => [file, hash(await readFile(file))]))
);
const browser = await chromium.launch({ headless: true });
try {
	const page = await browser.newPage();
	await page.goto(new URL('/robots.txt', baseUrl).href);
	const result = await page.evaluate(async () => {
		const { createNeuralField } = await import('/src/lib/scene/neural-field.ts');
		document.body.innerHTML = '';
		const canvas = document.createElement('canvas');
		canvas.style.cssText = 'width:400px;height:400px;';
		document.body.append(canvas);
		const view = createNeuralField(canvas, {
			onselect() {},
			onhover() {},
			onerror(message) {
				throw new Error(message);
			}
		});
		try {
			// These locations lie along the initial camera direction. The earlier foreground
			// instance overlaps the selected layer's later instance exactly at the center pixel.
			const camera = view.inspect().camera,
				norm = Math.hypot(...camera);
			const direction = camera.map((coordinate) => coordinate / norm);
			const anchors = [
				{ id: 2, layer: 0, activation: 0, position: [-1, 0, 0] },
				{ id: 3, layer: 0, activation: 0, position: [1, 0, 0] }
			];
			const front = { id: 0, layer: 0, activation: 1, position: direction.map((x) => x * 0.1) };
			const back = { id: 1, layer: 1, activation: 1, position: direction.map((x) => -x * 0.1) };
			const both = [front, back, ...anchors],
				frontOnly = [front, ...anchors],
				backOnly = [back, ...anchors];
			const capture = async (points, activityMode, activeLayer) => {
				view.update({
					points,
					edges: [],
					selected: null,
					mode: 'activation',
					activityMode,
					activeLayer
				});
				// Ordinary snapshots retain their intentional transition; capture its settled state.
				if (!activityMode) await new Promise((resolve) => setTimeout(resolve, 550));
				view.rotate(false); // Request one render without modifying the camera or measurements.
				return await new Promise((resolve) =>
					requestAnimationFrame(() => {
						const gl = canvas.getContext('webgl2'),
							rgba = new Uint8Array(4);
						gl.readPixels(
							Math.floor(canvas.width / 2),
							Math.floor(canvas.height / 2),
							1,
							1,
							gl.RGBA,
							gl.UNSIGNED_BYTE,
							rgba
						);
						resolve(Array.from(rgba));
					})
				);
			};
			const normalBeforeBoth = await capture(both, false, null);
			const normalBeforeFront = await capture(frontOnly, false, null);
			const activityBoth = await capture(both, true, 1);
			const activityBack = await capture(backOnly, true, 1);
			const activityFront = await capture(frontOnly, true, 1);
			const normalRestoredBoth = await capture(both, false, null);
			const normalRestoredFront = await capture(frontOnly, false, null);
			return {
				browser: navigator.userAgent,
				canvas: { width: canvas.width, height: canvas.height },
				pixels: {
					normalBeforeBoth,
					normalBeforeFront,
					activityBoth,
					activityBack,
					activityFront,
					normalRestoredBoth,
					normalRestoredFront
				}
			};
		} finally {
			view.destroy();
		}
	});
	const p = result.pixels;
	assert.deepEqual(
		p.normalBeforeBoth,
		p.normalBeforeFront,
		'Normal snapshot foreground should retain ordinary opaque-depth occlusion'
	);
	assert.deepEqual(
		p.activityBoth,
		p.activityBack,
		'Faint context must not suppress the focused activity behind it'
	);
	assert.notDeepEqual(
		p.activityBoth,
		p.activityFront,
		'Activity result must contain the focused layer, not just faint context'
	);
	assert.equal(
		p.activityBoth[3],
		255,
		'Active center should be opaque at maximum measured activation'
	);
	assert.deepEqual(
		p.normalRestoredBoth,
		p.normalRestoredFront,
		'Turning activity off should restore depth occlusion'
	);
	assert.deepEqual(
		p.normalRestoredBoth,
		p.normalBeforeBoth,
		'Normal rendering should be unchanged after the round trip'
	);
	const finalHashes = Object.fromEntries(
		await Promise.all(files.map(async (file) => [file, hash(await readFile(file))]))
	);
	assert.deepEqual(finalHashes, sourceHashes, 'Renderer sources changed during verification');
	const receipt = {
		version: 1,
		kind: 'live-renderer-depth-regression',
		recordedAt: new Date().toISOString(),
		method:
			'Real headless Chromium WebGL2 center-pixel readback from overlapping instanced nodes. Earlier foreground layer is faint context; later background layer carries measured maximum activity. No numerical model or frozen study source modified.',
		...result,
		checksPassed: [
			'focused activity visible behind faint context',
			'ordinary snapshot depth occlusion preserved',
			'activity off restores original snapshot rendering'
		],
		sourceHashes,
		url: baseUrl.href,
		scriptPath: 'scripts/verify-live-field.mjs',
		scriptSha256: hash(await readFile(new URL(import.meta.url)))
	};
	await writeFile(output, JSON.stringify(receipt, null, 2) + '\n');
	console.log(JSON.stringify(receipt, null, 2));
} finally {
	await browser.close();
}
