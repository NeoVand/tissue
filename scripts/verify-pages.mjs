/** Exercise the real static artifact, including worker dataset fetches, without a dev-server fallback.
 * node scripts/verify-pages.mjs
 * node scripts/verify-pages.mjs --url https://neovand.github.io/tissue/
 */
import assert from 'node:assert/strict';
import { createReadStream } from 'node:fs';
import { access, stat } from 'node:fs/promises';
import { createServer } from 'node:http';
import { dirname, extname, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium, expect as playwrightExpect } from '@playwright/test';

const expect = playwrightExpect.configure({ timeout: 60_000 });
const args = process.argv.slice(2);
assert(
	args.length === 0 || (args.length === 2 && args[0] === '--url'),
	'Usage: verify-pages.mjs [--url https://host/tissue/]'
);
const buildRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../build');
const base = '/tissue';
const mime = {
	'.html': 'text/html; charset=utf-8',
	'.js': 'text/javascript; charset=utf-8',
	'.mjs': 'text/javascript; charset=utf-8',
	'.css': 'text/css; charset=utf-8',
	'.json': 'application/json',
	'.wasm': 'application/wasm',
	'.svg': 'image/svg+xml',
	'.png': 'image/png',
	'.jpg': 'image/jpeg',
	'.ico': 'image/x-icon',
	'.txt': 'text/plain; charset=utf-8',
	'.woff': 'font/woff',
	'.woff2': 'font/woff2'
};

async function serveBuild() {
	await access(resolve(buildRoot, 'index.html'));
	const server = createServer(async (request, response) => {
		try {
			const path = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
			if (!['GET', 'HEAD'].includes(request.method) || !path.startsWith(`${base}/`)) {
				response.writeHead(404).end('Not found');
				return;
			}
			const relative = path.slice(base.length + 1);
			const file = resolve(buildRoot, relative || 'index.html');
			if (!file.startsWith(`${buildRoot}${sep}`)) {
				response.writeHead(404).end('Not found');
				return;
			}
			const info = await stat(file);
			if (!info.isFile()) {
				response.writeHead(404).end('Not found');
				return;
			}
			response.writeHead(200, {
				'Content-Type': mime[extname(file)] ?? 'application/octet-stream',
				'Content-Length': info.size,
				'Cache-Control': 'no-store'
			});
			if (request.method === 'HEAD') response.end();
			else
				createReadStream(file)
					.on('error', () => response.destroy())
					.pipe(response);
		} catch {
			if (!response.headersSent) response.writeHead(404).end('Not found');
			else response.destroy();
		}
	});
	await new Promise((resolve, reject) => {
		server.once('error', reject);
		server.listen(0, '127.0.0.1', resolve);
	});
	return { server, url: new URL(`http://127.0.0.1:${server.address().port}${base}/`) };
}

const started = performance.now();
const observations = [];
const requestPaths = new Set();
const failures = [];
const externalFailures = [];
const pageErrors = [];
const unprefixed = [];
let ownedServer;
let browser;
let page;
let site;
const stage = async (name, action) => {
	const start = performance.now();
	console.log(`Checking ${name}…`);
	await action();
	observations.push({ name, elapsedMs: Math.round(performance.now() - start) });
};

try {
	if (args.length) {
		site = new URL(args[1]);
		assert(['http:', 'https:'].includes(site.protocol), '--url must use HTTP or HTTPS');
		assert(
			[base, `${base}/`].includes(site.pathname) && !site.search && !site.hash,
			'--url must identify /tissue/'
		);
		site.pathname = `${base}/`;
	} else {
		const local = await serveBuild();
		ownedServer = local.server;
		site = local.url;
		for (const path of ['/data/tinystories/corpus.json', `${base}/missing-asset.js`, '/'])
			assert.equal(
				(await fetch(new URL(path, site))).status,
				404,
				'Static server must not mask missing files with HTML'
			);
	}
	browser = await chromium.launch({ headless: true });
	const context = await browser.newContext({ viewport: { width: 1360, height: 1000 } });
	context.on('request', (request) => {
		const url = new URL(request.url());
		if (url.origin !== site.origin) return;
		requestPaths.add(url.pathname);
		if (/\/(data|experiments)\//.test(url.pathname) && !url.pathname.startsWith(`${base}/`))
			unprefixed.push(url.pathname);
	});
	context.on('response', (response) => {
		if (new URL(response.url()).origin === site.origin && response.status() >= 400)
			failures.push({ url: response.url(), status: response.status() });
	});
	context.on('requestfailed', (request) => {
		const item = { url: request.url(), error: request.failure()?.errorText };
		(new URL(request.url()).origin === site.origin ? failures : externalFailures).push(item);
	});
	page = await context.newPage();
	page.on('pageerror', (error) => pageErrors.push(error.message));
	await page.goto(new URL('?view=tinystories', site).href);
	const tokens = page.locator('.token-story-lab');
	await stage('recorded BPE live trace loads and replays', async () => {
		await tokens.locator('.archive-item.reference').filter({ hasText: 'Live replay' }).click();
		const live = tokens.getByRole('region', { name: 'Live token generation', exact: true });
		await expect(live.locator('.trace-tokens button')).toHaveCount(24);
		await expect(tokens.getByRole('button', { name: /^Resume step/ })).toHaveCount(0);
		await live.getByRole('button', { name: 'Slow · 400 ms', exact: true }).click();
		await live.getByRole('button', { name: 'Replay trace', exact: true }).click();
		await expect(live).toHaveAttribute('data-frame', '0');
		await live.getByRole('button', { name: 'Pause replay', exact: true }).click();
		await expect(live).toHaveAttribute('data-state', 'paused');
		await tokens.getByRole('button', { name: 'Inspect', exact: true }).click();
		await tokens.getByLabel('Subword unit ID', { exact: true }).fill('1024');
		assert(
			Number.isFinite(Number(await live.getByLabel('Selected channel activation').innerText()))
		);
		await live.getByRole('button', { name: 'Stop replay', exact: true }).click();
		await expect(live.getByRole('button', { name: 'Replay trace', exact: true })).toBeEnabled();
	});
	await stage('BPE checkpoint restores on WASM and probes new input', async () => {
		await tokens.getByRole('button', { name: 'Model & runs', exact: true }).click();
		await tokens
			.locator('.archive-item.reference')
			.filter({ hasText: 'TinyStories BPE small · seed 42' })
			.click();
		const resume = tokens.getByRole('button', { name: 'Resume step 4096', exact: true });
		await expect(resume).toBeEnabled();
		await tokens.locator('summary').filter({ hasText: 'New model' }).click();
		await tokens.getByRole('button', { name: 'WASM', exact: true }).click();
		await resume.click();
		await expect(tokens.getByText('Restored step 4096 · WASM', { exact: true })).toBeVisible({
			timeout: 180_000
		});
		await tokens.getByRole('button', { name: 'Inspect', exact: true }).click();
		await tokens.locator('.prompt-probe > summary').click();
		await tokens.getByLabel('Probe context', { exact: true }).fill('Lily put the ball in a box.');
		await tokens.getByRole('button', { name: 'Run prompt', exact: true }).click();
		await expect(tokens.getByText('Prompt measured at step 4096', { exact: true })).toBeVisible();
		await expect(tokens.locator('.token-grid button')).not.toHaveCount(0);
	});
	await stage('character checkpoint restores on WASM and probes new input', async () => {
		await page.getByRole('button', { name: 'Characters', exact: true }).click();
		const characters = page.locator('.story-lab');
		await characters.getByRole('button', { name: /TinyStories small.*Reference/ }).click();
		const resume = characters.getByRole('button', { name: 'Resume step 100', exact: true });
		await expect(resume).toBeEnabled();
		await characters.locator('summary').filter({ hasText: 'New model' }).click();
		await characters.getByRole('button', { name: 'WASM', exact: true }).click();
		await resume.click();
		await expect(characters.getByText('Restored step 100 · WASM', { exact: true })).toBeVisible({
			timeout: 180_000
		});
		await characters.getByRole('button', { name: 'Inspect', exact: true }).click();
		await characters.locator('.prompt-probe > summary').click();
		await characters.getByLabel('Probe context', { exact: true }).fill('Once upon a time');
		await characters.getByRole('button', { name: 'Run prompt', exact: true }).click();
		await expect(
			characters.getByText('Prompt measured at step 100', { exact: true })
		).toBeVisible();
		await expect(characters.locator('.token-grid button')).toHaveCount(16);
	});
	await stage('controlled binding and paired-query reference evidence loads', async () => {
		await page.getByRole('button', { name: 'Workbench', exact: true }).click();
		await page.getByRole('button', { name: 'Explore that specimen', exact: true }).click();
		await expect(page.getByTestId('step')).toHaveText('2000');
		await page.getByRole('button', { name: 'Query shifts', exact: true }).click();
		await page.getByRole('button', { name: 'Reference 42', exact: true }).click();
		await expect(
			page.getByRole('heading', { name: 'Does similarity transfer?', exact: true })
		).toBeVisible();
		await page.getByLabel('Query-study unit ID', { exact: true }).fill('172');
		await page.getByRole('button', { name: 'Held out', exact: true }).click();
		await expect(
			page.getByRole('img', { name: /Unit 172 intervention effects.*test assignment 1/ })
		).toBeVisible();
	});
	for (const asset of [
		'data/tinystories/tokens.bin',
		'data/tinystories/corpus.json',
		'data/tinystories-bpe/tokens.bin',
		'data/tinystories-bpe/corpus.json',
		'data/tinystories-bpe/tokenizer.json',
		'experiments/stories-index.json',
		'experiments/token-stories-index.json',
		'experiments/binding-seed-42.json',
		'experiments/query-shifts-seed-42.json'
	])
		assert(
			requestPaths.has(`${base}/${asset}`),
			`Required deployed asset was not requested: ${asset}`
		);
	assert.deepEqual(
		unprefixed,
		[],
		'Same-origin data/experiment requests escaped the deployment base'
	);
	assert.deepEqual(failures, [], 'Static asset or worker requests failed');
	assert.deepEqual(pageErrors, [], 'Browser raised uncaught errors');
	console.log(
		JSON.stringify(
			{
				passed: true,
				site: site.href,
				serving: ownedServer ? 'strict static build/' : 'published site',
				elapsedMs: Math.round(performance.now() - started),
				observations,
				dataAndExperimentPaths: [...requestPaths]
					.filter((path) => /\/(data|experiments)\//.test(path))
					.sort(),
				failedSameOriginRequests: failures,
				pageErrors,
				externalFailures
			},
			null,
			2
		)
	);
} catch (error) {
	console.error(error);
	console.error(
		JSON.stringify(
			{
				site: site?.href,
				observations,
				unprefixed,
				failures,
				pageErrors,
				externalFailures,
				visibleText: await page
					?.locator('body')
					.innerText()
					.catch(() => '')
					.then((text) => text.slice(-5000))
			},
			null,
			2
		)
	);
	process.exitCode = 1;
} finally {
	await browser?.close();
	if (ownedServer) await new Promise((resolve) => ownedServer.close(resolve));
}
