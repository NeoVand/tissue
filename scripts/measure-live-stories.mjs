/** Capture a replayable live-inference trace from the existing trained specimen.
 * The original training record, weights, samples and source files are never edited.
 * With pnpm dev running:
 * node scripts/measure-live-stories.mjs [--backend webgpu] [--output directory] [--publish]
 * Public evidence omits duplicate weights; the complete resumable archive stays local.
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { chromium } from 'playwright';

const args = process.argv.slice(2);
const option = (name, fallback) => {
	const index = args.indexOf(name);
	if (index < 0) return fallback;
	assert(args[index + 1] && !args[index + 1].startsWith('--'), `${name} requires a value`);
	return args[index + 1];
};
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const readJSON = async (path) => JSON.parse(await readFile(path, 'utf8'));
const output = resolve(option('--output', `.local-experiments/live-story-demo-${Date.now()}`));
const baseUrl = option('--base-url', 'http://127.0.0.1:5173');
const backend = option('--backend', 'webgpu');
assert(['webgpu', 'wasm', 'auto'].includes(backend), 'Unsupported backend');
const sourceId = option('--source-id', 'cdded5d8-27cf-49af-9514-64eee6daf7c4');
const catalogPath = 'static/experiments/token-stories-index.json';
const catalog = await readJSON(catalogPath);
assert.equal(catalog.version, 1);
const reference = catalog.references.find((entry) => entry.id === sourceId);
assert(reference?.reportFile, 'The source must be an existing recorded specimen with its report');
const assetPath = (path) => {
	assert(
		/^\/experiments\/token-stories-[a-zA-Z0-9.-]+$/.test(path),
		'Unexpected source asset path'
	);
	return `static${path}`;
};
const sourceReportPath = assetPath(reference.reportFile);
const sourceReportBytes = await readFile(sourceReportPath);
const sourceReport = JSON.parse(sourceReportBytes);
assert.deepEqual(sourceReport.published, reference, 'Source catalog and report disagree');
const originalFiles = Object.fromEntries(
	await Promise.all(
		(reference.parts ?? [reference.file]).map(async (path) => {
			const bytes = await readFile(assetPath(path));
			return [assetPath(path), hash(bytes)];
		})
	)
);
originalFiles[sourceReportPath] = hash(sourceReportBytes);
const additionalSources = [
	'src/lib/token-stories/live-protocol.ts',
	'src/lib/token-stories/live-runtime.ts',
	'src/lib/token-stories/live-worker.ts',
	'src/lib/token-stories/live-engine.ts',
	'src/lib/token-stories/live-archive.ts',
	'scripts/measure-live-stories.mjs'
];
const sourceHashes = { ...sourceReport.provenance.sourceHashes };
for (const path of additionalSources) sourceHashes[path] = hash(await readFile(path));
async function verifySourceFiles() {
	for (const [path, expected] of Object.entries({ ...sourceHashes, ...originalFiles }))
		assert.equal(hash(await readFile(path)), expected, `Source changed during capture: ${path}`);
}
await verifySourceFiles();
const provenance = {
	...sourceReport.provenance,
	source: 'reference',
	commit: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(),
	workingTreeDirty: Boolean(
		execFileSync('git', ['status', '--porcelain'], { encoding: 'utf8' }).trim()
	),
	parentRunId: sourceId,
	sourceHashes
};
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu'] });
try {
	const page = await browser.newPage({ acceptDownloads: true });
	page.on('console', (message) => {
		if (message.text().startsWith('LIVE_STORY ')) console.log(message.text().slice(11));
	});
	page.on('pageerror', (error) => console.error(error));
	await page.goto(new URL('/robots.txt', baseUrl).href);
	const captured = await page.evaluate(
		async ({ reference, provenance, backend, sourceReportHash }) => {
			const { loadTokenStoryReference } = await import('/src/lib/token-stories/archive.ts');
			const { LiveTokenStoryEngine } = await import('/src/lib/token-stories/live-engine.ts');
			const { validateTokenStoryRun } = await import('/src/lib/token-stories/live-archive.ts');
			const source = await loadTokenStoryReference(reference);
			if (!source.checkpoint) throw new Error('The source does not contain resumable weights');
			const engine = new LiveTokenStoryEngine((event) => {
				if (event.type === 'status') console.log('LIVE_STORY ' + event.message);
			});
			window.__liveDemo = { engine };
			const equalCheckpoints = (left, right, label) => {
				for (const key of [
					'version',
					'architecture',
					'modelId',
					'seed',
					'step',
					'trainedTokens',
					'elapsedMs',
					'trainLoss',
					'trainRngState',
					'corpusId',
					'tokenizerId'
				])
					if (!Object.is(left[key], right[key]))
						throw new Error(`${label}: checkpoint ${key} differs`);
				for (const [key, value] of Object.entries(left.config))
					if (right.config[key] !== value) throw new Error(`${label}: architecture differs`);
				if (left.optimizer.t !== right.optimizer.t)
					throw new Error(`${label}: Adam counter differs`);
				for (const [a, b] of [
					[left.parameters, right.parameters],
					[left.optimizer.m, right.optimizer.m],
					[left.optimizer.v, right.optimizer.v]
				]) {
					if (a.length !== b.length) throw new Error(`${label}: tensor count differs`);
					for (let i = 0; i < a.length; i++) {
						if (
							JSON.stringify(a[i].shape) !== JSON.stringify(b[i].shape) ||
							a[i].values.length !== b[i].values.length
						)
							throw new Error(`${label}: tensor shape differs`);
						const aBits = new Uint32Array(
							a[i].values.buffer,
							a[i].values.byteOffset,
							a[i].values.length
						);
						const bBits = new Uint32Array(
							b[i].values.buffer,
							b[i].values.byteOffset,
							b[i].values.length
						);
						if (!aBits.every((value, j) => value === bBits[j]))
							throw new Error(`${label}: tensor bits differ`);
					}
				}
			};
			const restoreStart = performance.now();
			const initialized = await engine.loadCheckpoint(source.checkpoint, backend);
			const restorationMs = performance.now() - restoreStart;
			const before = await engine.exportCheckpoint();
			equalCheckpoints(source.checkpoint, before, 'After loading source checkpoint');
			const frames = [],
				prompt = 'Once upon a time';
			const options = { maxTokens: 24, seed: 71, temperature: 0.8, topK: 40 };
			const started = performance.now();
			const generation = await engine.generateLive(prompt, options, (frame) => {
				frames.push(frame);
				console.log(
					`LIVE_STORY acknowledged frame ${frame.index + 1}: ${JSON.stringify(frame.sampledPiece)}`
				);
			});
			const generationMs = performance.now() - started;
			const after = await engine.exportCheckpoint();
			equalCheckpoints(before, after, 'After live generation');
			if (frames.length !== generation.tokenIds.length || generation.cancelled)
				throw new Error(
					'Live demo did not complete with exactly one acknowledged frame per generated token'
				);
			const now = new Date().toISOString(),
				id = crypto.randomUUID();
			const generationIndex = source.samples?.length ?? 0;
			const inheritedMeasurements = {
				sourceRunId: source.id,
				sourceArchiveSHA256: reference.sha256,
				sourceReportSHA256: sourceReportHash,
				metricCount: source.metrics.length,
				mapCount: source.snapshots.length,
				sampleCount: source.samples?.length ?? 0,
				interventionCount: source.interventions?.length ?? 0,
				note: 'All training metrics, activation maps, earlier samples and selected-unit interventions are inherited unchanged from the source study. They were not remeasured during this inference-only capture.'
			};
			const record = {
				...source,
				id,
				title: `Live replay · TinyStories BPE ${source.presetId} at ${after.step}`,
				createdAt: now,
				updatedAt: now,
				checkpoint: after,
				samples: [...(source.samples ?? []), generation],
				liveTraces: [...(source.liveTraces ?? []), { generationIndex, frames }],
				observations: [
					...source.observations,
					{
						time: now,
						step: after.step,
						title: 'Inherited source checkpoint and measurements',
						detail: `Source run ${source.id}; archive SHA-256 ${reference.sha256}. Training metrics, four maps, prior samples and historical intervention vectors are retained unchanged. This run adds inference evidence only; no training or new intervention was performed.`
					},
					{
						time: now,
						step: after.step,
						title: 'Live token trace captured and checkpoint preserved',
						detail: `${frames.length} acknowledged output tokens from ${JSON.stringify(prompt)}, sampling seed 71, temperature 0.8, top-k 40, maximum 24 tokens. Every frame retains the exact sliding input IDs, final-input-position activations of all ${initialized.unitCount} post-ReLU channels, and raw ${initialized.config.vocabularySize}-way next-token probabilities. No artificial pacing was inserted. Weights, Adam moments and counter, training RNG, supervised-token count and checkpoint metadata matched bit-for-bit before and after.`
					}
				],
				provenance: { ...provenance, browser: navigator.userAgent }
			};
			validateTokenStoryRun(record);
			window.__liveDemo.record = record;
			const liveCapture = {
				version: 1,
				generationIndex,
				frameCount: frames.length,
				checkpointStep: after.step,
				modelId: after.modelId,
				prompt,
				options,
				backend: initialized.backend,
				restorationMs,
				generationMs,
				restoredEvaluation: initialized.metrics,
				checkpointBitwisePreservedOnLoad: true,
				checkpointBitwisePreservedAfterGeneration: true,
				inputPositionMeaning:
					'Final real input position predicts the sampled output token; hidden activations are post-ReLU and ordered by layer then channel.',
				probabilityMeaning:
					'Full raw next-token distribution before temperature, top-k filtering, and BOS exclusion.',
				acknowledgement:
					'Each callback stored its complete measured frame and returned immediately. The worker committed that token only after ACK.',
				limitation:
					'Inherited maps and intervention vectors come from the original worker. The new capture is a separate forward evaluation of the same saved weights, not a claim of cross-worker bitwise activation reproducibility.'
			};
			return {
				id,
				title: record.title,
				presetId: record.presetId,
				seed: record.seed,
				metrics: record.metrics,
				samples: record.samples,
				generation,
				corpusId: record.corpusId,
				tokenizerId: record.tokenizer.id,
				provenance: record.provenance,
				inheritedMeasurements,
				liveCapture
			};
		},
		{ reference, provenance, backend, sourceReportHash: hash(sourceReportBytes) }
	);

	const fullDownload = page.waitForEvent('download', { timeout: 180000 });
	await page.evaluate(async () => {
		const { exportTokenStoryRun } = await import('/src/lib/token-stories/live-archive.ts');
		exportTokenStoryRun(window.__liveDemo.record);
	});
	const fullPath = join(output, 'live-token-stories-complete.tissue');
	await (await fullDownload).saveAs(fullPath);
	const fullBytes = await readFile(fullPath);
	const report = {
		...sourceReport,
		...captured,
		fullArchive: { path: fullPath, bytes: fullBytes.length, sha256: hash(fullBytes) },
		sourceRunId: reference.id,
		probe: { ...sourceReport.probe, inherited: true, sourceRunId: reference.id },
		published: undefined
	};
	await verifySourceFiles();
	if (args.includes('--publish')) {
		const publicDownload = page.waitForEvent('download', { timeout: 180000 });
		await page.evaluate(async () => {
			const { exportTokenStoryRun } = await import('/src/lib/token-stories/live-archive.ts');
			const record = window.__liveDemo.record;
			const published = {
				...record,
				checkpoint: null,
				observations: [
					...record.observations,
					{
						time: new Date().toISOString(),
						step: record.metrics.at(-1).step,
						title: 'Public replay evidence; original resumable source retained',
						detail: `This public demo omits duplicated weights and optimizer tensors. Its ${record.liveTraces.at(-1).frames.length} measured frames can replay without model allocation. Resume training from the original source specimen ${record.provenance.parentRunId}; the complete new inference archive was also retained locally.`
					}
				]
			};
			exportTokenStoryRun(published);
		});
		const filename = `token-stories-live-demo-${captured.id.slice(0, 8)}.tissue`;
		const localPublicPath = join(output, filename);
		await (await publicDownload).saveAs(localPublicPath);
		const bytes = await readFile(localPublicPath);
		assert(
			bytes.length <= 20 * 1024 * 1024,
			'Replay archive unexpectedly exceeds the single-asset size budget'
		);
		const entry = {
			id: captured.id,
			title: captured.title,
			presetId: captured.presetId,
			seed: captured.seed,
			step: captured.metrics.at(-1).step,
			parameterCount: sourceReport.parameterCount,
			unitCount: sourceReport.unitCount,
			file: `/experiments/${filename}`,
			reportFile: `/experiments/${filename.replace(/\.tissue$/, '-report.json')}`,
			bytes: bytes.length,
			sha256: hash(bytes),
			validationLoss: captured.metrics.at(-1).validationLoss,
			unigramLoss: captured.metrics.at(-1).unigramLoss
		};
		report.published = entry;
		await verifySourceFiles();
		await writeFile(assetPath(entry.file), bytes);
		await writeFile(assetPath(entry.reportFile), JSON.stringify(report, null, 2) + '\n');
		const latestCatalog = await readJSON(catalogPath);
		assert.deepEqual(
			latestCatalog.references.find((item) => item.id === reference.id),
			reference,
			'Source reference changed during capture'
		);
		assert(!latestCatalog.references.some((item) => item.id === entry.id), 'New run ID collision');
		latestCatalog.references.push(entry);
		await writeFile(catalogPath, JSON.stringify(latestCatalog, null, 2) + '\n');
	}
	await writeFile(
		join(output, 'live-token-stories-report.json'),
		JSON.stringify(report, null, 2) + '\n'
	);
	await page.evaluate(async () => {
		await window.__liveDemo.engine.dispose();
	});
	console.log(
		JSON.stringify(
			{
				output,
				id: captured.id,
				sourceRunId: reference.id,
				liveCapture: captured.liveCapture,
				completion: captured.generation.completion,
				fullArchive: report.fullArchive,
				published: report.published
			},
			null,
			2
		)
	);
} catch (error) {
	await writeFile(
		join(output, 'failure.json'),
		JSON.stringify(
			{ time: new Date().toISOString(), error: String(error), sourceId, provenance },
			null,
			2
		) + '\n'
	);
	throw error;
} finally {
	await browser.close();
}
