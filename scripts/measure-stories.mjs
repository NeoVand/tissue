/** Real browser training and measured maps; no fabricated model data.
 * pnpm dev, then node scripts/measure-stories.mjs --presets small,medium,large --publish
 * Full binary checkpoints stay in --output, including the large model's Adam state.
 */
import { chromium } from 'playwright';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';

const args = process.argv.slice(2);
const option = (name, fallback) => {
	const at = args.indexOf(name);
	if (at < 0) return fallback;
	if (!args[at + 1] || args[at + 1].startsWith('--')) throw new Error(`${name} requires a value`);
	return args[at + 1];
};
const presets = option('--presets', 'small,medium,large').split(',');
if (presets.some((value) => !['small', 'medium', 'large'].includes(value)))
	throw new Error('Invalid preset');
const seed = Number(option('--seed', '42'));
const trainedTokens = Number(option('--tokens', '51200'));
if (!Number.isSafeInteger(trainedTokens) || trainedTokens < 512 || trainedTokens % 512 !== 0)
	throw new Error('Token budget must be a positive multiple of 512');
const backend = option('--backend', 'webgpu');
if (!['webgpu', 'wasm', 'auto'].includes(backend)) throw new Error('Invalid backend');
const baseUrl = option('--base-url', 'http://127.0.0.1:5173');
const output = resolve(option('--output', `/tmp/tissue-stories-${Date.now()}`));
const publish = args.includes('--publish');
const hash = (data) => createHash('sha256').update(data).digest('hex');
const sources = [
	'src/lib/stories/protocol.ts',
	'src/lib/stories/dataset.ts',
	'src/lib/stories/model.ts',
	'src/lib/stories/runtime.ts',
	'src/lib/stories/worker.ts',
	'src/lib/stories/engine.ts',
	'src/lib/stories/geometry.ts',
	'src/lib/stories/geometry-worker.ts',
	'src/lib/stories/geometry-engine.ts',
	'src/lib/stories/archive.ts',
	'src/lib/stories/validate-geometry.ts',
	'src/lib/lab/model/adam.ts',
	'src/lib/lab/geometry.ts',
	'pnpm-lock.yaml',
	'docs/stories-design.md',
	'scripts/measure-stories.mjs',
	'static/data/tinystories/provenance.json'
];
const hashSources = async () =>
	Object.fromEntries(
		await Promise.all(sources.map(async (path) => [path, hash(await readFile(path))]))
	);
const sourceHashes = await hashSources();
const provenance = {
	source: 'reference',
	commit: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(),
	workingTreeDirty: Boolean(
		execFileSync('git', ['status', '--porcelain'], { encoding: 'utf8' }).trim()
	),
	sourceHashes
};
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu'] });
const reports = [];
try {
	for (const preset of presets) {
		const page = await browser.newPage({ acceptDownloads: true });
		page.on('console', (message) => {
			if (message.text().startsWith('STORY '))
				console.log(`[${preset}] ${message.text().slice(6)}`);
		});
		page.on('pageerror', (error) => console.error(`[${preset}] Browser error: ${error.message}`));
		await page.goto(new URL('/robots.txt', baseUrl).href);
		const initialized = await page.evaluate(
			async ({ preset, seed, backend, provenance }) => {
				const { StoryEngine } = await import('/src/lib/stories/engine.ts');
				const { StoryGeometryEngine } = await import('/src/lib/stories/geometry-engine.ts');
				const engine = new StoryEngine((event) => {
					if (event.type === 'status') console.log('STORY ' + event.message);
					if (event.type === 'metrics')
						console.log(
							`STORY step ${event.metrics.step}: held-out ${event.metrics.validationLoss.toFixed(4)}, train ${event.metrics.trainLoss?.toFixed(4)}`
						);
				});
				const geometry = new StoryGeometryEngine();
				const started = performance.now();
				const init = await engine.initialize(preset, seed, backend);
				const now = new Date().toISOString();
				const record = {
					version: 1,
					kind: 'tissue-story-run',
					id: crypto.randomUUID(),
					createdAt: now,
					updatedAt: now,
					title: `TinyStories ${preset} · seed ${seed}`,
					seed,
					presetId: preset,
					metrics: [init.metrics],
					snapshots: [],
					checkpoint: null,
					samples: [],
					observations: [],
					provenance: { ...provenance, browser: navigator.userAgent }
				};
				window.__story = {
					engine,
					geometry,
					record,
					init,
					initializationMs: performance.now() - started,
					timings: []
				};
				return { ...init, initializationMs: window.__story.initializationMs };
			},
			{ preset, seed, backend, provenance }
		);
		console.log(
			`[${preset}] Initialized ${initialized.parameterCount} parameters / ${initialized.unitCount} units in ${(initialized.initializationMs / 1000).toFixed(2)}s`
		);
		const tokensPerStep = initialized.config.batchSize * initialized.config.context;
		const steps = [
			...new Set([
				0,
				Math.floor(trainedTokens / 4 / tokensPerStep),
				Math.floor(trainedTokens / 2 / tokensPerStep),
				trainedTokens / tokensPerStep
			])
		].sort((a, b) => a - b);
		let lastStep = 0;
		for (const step of steps) {
			while (lastStep < step) {
				const burst = Math.min(25, step - lastStep);
				lastStep = await page.evaluate(async (burst) => {
					const { engine, record } = window.__story;
					const metrics = await engine.train(burst, (metric) => {
						record.metrics = [...record.metrics.filter((old) => old.step !== metric.step), metric];
					});
					return metrics.step;
				}, burst);
			}
			const download = page.waitForEvent('download', { timeout: 300000 });
			const capture = await page.evaluate(async () => {
				const { exportStoryRun } = await import('/src/lib/stories/archive.ts');
				const state = window.__story;
				const start = performance.now();
				const atlas = await state.engine.captureAtlas();
				const fitStart = performance.now();
				const geometry = await state.geometry.build(atlas);
				const geometryMs = performance.now() - fitStart;
				state.record.snapshots.push({ capturedAt: atlas.capturedAt, atlas, geometry });
				state.record.checkpoint = await state.engine.exportCheckpoint();
				state.record.updatedAt = new Date().toISOString();
				state.record.observations.push({
					time: state.record.updatedAt,
					step: atlas.step,
					title: 'Measured language-model structure',
					detail: `All ${atlas.unitCount} MLP units captured on eight calibration windows at 16 positions each. Neighbor retention audited on ${geometry.audit.scoredFocals} fixed focal units. Coordinates use calibration activations only.`
				});
				const timing = {
					step: atlas.step,
					atlasMs: atlas.elapsedMs,
					geometryMs,
					captureAndCheckpointMs: performance.now() - start
				};
				state.timings.push(timing);
				exportStoryRun(state.record);
				return {
					...timing,
					validUnits: geometry.valid.filter(Boolean).length,
					variance: geometry.explainedVariance,
					retention: geometry.audit.neighborRetention,
					pca: geometry.pca,
					metric: state.record.metrics.at(-1)
				};
			});
			await (await download).saveAs(join(output, `stories-${preset}-step-${step}.tissue`));
			console.log(`[${preset}] Captured step ${step}: ${JSON.stringify(capture)}`);
		}
		const final = await page.evaluate(async () => {
			const s = window.__story;
			const prompt = 'Once upon a time, there was a little';
			const intact = await s.engine.probe(prompt);
			const selected = Math.floor(intact.unitCount / 2);
			const lesioned = await s.engine.probe(prompt, selected);
			for (let token = 0; token < lesioned.prompt.tokenIds.length; token++)
				if (lesioned.activations[token * lesioned.unitCount + selected] !== 0)
					throw new Error('Selected-unit ablation did not zero its channel');
			const generation = await s.engine.generate(prompt, {
				maxTokens: 96,
				temperature: 0.8,
				topK: 20,
				seed: 71
			});
			const after = await s.engine.exportCheckpoint();
			const before = s.record.checkpoint;
			const equalLeaves = (a, b) =>
				a.length === b.length &&
				a.every((leaf, i) => leaf.values.every((value, j) => value === b[i].values[j]));
			if (
				before.trainRngState !== after.trainRngState ||
				before.step !== after.step ||
				before.optimizer.t !== after.optimizer.t ||
				!equalLeaves(before.parameters, after.parameters) ||
				!equalLeaves(before.optimizer.m, after.optimizer.m) ||
				!equalLeaves(before.optimizer.v, after.optimizer.v)
			)
				throw new Error('Probing or generation changed the resident checkpoint');
			s.record.samples.push(generation);
			s.record.interventions = [
				{
					step: before.step,
					modelId: before.modelId,
					neuron: selected,
					prompt: intact.prompt.original,
					capturedAt: new Date().toISOString(),
					probabilities: intact.probabilities,
					lesionedProbabilities: lesioned.probabilities
				}
			];
			s.record.updatedAt = new Date().toISOString();
			s.record.observations.push({
				time: s.record.updatedAt,
				step: before.step,
				title: 'Probe and sampling state verified',
				detail: `Unit ${selected} was exactly zero after ablation at every prompt position. Intact/ablated inference and 96 sampled characters left weights, Adam state, and training RNG unchanged.`
			});
			const last = s.record.snapshots.at(-1).geometry;
			return {
				id: s.record.id,
				presetId: s.record.presetId,
				seed: s.record.seed,
				title: s.record.title,
				parameterCount: s.init.parameterCount,
				unitCount: s.init.unitCount,
				initializationMs: s.initializationMs,
				metrics: s.record.metrics,
				timings: s.timings,
				generation,
				probe: {
					unit: selected,
					checkpointPreserved: true,
					maxProbabilityChange: Math.max(
						...Array.from(intact.probabilities, (p, i) => Math.abs(p - lesioned.probabilities[i]))
					)
				},
				projection: {
					validUnits: last.valid.filter(Boolean).length,
					variance: last.explainedVariance,
					retention: last.audit.neighborRetention,
					audit: last.audit,
					pca: last.pca
				},
				provenance: s.record.provenance
			};
		});
		const download = page.waitForEvent('download', { timeout: 300000 });
		await page.evaluate(async () => {
			const { exportStoryRun } = await import('/src/lib/stories/archive.ts');
			exportStoryRun(window.__story.record);
		});
		const fullPath = join(output, `stories-${preset}-complete.tissue`);
		await (await download).saveAs(fullPath);
		final.fullArchive = { path: fullPath, sha256: hash(await readFile(fullPath)) };
		if (JSON.stringify(await hashSources()) !== JSON.stringify(sourceHashes))
			throw new Error(
				'Implementation changed during the study. Raw checkpoints were preserved; provenance requires review.'
			);
		if (publish) {
			const download = page.waitForEvent('download', { timeout: 300000 });
			await page.evaluate(async () => {
				const { exportStoryRun } = await import('/src/lib/stories/archive.ts');
				const record = window.__story.record;
				if (record.presetId === 'large') {
					const published = {
						...record,
						checkpoint: null,
						observations: [
							...record.observations,
							{
								time: new Date().toISOString(),
								step: record.metrics.at(-1).step,
								title: 'Inspection-only public reference',
								detail:
									'The complete 129 MB weights and optimizer checkpoint was retained locally by the research runner. This public reference contains raw activations, geometry, learning curves and samples; it cannot resume training.'
							}
						]
					};
					exportStoryRun(published);
				} else exportStoryRun(record);
			});
			const filename = `stories-${preset}-seed-${seed}-step-${lastStep}-${final.id.slice(0, 8)}.tissue`;
			const destination = join(output, filename);
			await (await download).saveAs(destination);
			const bytes = await readFile(destination);
			const chunkBytes = 20 * 1024 * 1024;
			let parts;
			if (bytes.length > chunkBytes) {
				parts = [];
				for (let start = 0; start < bytes.length; start += chunkBytes) {
					const part = `${filename}.part-${parts.length}`;
					await writeFile(
						join('static/experiments', part),
						bytes.subarray(start, start + chunkBytes)
					);
					parts.push(`/experiments/${part}`);
				}
			} else await writeFile(join('static/experiments', filename), bytes);
			const entry = {
				id: final.id,
				title: final.title,
				presetId: preset,
				seed,
				step: lastStep,
				parameterCount: final.parameterCount,
				unitCount: final.unitCount,
				file: `/experiments/${filename}`,
				reportFile: `/experiments/${filename.replace(/\.tissue$/, '-report.json')}`,
				...(parts ? { parts } : {}),
				bytes: bytes.length,
				sha256: hash(bytes),
				validationLoss: final.metrics.at(-1).validationLoss,
				unigramLoss: final.metrics.at(-1).unigramLoss
			};
			const catalog = JSON.parse(await readFile('static/experiments/stories-index.json', 'utf8'));
			catalog.references = [...catalog.references.filter((r) => r.id !== entry.id), entry];
			await writeFile(
				'static/experiments/stories-index.json',
				JSON.stringify(catalog, null, 2) + '\n'
			);
			final.published = entry;
		}
		reports.push(final);
		await writeFile(
			join(output, `stories-${preset}-report.json`),
			JSON.stringify(final, null, 2) + '\n'
		);
		if (publish)
			await writeFile(`static${final.published.reportFile}`, JSON.stringify(final, null, 2) + '\n');
		await page.evaluate(async () => {
			await window.__story.engine.dispose();
			window.__story.geometry.destroy();
		});
		await page.close();
	}
} catch (error) {
	await writeFile(
		join(output, 'failure.json'),
		JSON.stringify(
			{ time: new Date().toISOString(), error: String(error), completed: reports, provenance },
			null,
			2
		) + '\n'
	);
	throw error;
} finally {
	await browser.close();
}
console.log(`Saved ${reports.length} model studies to ${output}`);
