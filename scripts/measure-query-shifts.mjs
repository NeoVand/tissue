/** Measure the locked paired-query study on saved checkpoints, preserving raw evidence.
 * Keep pnpm dev running. node scripts/measure-query-shifts.mjs --seeds 42,7
 * --analyze-only recomputes results from the saved measurements without running a model.
 */
import { chromium } from 'playwright';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';

const args = process.argv.slice(2);
const option = (name, fallback) => {
	const index = args.indexOf(name);
	if (index < 0) return fallback;
	if (!args[index + 1] || args[index + 1].startsWith('--'))
		throw new Error(`${name} needs a value`);
	return args[index + 1];
};
const seeds = option('--seeds', '42,7').split(',').map(Number);
if (!seeds.length || seeds.some((seed) => ![42, 7].includes(seed)))
	throw new Error('Supported saved references: seeds 42 and 7');
const baseUrl = option('--base-url', 'http://localhost:5173');
const output = resolve(option('--output', 'static/experiments'));
const backend = option('--backend', 'auto');
if (!['auto', 'webgpu', 'wasm', 'cpu'].includes(backend)) throw new Error('Invalid backend');
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const hashes = async (paths) =>
	Object.fromEntries(
		await Promise.all(paths.map(async (path) => [path, sha256(await readFile(path))]))
	);
const sourceFiles = [
	'src/lib/lab/protocol.ts',
	'src/lib/lab/query-protocol.ts',
	'src/lib/lab/model/dataset.ts',
	'src/lib/lab/model/query-dataset.ts',
	'src/lib/lab/model/transformer.ts',
	'src/lib/lab/model/runtime.ts',
	'src/lib/lab/model/adam.ts',
	'src/lib/lab/model/worker.ts',
	'src/lib/lab/engine.ts',
	'pnpm-lock.yaml',
	'scripts/measure-query-shifts.mjs'
];
const sourceHashes = await hashes(sourceFiles);
const designHash = sha256(await readFile('docs/query-shifts-design.md'));
const commit = execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
const workingTreeDirty = Boolean(
	execFileSync('git', ['status', '--porcelain'], { encoding: 'utf8' }).trim()
);
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu'] });
const reports = [];
try {
	const page = await browser.newPage();
	page.on('console', (message) => {
		if (message.text().startsWith('QUERY ')) console.log(message.text().slice(6));
	});
	page.on('pageerror', (error) => console.error('Browser error:', error.message));
	// A static page prevents Svelte HMR from interrupting independent measurement workers.
	await page.goto(new URL('/robots.txt', baseUrl).href);
	for (const seed of seeds) {
		const file = join(output, `query-shifts-seed-${seed}.json`);
		let record;
		if (args.includes('--analyze-only')) record = JSON.parse(await readFile(file, 'utf8'));
		else {
			const checkpointFile = `static/experiments/binding-seed-${seed}.json`;
			const checkpointBytes = await readFile(checkpointFile);
			const { checkpoint } = JSON.parse(checkpointBytes);
			console.log(`Measuring seed ${seed}, step ${checkpoint.step} on ${backend}`);
			const measurement = await page.evaluate(
				async ({ checkpoint, backend }) => {
					const { Engine } = await import('/src/lib/lab/engine.ts');
					const engine = new Engine((event) => {
						if (event.type === 'status') console.log('QUERY ' + event.message);
						if (event.type === 'measurement' && event.completed % 64 === 0)
							console.log(`QUERY ${event.completed}/${event.total} lesions`);
					});
					try {
						await engine.initialize(checkpoint.seed, backend);
						await engine.loadCheckpoint(checkpoint);
						return await engine.measureQueryShifts();
					} finally {
						await engine.dispose();
					}
				},
				{ checkpoint, backend }
			);
			record = {
				version: 1,
				kind: 'tissue-query-study',
				id: `query-reference-${seed}-step-${measurement.step}-v1`,
				createdAt: measurement.capturedAt,
				source: 'reference',
				measurement,
				provenance: {
					commit,
					workingTreeDirty,
					sourceHashes,
					checkpointFile,
					checkpointFileHash: sha256(checkpointBytes),
					designHash
				}
			};
			// Save evidence before analysis or any interpretation of its outcome.
			await writeFile(file, JSON.stringify(record));
			if (JSON.stringify(await hashes(sourceFiles)) !== JSON.stringify(sourceHashes))
				throw new Error(
					'Measurement implementation changed during capture; raw evidence was preserved, but provenance must be checked'
				);
			console.log(`Saved raw evidence: ${file} (${Math.round(measurement.elapsedMs)} ms)`);
		}
		const analysisHashes = await hashes([
			'src/lib/lab/query-analysis.ts',
			'src/lib/lab/geometry.ts',
			'docs/query-shifts-design.md'
		]);
		const analysis = await page.evaluate(async (record) => {
			const { validateQueryStudy } = await import('/src/lib/lab/query-journal.ts');
			const { analyzeQueryShifts } = await import('/src/lib/lab/query-analysis.ts');
			validateQueryStudy(record);
			return analyzeQueryShifts(record.measurement);
		}, record);
		const report = {
			seed,
			step: record.measurement.step,
			capturedAt: record.measurement.capturedAt,
			measurementHash: sha256(JSON.stringify(record.measurement)),
			sourceHashes: analysisHashes,
			analysis
		};
		await writeFile(
			join(output, `query-shifts-seed-${seed}-analysis.json`),
			JSON.stringify(report)
		);
		reports.push(report);
		console.log(`Analyzed seed ${seed}: ${JSON.stringify(analysis.overall)}`);
	}
	await writeFile(
		join(output, 'query-shifts-summary.json'),
		JSON.stringify(
			{
				version: 1,
				design: 'paired-query-v1',
				designHash,
				analyzedAt: new Date().toISOString(),
				runs: reports.map(({ analysis, ...metadata }) => ({
					...metadata,
					summary: analysis.overall,
					layers: analysis.layers,
					projection: {
						neighborRetention: analysis.calibrationGeometry.neighborRetention,
						explainedVariance: analysis.calibrationGeometry.explainedVariance
					}
				}))
			},
			null,
			2
		)
	);
} finally {
	await browser.close();
}
