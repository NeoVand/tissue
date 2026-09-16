/** Slow, explicit research reproduction, separate from the routine unit suite.
 * Start pnpm dev, then:
 * node scripts/verify-model.mjs --output /tmp/tissue-study --checkpoints 500,2000
 * Every checkpoint is saved before the learning assertion, including failures.
 */
import { chromium } from 'playwright';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { resolve, join } from 'node:path';

const argv = process.argv.slice(2);
const flag = (name) => argv.includes(name);
function option(name, fallback) {
	const index = argv.indexOf(name);
	if (index < 0) return fallback;
	if (!argv[index + 1] || argv[index + 1].startsWith('--'))
		throw new Error(`${name} requires a value`);
	return argv[index + 1];
}
const output = resolve(option('--output', '/tmp/tissue-study'));
const baseUrl = option('--base-url', 'http://localhost:5173');
const seed = Number(option('--seed', '42'));
const backend = option('--backend', 'auto');
const minimumAccuracy = Number(option('--minimum-accuracy', '0.8'));
const checkpoints = option('--checkpoints', '500,2000').split(',').map(Number);
if (
	!Number.isInteger(seed) ||
	seed < 0 ||
	seed > 0xffffffff ||
	!['auto', 'webgpu', 'wasm', 'cpu'].includes(backend) ||
	!Number.isFinite(minimumAccuracy) ||
	minimumAccuracy < 0 ||
	minimumAccuracy > 1 ||
	checkpoints.some(
		(step, index) =>
			!Number.isInteger(step) ||
			step < 1 ||
			step > 10000 ||
			(index > 0 && step <= checkpoints[index - 1])
	)
)
	throw new Error('Invalid seed, backend, accuracy threshold, or strictly increasing checkpoints');
const sources = [
	'src/lib/lab/protocol.ts',
	'src/lib/lab/model/dataset.ts',
	'src/lib/lab/model/transformer.ts',
	'src/lib/lab/model/runtime.ts',
	'src/lib/lab/model/adam.ts',
	'src/lib/lab/model/repair.ts',
	'src/lib/lab/engine.ts',
	'pnpm-lock.yaml',
	'scripts/verify-model.mjs'
];
const sourceHashes = Object.fromEntries(
	await Promise.all(
		sources.map(async (path) => [
			path,
			createHash('sha256')
				.update(await readFile(path))
				.digest('hex')
		])
	)
);
const provenance = {
	recordedAt: new Date().toISOString(),
	seed,
	requestedBackend: backend,
	commit: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(),
	workingTreeDirty: Boolean(
		execFileSync('git', ['status', '--porcelain'], { encoding: 'utf8' }).trim()
	),
	sourceHashes,
	minimumAccuracy,
	checkpoints,
	objective:
		'Answer-only next-token cross entropy over14tokens; three distinct key-value assignments;256MLP neurons.',
	baselines: { uniformAnswerAccuracy: 1 / 8, randomInputCopyAccuracy: 1 / 3 },
	split:
		'Whole binding mappings236train/50calibration/50test, all queries and presentation orders kept together.16calibration and96fixed balanced test sequences.'
};
await mkdir(output, { recursive: true });
await writeFile(join(output, 'provenance.json'), JSON.stringify(provenance, null, 2));
const browser = await chromium.launch({ headless: true, args: ['--enable-unsafe-webgpu'] });
let finalMetrics;
try {
	const page = await browser.newPage();
	page.on('console', (message) => {
		if (message.text().startsWith('TISSUE ')) console.log(message.text().slice(7));
	});
	page.on('pageerror', (error) => console.error('Browser error:', error.message));
	// A static resource avoids HMR navigation while the model trains in its own worker.
	await page.goto(new URL('/robots.txt', baseUrl).href);
	const initialized = await page.evaluate(
		async ({ seed, backend }) => {
			const { Engine } = await import('/src/lib/lab/engine.ts');
			const trajectory = [];
			const engine = new Engine((event) => {
				if (event.type === 'metrics') {
					trajectory.push(event.metrics);
					if (event.metrics.step % 100 === 0)
						console.log('TISSUE ' + JSON.stringify(event.metrics));
				} else if (event.type === 'status') console.log('TISSUE ' + event.message);
			});
			globalThis.__tissueVerification = { engine, trajectory };
			return engine.initialize(seed, backend);
		},
		{ seed, backend }
	);
	for (const step of checkpoints) {
		const run = await page.evaluate(
			async ({ step, includeEffects }) => {
				const { engine, trajectory } = globalThis.__tissueVerification;
				const before = await engine.exportCheckpoint();
				const start = performance.now();
				const metrics = await engine.train(step - before.step);
				const trainMs = performance.now() - start;
				const measurementStart = performance.now();
				const atlas = await engine.captureAtlas(includeEffects);
				const measurementMs = performance.now() - measurementStart;
				const probe = await engine.probe(0);
				const checkpoint = await engine.exportCheckpoint();
				globalThis.__tissueVerification.atlas = atlas;
				return {
					metrics,
					trajectory,
					atlas,
					probe,
					checkpoint,
					timings: { trainMs, measurementMs }
				};
			},
			{ step, includeEffects: !flag('--skip-effects') }
		);
		await writeFile(
			join(output, `seed-${seed}-step-${step}.json`),
			JSON.stringify({ provenance, initialized, ...run })
		);
		console.log(
			`Saved step${step}: ${(100 * run.metrics.accuracy).toFixed(2)}% held-out accuracy, CE${run.metrics.validationLoss.toFixed(4)}`
		);
		finalMetrics = run.metrics;
	}
	if (!flag('--skip-repair')) {
		const receipt = await page.evaluate(async () => {
			const { engine, atlas } = globalThis.__tissueVerification;
			if (!atlas.effectFingerprints)
				throw new Error(
					'Repair lesion selection requires effects; remove --skip-effects or add --skip-repair'
				);
			const lesionNeuron = atlas.effectFingerprints
				.map((row, id) => ({ id, energy: row.reduce((sum, value) => sum + value * value, 0) }))
				.sort((a, b) => b.energy - a.energy || a.id - b.id)[0].id;
			const before = await engine.exportCheckpoint();
			const start = performance.now();
			const repair = await engine.controlledRepair({
				atlas,
				lesionNeuron,
				steps: 50,
				neighborCount: 8
			});
			const elapsedMs = performance.now() - start;
			const after = await engine.exportCheckpoint();
			if (JSON.stringify(before) !== JSON.stringify(after))
				throw new Error('Repair modified the resident checkpoint');
			const pending = engine.controlledRepair({ atlas, lesionNeuron, steps: 50, neighborCount: 8 });
			setTimeout(() => void engine.pause(), 100);
			let cancellation = '';
			try {
				await pending;
			} catch (error) {
				cancellation = String(error);
			}
			if (!cancellation.includes('cancelled'))
				throw new Error('Repair did not acknowledge cancellation');
			if (JSON.stringify(before) !== JSON.stringify(await engine.exportCheckpoint()))
				throw new Error('Cancelled repair modified the resident checkpoint');
			return {
				repair,
				elapsedMs,
				cancellation,
				checkpointPreserved: true,
				lesionSelection:
					'Highest total squared effect on16calibration probes, selected without test outcomes'
			};
		});
		await writeFile(
			join(output, `seed-${seed}-repair.json`),
			JSON.stringify({ provenance, ...receipt })
		);
		console.log('Repair and cancellation preserve the complete source checkpoint.');
	}
	await page.evaluate(() => globalThis.__tissueVerification.engine.dispose());
	const summary = {
		provenance,
		finalMetrics,
		passedLearningThreshold: finalMetrics.accuracy >= minimumAccuracy
	};
	await writeFile(join(output, 'summary.json'), JSON.stringify(summary, null, 2));
	if (!summary.passedLearningThreshold)
		throw new Error(
			`Learning criterion failed: accuracy${finalMetrics.accuracy} < ${minimumAccuracy}. Raw evidence is preserved in ${output}`
		);
	console.log(`Verified. Evidence: ${output}`);
} finally {
	await browser.close();
}
