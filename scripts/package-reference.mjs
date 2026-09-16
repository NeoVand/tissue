/** Package measured captures without retraining or synthesizing intermediate anatomy.
 * Requires Node 24 (native TypeScript type stripping).
 * node scripts/package-reference.mjs --input-dir /tmp --output-dir static/experiments
 * node scripts/package-reference.mjs --verify-dir /tmp/tissue-study-seed7
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildGeometry, getNeighbors } from '../src/lib/lab/geometry.ts';
import { MODEL_CONFIG } from '../src/lib/lab/protocol.ts';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
function option(name, fallback) {
	const index = args.indexOf(name);
	if (index === -1) return fallback;
	if (!args[index + 1] || args[index + 1].startsWith('--'))
		throw new Error(`${name} needs a value`);
	return args[index + 1];
}
const allowed = new Set([
	'--input-dir',
	'--output-dir',
	'--capture500',
	'--capture2000',
	'--repair',
	'--verify-dir'
]);
for (let i = 0; i < args.length; i += 2) {
	if (!allowed.has(args[i])) throw new Error(`Unknown option ${args[i]}`);
}
const input = resolve(option('--input-dir', '/tmp'));
const output = resolve(option('--output-dir', join(root, 'static/experiments')));
if (args.includes('--verify-dir')) {
	await packageVerification(resolve(option('--verify-dir', '')), output);
	process.exit(0);
}
const inputs = {
	capture500: resolve(option('--capture500', join(input, 'tissue-reference-run.json'))),
	capture2000: resolve(option('--capture2000', join(input, 'tissue-reference-2000.json'))),
	repair: resolve(option('--repair', join(input, 'tissue-repair-2000.json')))
};
const sources = Object.fromEntries(
	await Promise.all(
		Object.entries(inputs).map(async ([name, path]) => [name, await readFile(path, 'utf8')])
	)
);
const earlier = JSON.parse(sources.capture500);
const latest = JSON.parse(sources.capture2000);
const repairCapture = JSON.parse(sources.repair);
const repair = repairCapture.repair;

// This package has a stable identity; reject accidentally substituted seeds or captures.
assert.equal(earlier.initialized.seed, 42);
assert.equal(latest.initialized.seed, 42);
assert.deepEqual(earlier.initialized, latest.initialized);
assert.equal(earlier.trajectory.length, 20);
assert.equal(latest.trajectory.length, 80);
assert.deepEqual(earlier.trajectory, latest.trajectory.slice(0, earlier.trajectory.length));
for (const [capture, step] of [
	[earlier, 500],
	[latest, 2000]
]) {
	assert.equal(capture.checkpoint.step, step);
	assert.equal(capture.atlas.step, step);
	assert.equal(capture.probe.step, step);
	assert.equal(capture.trajectory.at(-1).step, step);
	assert.equal(capture.checkpoint.architecture, 'binding-transformer-v1');
	assert.equal(capture.atlas.intervention, 'zero-all-token-positions');
	assert.equal(capture.atlas.activationFingerprints.length, 256);
	assert.equal(capture.atlas.effectFingerprints.length, 256);
	assert.equal(capture.initialized.backend, 'webgpu');
}
assert.equal(repair.step, 2000);
assert.equal(repair.seed, 42);
assert.equal(repairCapture.checkpointPreserved, true);

function compact(geometry) {
	return {
		positions: geometry.positions,
		edges: geometry.edges,
		magnitudes: geometry.magnitudes,
		valid: geometry.valid,
		neighbors: geometry.positions.map((_, i) =>
			getNeighbors(geometry, i, 5).map((neighbor) => neighbor.index)
		),
		neighborRetention: geometry.neighborRetention,
		explainedVariance: geometry.explainedVariance,
		rank: geometry.rank,
		boundaryDegenerate: geometry.boundaryDegenerate,
		alignment: geometry.alignment
	};
}

let previousActivation;
let previousEffect;
const snapshots = [earlier, latest].map((capture) => {
	const activation = buildGeometry(capture.atlas.activationFingerprints, {
		kind: 'activation',
		previous: previousActivation
	});
	const effect = buildGeometry(capture.atlas.effectFingerprints, {
		kind: 'effect',
		previous: previousEffect
	});
	previousActivation = activation;
	previousEffect = effect;
	return {
		step: capture.atlas.step,
		metrics: capture.trajectory.at(-1),
		activation: compact(activation),
		effect: compact(effect),
		probe: capture.probe
	};
});

const pct = (value) => `${(value * 100).toFixed(1)}%`;
const number = (value) => value.toFixed(4);
const plateau = earlier.trajectory.at(-1);
const final = latest.trajectory.at(-1);
const step1750 = latest.trajectory.find((metric) => metric.step === 1750);
const armOutcomes = repair.arms
	.map((arm) => {
		const last = arm.curve.at(-1);
		return `${arm.method}: ${pct(last.accuracy)}, loss ${number(last.loss)}`;
	})
	.join('; ');
const id = 'tissue-reference-binding-seed-42-v1';
const run = {
	version: 1,
	id,
	createdAt: earlier.atlas.capturedAt,
	updatedAt: repair.completedAt,
	title: 'Seed 42 / a late learning transition',
	seed: 42,
	initialization: latest.initialized,
	metrics: [latest.initialized.metrics, ...latest.trajectory],
	snapshots,
	observations: [
		{
			id: `${id}-500`,
			time: earlier.atlas.capturedAt,
			step: 500,
			kind: 'training',
			title: 'Falling loss, little evidence of retrieval',
			detail: `At 500 steps, held-out answer accuracy was ${pct(plateau.accuracy)} (${Math.round(plateau.accuracy * plateau.validationExamples)}/${plateau.validationExamples}), with answer loss ${number(plateau.validationLoss)} nats. This is near the 33.3% random input-copy baseline. Falling training loss alone did not establish variable binding. The measured activation and intervention atlases at this checkpoint are retained.`
		},
		{
			id: `${id}-2000`,
			time: latest.atlas.capturedAt,
			step: 2000,
			kind: 'training',
			title: 'A late improvement on held-out assignments',
			detail: `Accuracy was ${pct(step1750.accuracy)} at step 1750 and reached ${pct(final.accuracy)} (${Math.round(final.accuracy * final.validationExamples)}/${final.validationExamples}) at step 2000; final answer loss was ${number(final.validationLoss)} nats. This single seed demonstrates held-out learning on this fixed test set. It does not establish a general learning transition or a causal relation between geometry and capability.`
		},
		{
			id: `${id}-repair`,
			time: repair.completedAt,
			step: 2000,
			kind: 'intervention',
			title: 'The first lesion was too mild to separate the maps',
			detail: `Silencing neuron ${repair.lesionNeuron}, selected by the largest squared effect fingerprint on calibration prompts, changed held-out accuracy from ${pct(repair.intact.accuracy)} to ${pct(repair.lesioned.accuracy)} and loss from ${number(repair.intact.loss)} to ${number(repair.lesioned.loss)}. Four arms each updated ${repair.neighborCount} surviving units for ${repair.steps} steps: ${armOutcomes}. The small initial injury, one seed, and one lesion cannot support a superiority claim for either map. Random neighbors were not matched for activation strength. Original weights were restored after the experiment.`
		},
		{
			id: `${id}-record`,
			time: repair.completedAt,
			step: 2000,
			kind: 'measurement',
			title: 'Measured captures, with gaps kept visible',
			detail:
				'This reference contains the measured step-0 evaluation and 80 evaluations at 25-step intervals, but only two captured anatomies: steps 500 and 2000. No intermediate geometry or initial anatomy was synthesized. The latest raw fingerprints, resumable checkpoint, and repair outcomes are included. The previous raw atlas and checkpoint are preserved in binding-seed-42-500-raw.json. Browser runs used Playwright Chromium with the WebGPU backend; browser version, user-agent string, and adapter identity were not captured.'
		}
	],
	repairs: [repair],
	checkpoint: latest.checkpoint,
	atlas: latest.atlas,
	provenance: {
		source: 'reference',
		appVersion: '0.0.1',
		userAgent:
			'Measured in headless Playwright Chromium with JaxJS WebGPU; exact browser user-agent and graphics adapter were not captured.'
	}
};

const contents = JSON.stringify(run);
const provenance = {
	version: 1,
	runId: id,
	packagedUsing: 'scripts/package-reference.mjs; geometric projection only, no retraining',
	modelConfig: MODEL_CONFIG,
	architecture: latest.checkpoint.architecture,
	seed: 42,
	backend: latest.initialized.backend,
	captureEnvironment: {
		browser: 'headless Playwright Chromium',
		userAgent: null,
		graphicsAdapter: null
	},
	inputSha256: Object.fromEntries(
		Object.entries(sources).map(([name, text]) => [
			name,
			createHash('sha256').update(text).digest('hex')
		])
	),
	capturedAt: {
		step500: earlier.atlas.capturedAt,
		step2000: latest.atlas.capturedAt,
		repair: repair.completedAt
	},
	timings: { step500: earlier.timings, step2000: latest.timings, repairMs: repairCapture.repairMs },
	repairCheckpointPreserved: repairCapture.checkpointPreserved,
	cancellationCheck: repairCapture.cancellation,
	outputSha256: createHash('sha256').update(contents).digest('hex'),
	note: 'MODEL_CONFIG describes the packaged architecture. Input captures did not record a source commit, browser version, or hardware adapter. Capture times are retained; createdAt is the earliest atlas capture, not an inferred training start.'
};
await mkdir(output, { recursive: true });
await writeFile(join(output, 'binding-seed-42.json'), contents);
await copyFile(inputs.capture500, join(output, 'binding-seed-42-500-raw.json'));
await writeFile(
	join(output, 'binding-seed-42-provenance.json'),
	JSON.stringify(provenance, null, 2) + '\n'
);
console.log(
	JSON.stringify(
		{
			path: join(output, 'binding-seed-42.json'),
			bytes: Buffer.byteLength(contents),
			metrics: run.metrics.length,
			snapshots: run.snapshots.map((snapshot) => snapshot.step),
			repairs: run.repairs.length,
			accuracy: final.accuracy,
			sha256: provenance.outputSha256
		},
		null,
		2
	)
);

/** Package any complete verify-model capture directory, preserving its recorded provenance. */
async function packageVerification(directory, destination) {
	const sourceProvenance = await readFile(join(directory, 'provenance.json'), 'utf8');
	const sourceSummary = await readFile(join(directory, 'summary.json'), 'utf8');
	const verified = JSON.parse(sourceProvenance);
	const summary = JSON.parse(sourceSummary);
	assert.deepEqual(summary.provenance, verified);
	assert.ok(Number.isInteger(verified.seed));
	assert.ok(Array.isArray(verified.checkpoints) && verified.checkpoints.length > 0);
	const seed = verified.seed;
	const raw = await Promise.all(
		verified.checkpoints.map(async (step) => {
			const name = `seed-${seed}-step-${step}.json`;
			const text = await readFile(join(directory, name), 'utf8');
			return { step, name, text, data: JSON.parse(text) };
		})
	);
	const latest = raw.at(-1).data;
	const latestMetric = latest.metrics;
	assert.deepEqual(summary.finalMetrics, latestMetric);
	assert.equal(summary.passedLearningThreshold, latestMetric.accuracy >= verified.minimumAccuracy);
	for (const [index, { step, data }] of raw.entries()) {
		assert.deepEqual(data.provenance, verified);
		assert.deepEqual(data.initialized, latest.initialized);
		assert.equal(data.initialized.seed, seed);
		assert.equal(data.checkpoint.seed, seed);
		assert.equal(data.atlas.seed, seed);
		assert.equal(data.checkpoint.step, step);
		assert.equal(data.atlas.step, step);
		assert.equal(data.probe.step, step);
		assert.equal(data.metrics.step, step);
		assert.equal(data.checkpoint.architecture, 'binding-transformer-v1');
		assert.deepEqual(data.metrics, data.trajectory.at(-1));
		assert.deepEqual(data.trajectory, latest.trajectory.slice(0, data.trajectory.length));
		if (index > 0) assert.ok(step > raw[index - 1].step);
	}
	let precedingActivation;
	let precedingEffect;
	const snapshots = raw.map(({ data }) => {
		const activation = buildGeometry(data.atlas.activationFingerprints, {
			kind: 'activation',
			previous: precedingActivation
		});
		const effect = data.atlas.effectFingerprints
			? buildGeometry(data.atlas.effectFingerprints, { kind: 'effect', previous: precedingEffect })
			: undefined;
		precedingActivation = activation;
		precedingEffect = effect;
		return {
			step: data.metrics.step,
			metrics: data.metrics,
			activation: compact(activation),
			...(effect ? { effect: compact(effect) } : {}),
			probe: data.probe
		};
	});
	const percentage = (value) => `${(value * 100).toFixed(1)}%`;
	const id = `tissue-reference-binding-seed-${seed}-v1`;
	const run = {
		version: 1,
		id,
		createdAt: verified.recordedAt,
		updatedAt: latest.atlas.capturedAt,
		title: `Seed ${seed} / ${summary.passedLearningThreshold ? 'a second late learning transition' : 'a learning threshold not reached'}`,
		seed,
		initialization: latest.initialized,
		metrics: [latest.initialized.metrics, ...latest.trajectory],
		snapshots,
		observations: [
			...raw.map(({ data, step }) => ({
				id: `${id}-${step}`,
				time: data.atlas.capturedAt,
				step,
				kind: 'training',
				title:
					data.metrics.accuracy > 0.8
						? 'Held-out retrieval emerged'
						: 'Accuracy remained near input-copy performance',
				detail: `At step ${step}, measured held-out accuracy was ${percentage(data.metrics.accuracy)} (${Math.round(data.metrics.accuracy * data.metrics.validationExamples)}/${data.metrics.validationExamples}), with answer loss ${data.metrics.validationLoss.toFixed(4)} nats. The random input-copy baseline is 33.3%. Activation${data.atlas.effectFingerprints ? ' and intervention' : ''} anatomy was captured at this checkpoint. ${data.metrics.accuracy > 0.8 ? 'This is evidence of learning on this fixed test set, without establishing that the geometry caused or predicted the change.' : 'Lower loss at this point is insufficient evidence that the model is retrieving the queried binding.'}`
			})),
			{
				id: `${id}-provenance`,
				time: latest.atlas.capturedAt,
				step: latestMetric.step,
				kind: 'measurement',
				title: 'A second seed, with its full evidence retained',
				detail: `The verification run retained ${latest.trajectory.length + 1} measured evaluations and exactly ${snapshots.length} anatomy snapshots (${snapshots.map((snapshot) => snapshot.step).join(', ')}). Original capture files, source hashes, checkpoint weights, optimizer state, and verification summary are archived with this record. The recorded backend was ${latest.initialized.backend}; browser version and adapter identity were not captured. No repair experiment was performed for this seed. Two seeds with late improvements do not establish a general law or superiority of either map.`
			}
		],
		checkpoint: latest.checkpoint,
		atlas: latest.atlas,
		provenance: {
			source: 'reference',
			appVersion: '0.0.1',
			userAgent: `Measured in headless Playwright Chromium with JaxJS ${latest.initialized.backend}; exact browser user-agent and graphics adapter were not captured. Source hashes are preserved in binding-seed-${seed}-provenance.json.`
		}
	};
	const contents = JSON.stringify(run);
	await mkdir(destination, { recursive: true });
	await writeFile(join(destination, `binding-seed-${seed}.json`), contents);
	await writeFile(join(destination, `binding-seed-${seed}-provenance.json`), sourceProvenance);
	await writeFile(join(destination, `binding-seed-${seed}-summary.json`), sourceSummary);
	for (const { step, text } of raw)
		await writeFile(join(destination, `binding-seed-${seed}-${step}-raw.json`), text);
	const hash = (value) => createHash('sha256').update(value).digest('hex');
	await writeFile(
		join(destination, `binding-seed-${seed}-package.json`),
		JSON.stringify(
			{
				version: 1,
				runId: id,
				modelConfig: MODEL_CONFIG,
				inputSha256: Object.fromEntries([
					['provenance.json', hash(sourceProvenance)],
					['summary.json', hash(sourceSummary)],
					...raw.map(({ name, text }) => [name, hash(text)])
				]),
				outputSha256: hash(contents),
				snapshots: snapshots.map((snapshot) => snapshot.step),
				note: 'Projection recomputed from raw captures; no training or intermediate anatomy synthesized. Original capture/provenance/summary files are preserved byte for byte.'
			},
			null,
			2
		) + '\n'
	);
	console.log(
		JSON.stringify(
			{
				path: join(destination, `binding-seed-${seed}.json`),
				bytes: Buffer.byteLength(contents),
				metrics: run.metrics.length,
				snapshots: snapshots.map((snapshot) => snapshot.step),
				repairs: 0,
				accuracy: latestMetric.accuracy,
				sha256: hash(contents)
			},
			null,
			2
		)
	);
}
