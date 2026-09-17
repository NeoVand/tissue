/** Recompute descriptive fingerprint comparisons from the recorded reference runs.
 * Requires Node 24 (native TypeScript type stripping). No model training or inference.
 * node scripts/compare-fingerprints.mjs
 * node scripts/compare-fingerprints.mjs --input-dir static/experiments --output /tmp/comparison.json
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildGeometry, getNeighbors } from '../src/lib/lab/geometry.ts';
import { MODEL_CONFIG, NEURON_COUNT } from '../src/lib/lab/protocol.ts';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const k = 6;
const args = process.argv.slice(2);
const options = new Map();
for (let i = 0; i < args.length; i += 2) {
	assert(['--input-dir', '--output'].includes(args[i]), `Unknown option ${args[i]}`);
	assert(args[i + 1] && !args[i + 1].startsWith('--'), `${args[i]} needs a value`);
	options.set(args[i], args[i + 1]);
}
const input = resolve(options.get('--input-dir') ?? join(root, 'static/experiments'));
const output = resolve(
	options.get('--output') ?? join(root, 'static/experiments/fingerprint-comparison.json')
);
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const sourcePath = (path) => relative(root, path).split('\\').join('/');

function summarize(geometry, layers) {
	const resolved = geometry.valid.flatMap((valid, id) => (valid ? [id] : []));
	const layerCounts = new Map();
	for (const id of resolved) layerCounts.set(layers[id], (layerCounts.get(layers[id]) ?? 0) + 1);
	const neighbors = geometry.positions.map((_, id) =>
		getNeighbors(geometry, id, k).map((neighbor) => neighbor.index)
	);
	let directedLinks = 0;
	let sameLayerLinks = 0;
	let nullExpectedLinks = 0;
	for (const id of resolved) {
		const nearby = neighbors[id];
		directedLinks += nearby.length;
		sameLayerLinks += nearby.filter((other) => layers[other] === layers[id]).length;
		if (resolved.length > 1) {
			nullExpectedLinks +=
				(nearby.length * (layerCounts.get(layers[id]) - 1)) / (resolved.length - 1);
		}
	}
	return {
		neighbors,
		measurement: {
			validNeurons: resolved.length,
			unresolvedNeurons: geometry.valid.length - resolved.length,
			fingerprintDimensions: geometry.fingerprints[0]?.length ?? 0,
			pcaRank: geometry.rank,
			explainedVariance: geometry.explainedVariance,
			explainedVarianceByAxis: geometry.explainedVarianceByAxis,
			neighborRetention: geometry.neighborRetention,
			neighborCount: geometry.neighborCount,
			boundaryDegenerate: geometry.boundaryDegenerate,
			layerCounts: [...layerCounts]
				.sort(([a], [b]) => a - b)
				.map(([layer, count]) => ({ layer, validNeurons: count })),
			withinLayer: {
				directedLinks,
				sameLayerLinks,
				share: directedLinks ? sameLayerLinks / directedLinks : null,
				nullExpectedLinks,
				nullExpectedShare: directedLinks ? nullExpectedLinks / directedLinks : null
			}
		}
	};
}

function compareNeighborhoods(activation, effect, activationNeighbors, effectNeighbors) {
	const common = activation.valid.flatMap((valid, id) => (valid && effect.valid[id] ? [id] : []));
	const activationCount = activation.valid.filter(Boolean).length;
	const effectCount = effect.valid.filter(Boolean).length;
	let comparedNeurons = 0;
	let directedActivationNeighbors = 0;
	let sharedNeighborLinks = 0;
	let nullExpectedSharedLinks = 0;
	for (const id of common) {
		const a = activationNeighbors[id];
		const b = new Set(effectNeighbors[id]);
		if (!a.length || !b.size) continue;
		comparedNeurons++;
		directedActivationNeighbors += a.length;
		sharedNeighborLinks += a.filter((other) => b.has(other)).length;
		// Each map draws independently from its own resolved-neuron pool, excluding
		// the focal neuron. Only common valid identities can enter the intersection.
		nullExpectedSharedLinks +=
			(common.length - 1) * (a.length / (activationCount - 1)) * (b.size / (effectCount - 1));
	}
	return {
		commonValidNeurons: common.length,
		comparedNeurons,
		directedActivationNeighbors,
		sharedNeighborLinks,
		share: directedActivationNeighbors ? sharedNeighborLinks / directedActivationNeighbors : null,
		nullExpectedSharedLinks,
		nullExpectedShare: directedActivationNeighbors
			? nullExpectedSharedLinks / directedActivationNeighbors
			: null
	};
}

function analyzeCapture({ seed, path, bytes, atlas, checkpoint, metrics, saved }) {
	assert.equal(atlas.seed, seed);
	assert.equal(checkpoint.architecture, 'binding-transformer-v1');
	assert.equal(checkpoint.step, atlas.step);
	assert.equal(atlas.neurons.length, NEURON_COUNT);
	assert.equal(atlas.activationFingerprints.length, NEURON_COUNT);
	const hasEffects = (atlas.effectFingerprints?.length ?? 0) > 0;
	assert.equal(atlas.intervention, hasEffects ? 'zero-all-token-positions' : 'not-measured');
	if (hasEffects) assert.equal(atlas.effectFingerprints.length, NEURON_COUNT);
	atlas.neurons.forEach((neuron, id) => {
		assert.equal(neuron.id, id);
		assert.equal(neuron.layer, Math.floor(id / MODEL_CONFIG.hidden));
		assert.equal(neuron.channel, id % MODEL_CONFIG.hidden);
	});
	atlas.examples.forEach((example) => assert.equal(example.split, 'calibration'));
	atlas.activationFingerprints.forEach((row) =>
		assert.equal(row.length, atlas.examples.length * MODEL_CONFIG.sequenceLength)
	);
	atlas.effectFingerprints?.forEach((row) =>
		assert.equal(row.length, atlas.examples.length * (MODEL_CONFIG.vocabulary.length - 6))
	);
	assert(metrics, 'Atlas needs an evaluation at its exact checkpoint');
	assert.equal(metrics.step, atlas.step);
	assert(saved?.activation, 'Atlas needs its recorded activation geometry diagnostics');
	if (hasEffects) assert(saved.effect, 'Atlas needs its recorded effect geometry diagnostics');
	const activation = buildGeometry(atlas.activationFingerprints, {
		kind: 'activation',
		neighbors: k
	});
	const effect = hasEffects
		? buildGeometry(atlas.effectFingerprints, {
				kind: 'effect',
				neighbors: k,
				normalizeEffects: true
			})
		: null;
	const layers = atlas.neurons.map((neuron) => neuron.layer);
	for (const [kind, geometry] of [
		['activation', activation],
		['effect', effect]
	]) {
		if (!geometry) continue;
		for (const key of ['explainedVariance', 'neighborRetention']) {
			assert.equal(typeof saved[kind][key], 'number');
			assert(
				Math.abs(saved[kind][key] - geometry[key]) <= 1e-10,
				`${sourcePath(path)} ${kind} ${key} differs from the saved map`
			);
		}
		assert.equal(geometry.neighborCount, k);
	}
	const a = summarize(activation, layers);
	const e = effect ? summarize(effect, layers) : null;
	return {
		seed,
		checkpoint: atlas.step,
		k,
		source: sourcePath(path),
		sourceSha256: sha256(bytes),
		atlasCapturedAt: atlas.capturedAt,
		calibrationExamples: atlas.examples.length,
		evaluation: {
			step: metrics.step,
			examples: metrics.validationExamples,
			accuracy: metrics.accuracy,
			answerLoss: metrics.validationLoss,
			backend: metrics.backend
		},
		recordedPcaDiagnosticsMatched: true,
		activation: a.measurement,
		effect: e?.measurement ?? null,
		overlap: effect && e ? compareNeighborhoods(activation, effect, a.neighbors, e.neighbors) : null
	};
}

const runs = [];
const history = [];
for (const seed of [42, 7]) {
	const path = join(input, `binding-seed-${seed}.json`);
	const bytes = await readFile(path);
	const run = JSON.parse(bytes.toString('utf8'));
	assert.equal(run.seed, seed);
	runs.push(
		analyzeCapture({
			seed,
			path,
			bytes,
			atlas: run.atlas,
			checkpoint: run.checkpoint,
			metrics: run.metrics.find((entry) => entry.step === run.atlas.step),
			saved: run.snapshots.find((entry) => entry.step === run.atlas.step)
		})
	);
	for (const step of seed === 42 ? [500] : [500, 1500]) {
		const rawPath = join(input, `binding-seed-${seed}-${step}-raw.json`);
		const rawBytes = await readFile(rawPath);
		const raw = JSON.parse(rawBytes.toString('utf8'));
		assert.equal(raw.initialized.seed, seed);
		assert.equal(raw.atlas.step, step);
		const metrics = raw.trajectory.find((entry) => entry.step === step);
		const recorded = run.metrics.find((entry) => entry.step === step);
		assert.equal(metrics?.accuracy, recorded?.accuracy);
		assert.equal(metrics?.validationLoss, recorded?.validationLoss);
		history.push(
			analyzeCapture({
				seed,
				path: rawPath,
				bytes: rawBytes,
				atlas: raw.atlas,
				checkpoint: raw.checkpoint,
				metrics,
				saved: run.snapshots.find((entry) => entry.step === step)
			})
		);
	}
}

const implementationFiles = [
	'scripts/compare-fingerprints.mjs',
	'src/lib/lab/geometry.ts',
	'src/lib/lab/protocol.ts'
];
const implementation = await Promise.all(
	implementationFiles.map(async (path) => ({
		path,
		sha256: sha256(await readFile(join(root, path)))
	}))
);
const result = {
	version: 1,
	analysis: 'fingerprint-comparison-v1',
	k,
	runtime: { node: process.version },
	implementation,
	methods: {
		activation:
			'Each neuron row concatenates post-ReLU activations over the same calibration prompts and token positions. Center each row across probes and divide by its L2 norm. Preserve pre-normalization RMS magnitude separately.',
		effect:
			'Each row concatenates lesion-minus-intact answer probabilities over the same calibration prompts and eight answer values. L2 normalize each row without centering. The lesion zeros one MLP unit at all token positions.',
		validity:
			'Exclude rows with L2 norm at most 1e-12. Activation norms are computed after centering. The two maps can have different valid-neuron pools.',
		neighbors:
			'Choose six nearest Euclidean neighbors in the original normalized fingerprint space, excluding the focal neuron and invalid rows. Break exact distance ties by ascending neuron ID. Counts here are directed selections, not the undirected union of displayed edges.',
		projection:
			'PCA of the centered neuron cloud. Explained variance and numerical rank use geometry.ts. Neighbor retention is the fraction of projected top-six neighbors within the original-space sixth-neighbor radius, with the same tie tolerance as the lab: 1e-9 times that neuron’s largest original distance.',
		withinLayerShare:
			'Sum selected neighbor pairs whose recorded layer IDs match, divided by all selected directed neighbor pairs.',
		withinLayerNull:
			'Hold each focal neuron, its layer, the valid pool, and its outgoing neighbor count fixed. Draw neighbors uniformly from all other valid neurons. The focal same-layer probability is (valid neurons in its layer minus one)/(all valid neurons minus one). Weight expectations by outgoing neighbor count.',
		overlap:
			'For each focal neuron valid in both maps with nonempty neighborhoods, count identities shared by its activation and effect top-six sets. Divide the total intersection count by the total number of activation-neighbor selections across those focal neurons. Candidate neighbors stay in each map’s own valid pool.',
		overlapNull:
			'Draw the two neighbor sets independently and uniformly from their respective valid pools, excluding the focal neuron. Expected intersection size is (common-valid count minus one) times kA/(activation-valid count minus one) times kE/(effect-valid count minus one); aggregate with the same denominator as observed overlap.',
		verification:
			'Recomputed explained variance and neighbor retention must match the saved step-matched maps within 1e-10. Temporal frame alignment is omitted because orthogonal transforms preserve projected distances.',
		limits:
			'Descriptive measurements of selected checkpoints from two recorded seeds, not significance tests, novel algorithms, evidence of semantic concepts, or proof of recovery value. Uniform nulls do not control for activation strength, feature statistics, or training dependence. Missing raw effect fingerprints produce null effect and overlap measurements.'
	},
	runs,
	history
};
await mkdir(dirname(output), { recursive: true });
await writeFile(output, `${JSON.stringify(result, null, 2)}\n`);
console.log(`Wrote ${sourcePath(output)}`);
console.table(
	[...runs, ...history].map((run) => ({
		seed: run.seed,
		step: run.checkpoint,
		activationRetention: run.activation.neighborRetention,
		effectRetention: run.effect?.neighborRetention,
		activationWithinLayer: run.activation.withinLayer.share,
		effectWithinLayer: run.effect?.withinLayer.share,
		neighborhoodOverlap: run.overlap?.share,
		overlapUniformNull: run.overlap?.nullExpectedShare
	}))
);
