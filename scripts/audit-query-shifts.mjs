/** Independent arithmetic/provenance audit of recorded query studies; no model execution.
 * node scripts/audit-query-shifts.mjs
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';

const methods = ['query-effect', 'full-effect', 'query-activation', 'outgoing-weight'];
const floor = 1e-8;
const read = async (path) => JSON.parse(await readFile(path, 'utf8'));
const hash = (value) => createHash('sha256').update(value).digest('hex');
const rms = (row) => Math.sqrt(row.reduce((sum, x) => sum + x * x, 0) / row.length);
const mean = (values) => values.reduce((sum, x) => sum + x, 0) / values.length;
const close = (a, b) => assert(Math.abs(a - b) < 1e-12, `${a} differs from ${b}`);
const cosine = (a, b) => {
	if (rms(a) <= floor || rms(b) <= floor) return 0;
	return (
		a.reduce((sum, x, i) => sum + x * b[i], 0) /
		Math.sqrt(a.reduce((sum, x) => sum + x * x, 0) * b.reduce((sum, x) => sum + x * x, 0))
	);
};
function contrast(rows, coordinates) {
	return rows.map((row) =>
		Array.from({ length: 16 }, (_, group) => {
			const values = [0, 1, 2].map((query) =>
				row.slice((3 * group + query) * coordinates, (3 * group + query + 1) * coordinates)
			);
			return [
				values[0].map((a, i) => (values[1][i] - a) / Math.sqrt(2)),
				values[0].map((a, i) => (2 * values[2][i] - a - values[1][i]) / Math.sqrt(6))
			].flat();
		}).flat()
	);
}
function distribution(values) {
	const sorted = [...values].sort((a, b) => a - b);
	const quantile = (p) => {
		const index = (sorted.length - 1) * p;
		const low = Math.floor(index),
			high = Math.ceil(index);
		return sorted[low] + (sorted[high] - sorted[low]) * (index - low);
	};
	return {
		count: values.length,
		median: quantile(0.5),
		p90: quantile(0.9),
		max: sorted.at(-1),
		mean: mean(values)
	};
}

const runs = [];
for (const seed of [42, 7]) {
	const rawPath = `static/experiments/query-shifts-seed-${seed}.json`;
	const analysisPath = `static/experiments/query-shifts-seed-${seed}-analysis.json`;
	const record = await read(rawPath);
	const report = await read(analysisPath);
	const m = record.measurement,
		a = report.analysis;
	const summary = (await read('static/experiments/query-shifts-summary.json')).runs.find(
		(run) => run.seed === seed
	);
	assert.deepEqual(summary.summary, a.overall);
	assert.deepEqual(summary.layers, a.layers);
	for (const [path, expected] of Object.entries({
		...record.provenance.sourceHashes,
		...report.sourceHashes
	})) {
		assert.equal(hash(await readFile(path)), expected, `Source hash: ${path}`);
	}
	assert.equal(
		hash(await readFile(record.provenance.checkpointFile)),
		record.provenance.checkpointFileHash
	);
	assert.equal(
		hash(JSON.stringify((await read(record.provenance.checkpointFile)).checkpoint)),
		m.checkpointHash
	);
	assert.equal(hash(JSON.stringify(m)), report.measurementHash);
	assert.equal(m.checkpointHash, a.checkpointHash);
	assert.equal(hash(await readFile('docs/query-shifts-design.md')), record.provenance.designHash);
	assert.equal(a.design.neighbors, 6);
	assert.equal(a.design.candidatePool, 32);
	assert.equal(a.design.rmsFloor, floor);
	const vectors = {
		'query-effect': contrast(m.splits.calibration.effects, 8),
		'full-effect': m.splits.calibration.effects,
		'query-activation': contrast(m.splits.calibration.activations, 1),
		'outgoing-weight': m.outgoingWeights
	};
	const target = contrast(m.splits.test.effects, 8);
	const eligible = Array.from({ length: 256 }, (_, id) =>
		methods.every((method) => rms(vectors[method][id]) > floor)
	);
	const strengths = vectors['full-effect'].map(rms);
	const matchedRatios = [],
		selectedRatios = Object.fromEntries(methods.map((method) => [method, []]));
	const tiedBoundaries = Object.fromEntries(methods.map((method) => [method, 0]));
	const excluded = [];
	for (const unit of a.units) {
		const id = unit.id;
		assert.equal(unit.calibrationEligible, eligible[id]);
		if (!eligible[id]) {
			excluded.push({
				id,
				layer: unit.layer,
				methods: methods.filter((method) => rms(vectors[method][id]) <= floor),
				calibrationLastTokenAllZero: m.splits.calibration.activations[id].every(
					(value) => value === 0
				),
				calibrationLastTokenMin: Math.min(...m.splits.calibration.activations[id]),
				calibrationLastTokenMax: Math.max(...m.splits.calibration.activations[id]),
				calibrationQueryEffectRms: rms(vectors['query-effect'][id]),
				testQueryEffectRms: rms(target[id])
			});
			continue;
		}
		const candidates = a.units
			.filter((other) => other.id !== id && other.layer === unit.layer && eligible[other.id])
			.map((other) => other.id)
			.sort(
				(x, y) =>
					Math.abs(Math.log(strengths[x] / strengths[id])) -
						Math.abs(Math.log(strengths[y] / strengths[id])) || x - y
			)
			.slice(0, 32);
		assert.deepEqual(unit.candidatePool, candidates);
		const expectedRandom = mean(candidates.map((other) => cosine(target[id], target[other])));
		close(unit.randomScore, expectedRandom);
		const ratio = (other) =>
			Math.max(strengths[id], strengths[other]) / Math.min(strengths[id], strengths[other]);
		matchedRatios.push(...candidates.map(ratio));
		for (const method of methods) {
			const selection = unit.methods[method];
			assert.equal(selection.neighbors.length, 6);
			assert.equal(new Set(selection.neighbors).size, 6);
			assert(selection.neighbors.every((other) => candidates.includes(other)));
			const scores = candidates
				.map((other) => cosine(vectors[method][id], vectors[method][other]))
				.sort((x, y) => y - x);
			assert(
				selection.neighbors.every(
					(other) => cosine(vectors[method][id], vectors[method][other]) >= scores[5] - 1e-12
				)
			);
			if (Math.abs(scores[5] - scores[6]) <= 1e-12) tiedBoundaries[method]++;
			const score = mean(selection.neighbors.map((other) => cosine(target[id], target[other])));
			close(selection.score, score);
			close(selection.delta, score - expectedRandom);
			const shuffled = a.shuffledTestIdentities;
			const shuffledScore = mean(
				selection.neighbors.map((other) => cosine(target[shuffled[id]], target[shuffled[other]]))
			);
			const shuffledRandom = mean(
				candidates.map((other) => cosine(target[shuffled[id]], target[shuffled[other]]))
			);
			close(selection.shuffledScore, shuffledScore);
			close(selection.shuffledDelta, shuffledScore - shuffledRandom);
			selectedRatios[method].push(...selection.neighbors.map(ratio));
		}
	}
	assert.equal(new Set(a.shuffledTestIdentities).size, 256);
	a.shuffledTestIdentities.forEach((donor, id) => {
		assert.equal(a.units[donor].layer, a.units[id].layer);
		assert.equal(rms(target[donor]) > floor, rms(target[id]) > floor);
	});
	for (const layer of [null, 0, 1]) {
		const units = a.units.filter(
			(unit) => unit.status === 'scored' && (layer === null || unit.layer === layer)
		);
		const stored = layer === null ? a.overall : a.layers.find((item) => item.layer === layer);
		assert.equal(units.length, stored.scoredUnits);
		for (const method of methods) {
			close(
				mean(units.map((unit) => unit.methods[method].score)),
				stored.methods[method].meanScore
			);
			close(
				mean(units.map((unit) => unit.methods[method].delta)),
				stored.methods[method].meanDelta
			);
		}
	}
	runs.push({
		seed,
		step: m.step,
		rawArtifactHash: hash(await readFile(rawPath)),
		analysisArtifactHash: hash(await readFile(analysisPath)),
		allSourceCheckpointMeasurementHashesMatched: true,
		allPoolsSelectionsScoresAndSummariesMatched: true,
		matchingFoldRatio: {
			candidatePool: distribution(matchedRatios),
			methods: Object.fromEntries(
				methods.map((method) => [method, distribution(selectedRatios[method])])
			)
		},
		numericallyTiedSelectionBoundaries: tiedBoundaries,
		excluded,
		strongestExcludedFirstLayer: excluded
			.filter((unit) => unit.layer === 0 && unit.calibrationQueryEffectRms > floor)
			.sort((a, b) => b.calibrationQueryEffectRms - a.calibrationQueryEffectRms || a.id - b.id)
			.slice(0, 3),
		queryEffectRmsMinimumScored: Math.min(
			...a.units
				.filter((unit) => unit.status === 'scored')
				.map((unit) => unit.strengths.calibrationQueryEffectRms)
		),
		testQueryEffectRmsMinimumScored: Math.min(
			...a.units
				.filter((unit) => unit.status === 'scored')
				.map((unit) => unit.strengths.testQueryEffectRms)
		),
		candidatePoolSizes: [
			...new Set(
				a.units.filter((unit) => unit.status === 'scored').map((unit) => unit.candidatePoolSize)
			)
		]
	});
}
const result = {
	version: 1,
	auditScriptHash: hash(await readFile('scripts/audit-query-shifts.mjs')),
	methods:
		'Independent sums-of-squares/cosine arithmetic, no import of query-analysis.ts. All selected neighbors must lie within 1e-12 of the calibration top-six threshold. Fold ratio is larger/smaller raw calibration full-effect RMS, measured over directed pairs. Quantiles linearly interpolate sorted observations at (n-1)p. These post-result diagnostics do not change the locked analysis.',
	runs
};
await writeFile(
	'static/experiments/query-shifts-audit.json',
	`${JSON.stringify(result, null, 2)}\n`
);
console.log(
	JSON.stringify(
		runs.map(
			({
				seed,
				matchingFoldRatio,
				numericallyTiedSelectionBoundaries,
				excluded,
				queryEffectRmsMinimumScored,
				testQueryEffectRmsMinimumScored
			}) => ({
				seed,
				matchingFoldRatio,
				numericallyTiedSelectionBoundaries,
				excludedCount: excluded.length,
				queryEffectRmsMinimumScored,
				testQueryEffectRmsMinimumScored
			})
		),
		null,
		2
	)
);
