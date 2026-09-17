import { storyAuditIds, type StoryGeometryResult } from './geometry';
import type { StoryAtlas } from './protocol';

function check(condition: unknown, message: string): asserts condition {
	if (!condition) throw new Error(`Invalid story geometry: ${message}`);
}
const finite = (value: unknown, min = -Infinity, max = Infinity): value is number =>
	typeof value === 'number' && Number.isFinite(value) && value >= min && value <= max;
const integer = (value: unknown, min: number, max: number): value is number =>
	finite(value, min, max) && Number.isSafeInteger(value);
const close = (a: number, b: number, tolerance = 1e-8) =>
	Math.abs(a - b) <= tolerance * Math.max(1, Math.abs(a), Math.abs(b));

/**
 * Validate a report's complete schema and internal arithmetic against its raw atlas.
 * This does not rerun PCA/neighbor search or authenticate externally supplied evidence.
 */
export function validateStoryGeometry(
	value: unknown,
	atlas: StoryAtlas
): asserts value is StoryGeometryResult {
	check(value && typeof value === 'object', 'missing report');
	const g = value as StoryGeometryResult;
	const n = atlas.unitCount,
		p = atlas.dimensions;
	check(
		integer(n, 1, 100_000) &&
			integer(p, 1, 512) &&
			atlas.fingerprints instanceof Float32Array &&
			atlas.fingerprints.length === n * p,
		'invalid source atlas dimensions'
	);
	check(
		g.version === 1 &&
			g.modelId === atlas.modelId &&
			g.corpusId === atlas.corpusId &&
			g.step === atlas.step &&
			g.unitCount === n &&
			g.dimensions === p &&
			g.layerWidth === atlas.config.hidden,
		'atlas identity mismatch'
	);
	check(finite(g.rmsFloor, Number.MIN_VALUE), 'invalid RMS floor');
	const identity = JSON.stringify([
		atlas.modelId,
		atlas.corpusId,
		atlas.config.layers,
		atlas.config.hidden,
		p,
		atlas.positions,
		atlas.examples,
		g.rmsFloor
	]);
	check(
		g.identity === identity,
		'probe identity does not match model, corpus, feature order and RMS floor'
	);
	check(
		Array.isArray(g.positions) &&
			g.positions.length === n &&
			g.positions.every(
				(position) =>
					Array.isArray(position) && position.length === 3 && position.every((v) => finite(v))
			),
		'invalid position array'
	);
	check(
		Array.isArray(g.valid) && g.valid.length === n && g.valid.every((v) => typeof v === 'boolean'),
		'invalid resolution flags'
	);
	check(
		Array.isArray(g.magnitudes) &&
			g.magnitudes.length === n &&
			g.magnitudes.every((v) => finite(v, 0)),
		'invalid magnitude array'
	);
	let validCount = 0;
	const normalizedSums = new Float64Array(p);
	for (let i = 0; i < n; i++) {
		let mean = 0;
		for (let d = 0; d < p; d++) {
			const raw = atlas.fingerprints[i * p + d];
			check(finite(raw), `nonfinite source fingerprint at unit ${i}`);
			mean += raw / p;
		}
		let squared = 0;
		for (let d = 0; d < p; d++) squared += (atlas.fingerprints[i * p + d] - mean) ** 2;
		const rms = Math.sqrt(squared / p);
		check(
			Math.abs(g.magnitudes[i] - rms) <= Math.max(Number.MIN_VALUE, rms * 1e-10),
			`magnitude does not match raw fingerprint at unit ${i}`
		);
		check(g.valid[i] === rms > g.rmsFloor, `resolution flag disagrees with RMS floor at unit ${i}`);
		if (!g.valid[i]) continue;
		validCount++;
		const norm = Math.sqrt(squared);
		for (let d = 0; d < p; d++) normalizedSums[d] += (atlas.fingerprints[i * p + d] - mean) / norm;
	}

	const fit = g.pca;
	check(
		fit &&
			typeof fit === 'object' &&
			fit.method === 'feature-covariance-block-power' &&
			integer(fit.maxIterations, 1, 2048) &&
			integer(fit.iterations, 0, fit.maxIterations) &&
			finite(fit.tolerance, Number.MIN_VALUE) &&
			fit.tolerance < 1 &&
			finite(fit.relativeResidual, 0) &&
			finite(fit.totalVariance, 0) &&
			typeof fit.converged === 'boolean' &&
			typeof fit.boundaryUncertain === 'boolean',
		'incomplete PCA diagnostics'
	);
	check(
		Array.isArray(fit.eigenvalues) &&
			fit.eigenvalues.length === Math.min(4, p) &&
			fit.eigenvalues.every((v) => finite(v, 0)),
		'invalid PCA eigenvalues'
	);
	check(
		fit.eigenvalues.every(
			(v, i) => i === 0 || v <= fit.eigenvalues[i - 1] + 1e-10 * Math.max(1, fit.totalVariance)
		),
		'PCA eigenvalues are not descending'
	);
	check(
		fit.eigenvalues.reduce((sum, v) => sum + v, 0) <=
			fit.totalVariance + 1e-8 * Math.max(1, fit.totalVariance),
		'PCA spectrum exceeds total centered energy'
	);
	const expectedTrace = validCount
		? Math.max(0, validCount - normalizedSums.reduce((sum, v) => sum + v * v, 0) / validCount)
		: 0;
	// This O(NP) identity is independent of the covariance fit; cancellation near zero uses N-scaled slack.
	check(
		Math.abs(fit.totalVariance - expectedTrace) <= 1e-9 * Math.max(1, validCount),
		'PCA total centered energy disagrees with fingerprints'
	);
	check(
		fit.converged === fit.relativeResidual <= fit.tolerance,
		'PCA convergence flag disagrees with residual'
	);
	check(
		fit.converged || fit.iterations === fit.maxIterations,
		'unconverged PCA ended before iteration limit'
	);
	if (fit.totalVariance === 0)
		check(
			fit.iterations === 0 && fit.relativeResidual === 0 && fit.eigenvalues.every((v) => v === 0),
			'zero-variance PCA diagnostics are inconsistent'
		);
	else check(fit.iterations > 0, 'nonzero-variance PCA has no iterations');
	const boundaryUncertain =
		fit.totalVariance === 0 ||
		((fit.eigenvalues[2] ?? 0) - (fit.eigenvalues[3] ?? 0)) / fit.totalVariance <=
			Math.max(1e-8, fit.relativeResidual * 2);
	check(
		fit.boundaryUncertain === boundaryUncertain,
		'PCA boundary flag disagrees with spectrum/residual'
	);
	check(
		finite(g.explainedVariance, 0, 1) &&
			Array.isArray(g.explainedVarianceByAxis) &&
			g.explainedVarianceByAxis.length === 3 &&
			g.explainedVarianceByAxis.every((v) => finite(v, 0, 1 + 1e-8)),
		'invalid captured variance'
	);
	for (let axis = 0; axis < 3; axis++) {
		const expected = fit.totalVariance ? (fit.eigenvalues[axis] ?? 0) / fit.totalVariance : 0;
		check(
			close(g.explainedVarianceByAxis[axis], expected),
			`axis ${axis + 1} variance disagrees with PCA spectrum`
		);
	}
	check(
		close(
			g.explainedVariance,
			Math.min(
				1,
				g.explainedVarianceByAxis.reduce((sum, v) => sum + v, 0)
			)
		),
		'captured variance does not equal axis sum'
	);
	const center = [0, 0, 0];
	for (let i = 0; i < n; i++)
		if (g.valid[i]) for (let d = 0; d < 3; d++) center[d] += g.positions[i][d] / validCount;
	let projectedEnergy = 0;
	for (let i = 0; i < n; i++)
		if (g.valid[i])
			for (let d = 0; d < 3; d++) projectedEnergy += (g.positions[i][d] - center[d]) ** 2;
	check(
		close(projectedEnergy, fit.totalVariance * g.explainedVariance, 1e-7),
		'coordinate energy disagrees with captured variance'
	);

	const audit = g.audit;
	check(
		audit &&
			typeof audit === 'object' &&
			audit.schedule === 'fixed-seed-uniform-without-replacement-v1' &&
			audit.seed === 0x51a7c0de &&
			integer(audit.requested, 1, 1024) &&
			integer(audit.k, 1, 128),
		'invalid audit schedule'
	);
	const expectedIds = storyAuditIds(n, audit.requested);
	check(
		audit.planned === expectedIds.length &&
			Array.isArray(audit.ids) &&
			audit.ids.length === expectedIds.length &&
			audit.ids.every((id, i) => id === expectedIds[i]),
		'audit IDs differ from prespecified schedule'
	);
	check(
		audit.validPopulation === validCount &&
			audit.resolvedFocals === expectedIds.filter((id) => g.valid[id]).length,
		'audit resolution counts disagree with flags'
	);
	check(
		Array.isArray(audit.perFocal) && audit.perFocal.length === expectedIds.length,
		'invalid per-focal audit array'
	);
	let scored = 0,
		retentionSum = 0,
		neighborSelections = 0;
	for (let i = 0; i < expectedIds.length; i++) {
		const row = audit.perFocal[i],
			id = expectedIds[i];
		check(
			row && typeof row === 'object' && row.id === id,
			'per-focal IDs differ from audit schedule'
		);
		const neighborCount = g.valid[id] ? Math.min(audit.k, Math.max(0, validCount - 1)) : 0;
		check(row.neighborCount === neighborCount, `incorrect neighbor count for audit unit ${id}`);
		neighborSelections += neighborCount;
		if (neighborCount === 0 || fit.totalVariance === 0)
			check(row.retention === null, `unscorable audit unit ${id} has a retention score`);
		else {
			check(
				finite(row.retention, 0, 1) &&
					close(row.retention * neighborCount, Math.round(row.retention * neighborCount), 1e-10),
				`invalid retention fraction at audit unit ${id}`
			);
			retentionSum += row.retention;
			scored++;
		}
	}
	check(audit.scoredFocals === scored, 'audit scored-focal count disagrees with coverage');
	check(
		scored
			? finite(audit.neighborRetention, 0, 1) &&
					close(audit.neighborRetention, retentionSum / scored, 1e-12)
			: audit.neighborRetention === null,
		'audit mean retention disagrees with per-focal scores'
	);
	check(
		g.edgeScope === 'audit-sample' &&
			Array.isArray(g.edges) &&
			g.edges.length <= neighborSelections,
		'invalid sampled graph scope or size'
	);
	const ids = new Set(expectedIds),
		edgeKeys = new Set<string>(),
		degrees = new Map<number, number>();
	for (const edge of g.edges) {
		check(
			Array.isArray(edge) &&
				edge.length === 2 &&
				integer(edge[0], 0, n - 1) &&
				integer(edge[1], 0, n - 1) &&
				edge[0] < edge[1],
			'invalid canonical sampled edge'
		);
		const [a, b] = edge;
		check(
			g.valid[a] && g.valid[b] && (ids.has(a) || ids.has(b)),
			'edge is outside the resolved audit neighborhood'
		);
		const key = `${a},${b}`;
		check(!edgeKeys.has(key), 'duplicate sampled edge');
		edgeKeys.add(key);
		degrees.set(a, (degrees.get(a) ?? 0) + 1);
		degrees.set(b, (degrees.get(b) ?? 0) + 1);
	}
	for (const row of audit.perFocal)
		check(
			(degrees.get(row.id) ?? 0) >= row.neighborCount,
			`sampled graph omits neighbors of audit unit ${row.id}`
		);
	check(
		g.alignment &&
			typeof g.alignment === 'object' &&
			typeof g.alignment.applied === 'boolean' &&
			(g.alignment.applied
				? validCount >= 2 && finite(g.alignment.rmsDisplacement, 0)
				: g.alignment.rmsDisplacement === null),
		'invalid temporal alignment diagnostics'
	);
}
