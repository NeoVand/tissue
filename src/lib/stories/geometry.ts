import { alignPoints } from '../lab/geometry';
import type { StoryAtlas } from './protocol';

export type StoryPoint = [number, number, number];
export interface StoryNeighbor {
	index: number;
	distance: number;
	/** Signed cosine of the centered, unit-length activation fingerprints. */
	similarity: number;
}
export interface StoryGeometryResult {
	version: 1;
	modelId: string;
	corpusId: string;
	step: number;
	/** Includes model, corpus, feature order and architecture; never includes training step. */
	identity: string;
	unitCount: number;
	dimensions: number;
	layerWidth: number;
	positions: StoryPoint[];
	valid: boolean[];
	/** Population standard deviation across the fixed probe coordinates, before normalization. */
	magnitudes: number[];
	rmsFloor: number;
	edges: [number, number][];
	edgeScope: 'audit-sample';
	explainedVariance: number;
	explainedVarianceByAxis: [number, number, number];
	/** Approximate PCA quality, not an exact numerical rank claim. */
	pca: {
		method: 'feature-covariance-block-power';
		iterations: number;
		maxIterations: number;
		tolerance: number;
		converged: boolean;
		/** Largest top-four eigenpair residual divided by covariance trace. */
		relativeResidual: number;
		eigenvalues: number[];
		totalVariance: number;
		boundaryUncertain: boolean;
	};
	audit: {
		schedule: 'fixed-seed-uniform-without-replacement-v1';
		seed: number;
		requested: number;
		planned: number;
		/** IDs are chosen before consulting activation values; unresolved IDs are not replaced. */
		ids: number[];
		validPopulation: number;
		resolvedFocals: number;
		scoredFocals: number;
		k: number;
		neighborRetention: number | null;
		perFocal: { id: number; neighborCount: number; retention: number | null }[];
	};
	alignment: { applied: boolean; rmsDisplacement: number | null };
}

/** Worker-local O(NP) state; deliberately absent from the UI result. */
export interface StoryGeometryIndex {
	unitCount: number;
	dimensions: number;
	layerWidth: number;
	normalized: Float64Array;
	valid: boolean[];
}
export interface StoryGeometryOptions {
	previous?: StoryGeometryResult;
	auditSize?: number;
	k?: number;
	rmsFloor?: number;
	maxIterations?: number;
	tolerance?: number;
}

const AUDIT_SEED = 0x51a7c0de;
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
function randomGenerator(seed: number) {
	let state = seed >>> 0;
	return () => {
		state += 0x6d2b79f5;
		let t = state;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

/** Activation-independent schedule, stable across all checkpoints of the same width. */
export function storyAuditIds(unitCount: number, requested = 128): number[] {
	if (
		!Number.isInteger(unitCount) ||
		unitCount < 1 ||
		!Number.isInteger(requested) ||
		requested < 1
	)
		throw new Error('Invalid story audit size');
	const ids = Array.from({ length: unitCount }, (_, i) => i);
	const random = randomGenerator(AUDIT_SEED);
	const size = Math.min(unitCount, requested);
	for (let i = 0; i < size; i++) {
		const j = i + Math.floor(random() * (unitCount - i));
		[ids[i], ids[j]] = [ids[j], ids[i]];
	}
	return ids.slice(0, size).sort((a, b) => a - b);
}

function insertNeighbor(neighbors: StoryNeighbor[], neighbor: StoryNeighbor, k: number) {
	let i = neighbors.length;
	while (
		i > 0 &&
		(neighbor.distance < neighbors[i - 1].distance ||
			(neighbor.distance === neighbors[i - 1].distance && neighbor.index < neighbors[i - 1].index))
	)
		i--;
	if (i < k) {
		neighbors.splice(i, 0, neighbor);
		if (neighbors.length > k) neighbors.pop();
	}
}

/** Exact original-space scan. No pairwise distance matrix is retained or approximated. */
export function storyNearestNeighbors(
	index: StoryGeometryIndex,
	id: number,
	k = 6,
	layer?: number
): StoryNeighbor[] {
	if (
		!Number.isInteger(id) ||
		id < 0 ||
		id >= index.unitCount ||
		!Number.isInteger(k) ||
		k < 1 ||
		k > 128
	)
		throw new Error('Invalid story neighbor query');
	if (
		layer !== undefined &&
		(!Number.isInteger(layer) || layer < 0 || layer >= index.unitCount / index.layerWidth)
	)
		throw new Error('Invalid story neighbor layer');
	if (!index.valid[id]) return [];
	const neighbors: StoryNeighbor[] = [];
	const { dimensions: p, normalized: rows } = index;
	for (let j = 0; j < index.unitCount; j++) {
		if (
			j === id ||
			!index.valid[j] ||
			(layer !== undefined && Math.floor(j / index.layerWidth) !== layer)
		)
			continue;
		let squared = 0;
		for (let d = 0; d < p; d++) {
			const delta = rows[id * p + d] - rows[j * p + d];
			squared += delta * delta;
		}
		insertNeighbor(
			neighbors,
			{ index: j, distance: Math.sqrt(squared), similarity: clamp(1 - squared / 2, -1, 1) },
			k
		);
	}
	return neighbors;
}

function projectedNeighbors(positions: StoryPoint[], valid: boolean[], id: number, k: number) {
	const neighbors: StoryNeighbor[] = [];
	for (let j = 0; j < positions.length; j++) {
		if (j === id || !valid[j]) continue;
		const distance = Math.hypot(...positions[id].map((v, d) => v - positions[j][d]));
		insertNeighbor(neighbors, { index: j, distance, similarity: 0 }, k);
	}
	return neighbors;
}

// Tiny symmetric Jacobi solve used only inside the (at most eight-dimensional) Ritz problem.
function symmetricEigen(matrix: Float64Array, n: number) {
	const a = matrix.slice();
	const vectors = new Float64Array(n * n);
	for (let i = 0; i < n; i++) vectors[i * n + i] = 1;
	for (let sweep = 0; sweep < 80; sweep++) {
		let changed = false;
		for (let p = 0; p < n; p++)
			for (let q = p + 1; q < n; q++) {
				const apq = a[p * n + q];
				if (
					Math.abs(apq) <=
					1e-15 * Math.max(1e-30, Math.abs(a[p * n + p]) + Math.abs(a[q * n + q]))
				)
					continue;
				changed = true;
				const theta = (a[q * n + q] - a[p * n + p]) / (2 * apq);
				const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(1 + theta * theta));
				const c = 1 / Math.sqrt(1 + t * t),
					s = t * c;
				const app = a[p * n + p],
					aqq = a[q * n + q];
				a[p * n + p] = app - t * apq;
				a[q * n + q] = aqq + t * apq;
				a[p * n + q] = a[q * n + p] = 0;
				for (let r = 0; r < n; r++) {
					if (r !== p && r !== q) {
						const arp = a[r * n + p],
							arq = a[r * n + q];
						a[r * n + p] = a[p * n + r] = c * arp - s * arq;
						a[r * n + q] = a[q * n + r] = s * arp + c * arq;
					}
					const vrp = vectors[r * n + p],
						vrq = vectors[r * n + q];
					vectors[r * n + p] = c * vrp - s * vrq;
					vectors[r * n + q] = s * vrp + c * vrq;
				}
			}
		if (!changed) break;
	}
	const order = Array.from({ length: n }, (_, i) => i).sort(
		(aIndex, bIndex) => a[bIndex * n + bIndex] - a[aIndex * n + aIndex]
	);
	return {
		values: order.map((i) => Math.max(0, a[i * n + i])),
		vectors: order.map((i) => Float64Array.from({ length: n }, (_, r) => vectors[r * n + i]))
	};
}

function multiply(matrix: Float64Array, vector: Float64Array, n: number) {
	const out = new Float64Array(n);
	for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) out[r] += matrix[r * n + c] * vector[c];
	return out;
}
function dot(a: Float64Array, b: Float64Array) {
	let sum = 0;
	for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
	return sum;
}
function orthonormalize(vectors: Float64Array[], n: number) {
	const basis: Float64Array[] = [];
	for (const input of vectors) {
		let vector = input.slice();
		let norm = 0;
		for (let attempt = -1; attempt < n; attempt++) {
			if (attempt >= 0) {
				vector = new Float64Array(n);
				vector[attempt] = 1;
			}
			for (let pass = 0; pass < 2; pass++)
				for (const previous of basis) {
					const overlap = dot(vector, previous);
					for (let i = 0; i < n; i++) vector[i] -= overlap * previous[i];
				}
			norm = Math.sqrt(dot(vector, vector));
			if (norm > 1e-12) break;
		}
		if (norm <= 1e-12) break;
		for (let i = 0; i < n; i++) vector[i] /= norm;
		basis.push(vector);
	}
	return basis;
}

function leadingEigenspace(
	covariance: Float64Array,
	p: number,
	trace: number,
	maxIterations: number,
	tolerance: number
) {
	const size = Math.min(8, p);
	const random = randomGenerator(0x70636131);
	let basis = orthonormalize(
		Array.from({ length: size }, () => Float64Array.from({ length: p }, () => random() - 0.5)),
		p
	);
	let eigenvalues = Array.from({ length: size }, () => 0);
	let relativeResidual = 0;
	let iterations = 0;
	if (trace > 0)
		for (iterations = 1; iterations <= maxIterations; iterations++) {
			// Scaling makes rank-completion thresholds independent of the number of units.
			basis = orthonormalize(
				basis.map((vector) => multiply(covariance, vector, p).map((v) => v / trace)),
				p
			);
			const products = basis.map((vector) => multiply(covariance, vector, p));
			const ritz = new Float64Array(size * size);
			for (let i = 0; i < size; i++)
				for (let j = i; j < size; j++)
					ritz[i * size + j] = ritz[j * size + i] = dot(basis[i], products[j]);
			const eig = symmetricEigen(ritz, size);
			eigenvalues = eig.values;
			basis = eig.vectors.map((coefficients) =>
				Float64Array.from({ length: p }, (_, d) =>
					coefficients.reduce((sum, value, i) => sum + value * basis[i][d], 0)
				)
			);
			relativeResidual = 0;
			for (let i = 0; i < Math.min(4, size); i++) {
				const cv = multiply(covariance, basis[i], p);
				let residual = 0;
				for (let d = 0; d < p; d++) residual += (cv[d] - eigenvalues[i] * basis[i][d]) ** 2;
				relativeResidual = Math.max(relativeResidual, Math.sqrt(residual) / trace);
			}
			if (relativeResidual <= tolerance) break;
		}
	for (const vector of basis) {
		let pivot = 0;
		for (let i = 1; i < p; i++) if (Math.abs(vector[i]) > Math.abs(vector[pivot])) pivot = i;
		if (vector[pivot] < 0) for (let i = 0; i < p; i++) vector[i] *= -1;
	}
	return {
		basis,
		eigenvalues: eigenvalues.slice(0, 4),
		relativeResidual,
		iterations: Math.min(iterations, maxIterations)
	};
}

export function prepareStoryGeometryIndex(
	atlas: StoryAtlas,
	rmsFloor = 1e-8
): { index: StoryGeometryIndex; magnitudes: number[] } {
	const { unitCount: n, dimensions: p, fingerprints: raw } = atlas;
	if (
		!Number.isInteger(n) ||
		n < 1 ||
		n > 100_000 ||
		!Number.isInteger(p) ||
		p < 1 ||
		p > 512 ||
		!(raw instanceof Float32Array) ||
		raw.length !== n * p ||
		!Number.isInteger(atlas.config?.hidden) ||
		atlas.config.hidden < 1 ||
		!Number.isInteger(atlas.config.layers) ||
		atlas.config.layers < 1 ||
		n !== atlas.config.layers * atlas.config.hidden ||
		!Array.isArray(atlas.positions) ||
		!Array.isArray(atlas.examples) ||
		atlas.positions.length * atlas.examples.length !== p ||
		atlas.positions.some((v) => !Number.isInteger(v) || v < 0) ||
		atlas.examples.some((v) => typeof v !== 'string') ||
		typeof atlas.modelId !== 'string' ||
		atlas.modelId.length === 0 ||
		typeof atlas.corpusId !== 'string' ||
		atlas.corpusId.length === 0 ||
		!Number.isSafeInteger(atlas.step) ||
		atlas.step < 0
	)
		throw new Error('Invalid story atlas shape or identity');
	if (!Number.isFinite(rmsFloor) || rmsFloor <= 0)
		throw new Error('Invalid story geometry RMS floor');
	const normalized = new Float64Array(n * p),
		valid: boolean[] = Array(n).fill(false),
		magnitudes: number[] = Array(n).fill(0);
	for (let i = 0; i < n; i++) {
		let average = 0;
		for (let d = 0; d < p; d++) {
			const value = raw[i * p + d];
			if (!Number.isFinite(value))
				throw new Error(`Nonfinite story activation at unit ${i}, coordinate ${d}`);
			average += value / p;
		}
		let squared = 0;
		for (let d = 0; d < p; d++) squared += (raw[i * p + d] - average) ** 2;
		magnitudes[i] = Math.sqrt(squared / p);
		if (magnitudes[i] <= rmsFloor) continue;
		valid[i] = true;
		const norm = Math.sqrt(squared);
		for (let d = 0; d < p; d++) {
			normalized[i * p + d] = (raw[i * p + d] - average) / norm;
		}
	}
	return {
		index: { unitCount: n, dimensions: p, layerWidth: atlas.config.hidden, normalized, valid },
		magnitudes
	};
}

export function buildStoryGeometry(
	atlas: StoryAtlas,
	options: StoryGeometryOptions = {}
): { geometry: StoryGeometryResult; index: StoryGeometryIndex } {
	const rmsFloor = options.rmsFloor ?? 1e-8,
		maxIterations = options.maxIterations ?? 256,
		tolerance = options.tolerance ?? 1e-8;
	const k = options.k ?? 6,
		auditSize = options.auditSize ?? 128;
	if (
		!Number.isInteger(maxIterations) ||
		maxIterations < 1 ||
		maxIterations > 2048 ||
		!Number.isFinite(tolerance) ||
		tolerance <= 0 ||
		tolerance >= 1 ||
		!Number.isInteger(k) ||
		k < 1 ||
		k > 128 ||
		!Number.isInteger(auditSize) ||
		auditSize < 1 ||
		auditSize > 1024
	)
		throw new Error('Invalid story geometry options');
	const { index, magnitudes } = prepareStoryGeometryIndex(atlas, rmsFloor);
	const { unitCount: n, dimensions: p, normalized, valid } = index;
	const validPopulation = valid.filter(Boolean).length;
	const mean = new Float64Array(p);
	for (let i = 0; i < n; i++)
		if (valid[i]) for (let d = 0; d < p; d++) mean[d] += normalized[i * p + d];
	for (let d = 0; d < p; d++) mean[d] /= Math.max(1, validPopulation);
	const covariance = new Float64Array(p * p),
		centered = new Float64Array(p);
	for (let i = 0; i < n; i++)
		if (valid[i]) {
			for (let d = 0; d < p; d++) centered[d] = normalized[i * p + d] - mean[d];
			for (let d = 0; d < p; d++)
				for (let e = d; e < p; e++) covariance[d * p + e] += centered[d] * centered[e];
		}
	let trace = 0;
	for (let d = 0; d < p; d++) {
		trace += covariance[d * p + d];
		for (let e = d + 1; e < p; e++) covariance[e * p + d] = covariance[d * p + e];
	}
	// A cloud of identical directions can have roundoff variance after centering.
	if (trace <= Math.max(1, validPopulation) * 1e-24) {
		trace = 0;
		covariance.fill(0);
	}
	const fit = leadingEigenspace(covariance, p, trace, maxIterations, tolerance);
	let positions: StoryPoint[] = Array.from({ length: n }, () => [0, 0, 0]);
	const axisVariance: [number, number, number] = [0, 0, 0];
	if (trace > 0)
		for (let i = 0; i < n; i++)
			if (valid[i])
				for (let axis = 0; axis < Math.min(3, p); axis++) {
					let score = 0;
					for (let d = 0; d < p; d++)
						score += (normalized[i * p + d] - mean[d]) * fit.basis[axis][d];
					positions[i][axis] = score;
					axisVariance[axis] += (score * score) / trace;
				}
	const identity = JSON.stringify([
		atlas.modelId,
		atlas.corpusId,
		atlas.config.layers,
		atlas.config.hidden,
		p,
		atlas.positions,
		atlas.examples,
		rmsFloor
	]);
	let alignment: StoryGeometryResult['alignment'] = { applied: false, rmsDisplacement: null };
	const previous = options.previous;
	if (previous?.identity === identity && previous.positions.length === n) {
		const aligned = alignPoints(
			positions,
			previous.positions,
			valid.map((v, i) => v && previous.valid[i])
		);
		positions = aligned.positions;
		alignment = { applied: aligned.applied, rmsDisplacement: aligned.rmsDisplacement };
	}
	const ids = storyAuditIds(n, auditSize),
		edgeSet = new Set<string>();
	const perFocal = ids.map((id) => {
		const neighbors = storyNearestNeighbors(index, id, k);
		for (const neighbor of neighbors)
			edgeSet.add([Math.min(id, neighbor.index), Math.max(id, neighbor.index)].join(','));
		if (!neighbors.length || trace === 0)
			return { id, neighborCount: neighbors.length, retention: null };
		const projected = projectedNeighbors(positions, valid, id, neighbors.length);
		const radius = projected.at(-1)!.distance;
		// Boundary ties are counted as retained rather than imposing an arbitrary 3D ID tie-break.
		const slack = Math.max(1e-12, radius * 1e-9);
		const retained = neighbors.filter(
			(neighbor) =>
				Math.hypot(...positions[id].map((v, d) => v - positions[neighbor.index][d])) <=
				radius + slack
		).length;
		return { id, neighborCount: neighbors.length, retention: retained / neighbors.length };
	});
	const scored = perFocal.filter((row) => row.retention !== null);
	const geometry: StoryGeometryResult = {
		version: 1,
		modelId: atlas.modelId,
		corpusId: atlas.corpusId,
		step: atlas.step,
		identity,
		unitCount: n,
		dimensions: p,
		layerWidth: atlas.config.hidden,
		positions,
		valid,
		magnitudes,
		rmsFloor,
		edges: [...edgeSet].map((key) => key.split(',').map(Number) as [number, number]),
		edgeScope: 'audit-sample',
		explainedVariance: clamp(
			axisVariance.reduce((a, b) => a + b, 0),
			0,
			1
		),
		explainedVarianceByAxis: axisVariance,
		pca: {
			method: 'feature-covariance-block-power',
			iterations: fit.iterations,
			maxIterations,
			tolerance,
			converged: fit.relativeResidual <= tolerance,
			relativeResidual: fit.relativeResidual,
			eigenvalues: fit.eigenvalues,
			totalVariance: trace,
			boundaryUncertain:
				trace === 0 ||
				((fit.eigenvalues[2] ?? 0) - (fit.eigenvalues[3] ?? 0)) / trace <=
					Math.max(1e-8, fit.relativeResidual * 2)
		},
		audit: {
			schedule: 'fixed-seed-uniform-without-replacement-v1',
			seed: AUDIT_SEED,
			requested: auditSize,
			planned: ids.length,
			ids,
			validPopulation,
			resolvedFocals: ids.filter((id) => valid[id]).length,
			scoredFocals: scored.length,
			k,
			neighborRetention: scored.length
				? scored.reduce((sum, row) => sum + row.retention!, 0) / scored.length
				: null,
			perFocal
		},
		alignment
	};
	return { geometry, index };
}
