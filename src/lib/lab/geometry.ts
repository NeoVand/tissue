/** Functional geometry. Distances here are measured; rendering scale belongs to the view. */
export type Point3 = [number, number, number];
export type GeometryKind = 'activation' | 'effect';

export interface GeometryOptions {
	kind: GeometryKind;
	previous?: Pick<GeometryResult, 'kind' | 'positions' | 'valid'>;
	neighbors?: number;
	/** Compare directions by default; false preserves effect strength in the metric. */
	normalizeEffects?: boolean;
}

export interface GeometryResult {
	kind: GeometryKind;
	positions: Point3[];
	/** Undirected union of ORIGINAL-space nearest-neighbor relations, not causal edges. */
	edges: [number, number][];
	magnitudes: number[];
	valid: boolean[];
	fingerprints: number[][];
	originalDistances: number[][];
	projectedDistances: number[][];
	neighborCount: number;
	/** Tie-aware fraction of projected neighbors that remain original-space neighbors. */
	neighborRetention: number | null;
	perNodeRetention: (number | null)[];
	explainedVariance: number;
	explainedVarianceByAxis: Point3;
	rank: number;
	/** A tied third/fourth eigenvalue makes the chosen 3D subspace non-unique. */
	boundaryDegenerate: boolean;
	alignment: { applied: boolean; rmsDisplacement: number | null };
}

export interface Neighbor {
	index: number;
	distance: number;
}

const EPS = 1e-12;
const ZERO = (): Point3 => [0, 0, 0];

function squaredDistance(a: readonly number[], b: readonly number[]): number {
	let result = 0;
	for (let i = 0; i < a.length; i++) result += (a[i] - b[i]) ** 2;
	return result;
}

function distances(rows: readonly number[][]): number[][] {
	const result = rows.map(() => Array<number>(rows.length).fill(0));
	for (let i = 0; i < rows.length; i++) {
		for (let j = i + 1; j < rows.length; j++) {
			const d = Math.sqrt(squaredDistance(rows[i], rows[j]));
			result[i][j] = d;
			result[j][i] = d;
		}
	}
	return result;
}

/** Jacobi diagonalization of a real symmetric matrix; eigenvectors are columns. */
function symmetricEigen(input: readonly number[][]): { values: number[]; vectors: number[][] } {
	const n = input.length;
	const a = input.map((row) => [...row]);
	const v: number[][] = Array.from({ length: n }, (_, i) =>
		Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))
	);
	let scale = 0;
	for (const row of a) for (const value of row) scale = Math.max(scale, Math.abs(value));
	if (scale === 0) return { values: Array<number>(n).fill(0), vectors: v };
	for (let sweep = 0; sweep < 40; sweep++) {
		let offDiagonal = 0;
		for (let p = 0; p < n; p++) {
			for (let q = p + 1; q < n; q++) offDiagonal += Math.abs(a[p][q]);
		}
		if (offDiagonal <= scale * n * EPS) break;
		const threshold = sweep < 4 ? (0.2 * offDiagonal) / (n * n) : scale * EPS;
		for (let p = 0; p < n; p++) {
			for (let q = p + 1; q < n; q++) {
				const apq = a[p][q];
				if (Math.abs(apq) <= threshold) continue;
				const tau = (a[q][q] - a[p][p]) / (2 * apq);
				const t = (tau < 0 ? -1 : 1) / (Math.abs(tau) + Math.hypot(1, tau));
				const c = 1 / Math.hypot(1, t);
				const s = t * c;
				a[p][p] -= t * apq;
				a[q][q] += t * apq;
				a[p][q] = 0;
				a[q][p] = 0;
				for (let k = 0; k < n; k++) {
					if (k !== p && k !== q) {
						const akp = a[k][p];
						const akq = a[k][q];
						a[k][p] = a[p][k] = c * akp - s * akq;
						a[k][q] = a[q][k] = s * akp + c * akq;
					}
					const vkp = v[k][p];
					const vkq = v[k][q];
					v[k][p] = c * vkp - s * vkq;
					v[k][q] = s * vkp + c * vkq;
				}
			}
		}
	}
	const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => a[j][j] - a[i][i]);
	return { values: order.map((i) => a[i][i]), vectors: v.map((row) => order.map((i) => row[i])) };
}

function dot(a: readonly number[], b: readonly number[]): number {
	let sum = 0;
	for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
	return sum;
}

function subtractProjection(vector: number[], basis: readonly number[][]): void {
	for (const axis of basis) {
		const coefficient = dot(vector, axis);
		for (let i = 0; i < vector.length; i++) vector[i] -= coefficient * axis[i];
	}
}

/** Completes an orthonormal frame even when the cross-covariance has deficient rank. */
function completeAxis(candidate: number[], basis: readonly number[][]): number[] {
	subtractProjection(candidate, basis);
	let norm = Math.hypot(...candidate);
	if (norm < EPS) {
		let best: number[] = [1, 0, 0];
		let bestNorm = -1;
		for (let axis = 0; axis < 3; axis++) {
			const trial = [0, 0, 0];
			trial[axis] = 1;
			subtractProjection(trial, basis);
			const trialNorm = Math.hypot(...trial);
			if (trialNorm > bestNorm) {
				best = trial;
				bestNorm = trialNorm;
			}
		}
		candidate = best;
		norm = bestNorm;
	}
	return candidate.map((value) => value / norm);
}

/**
 * Align corresponding points by orthogonal Procrustes, including reflections.
 * No dilation is fitted: changed distances and real growth are preserved.
 */
export function alignPoints(
	points: readonly Point3[],
	reference: readonly Point3[],
	include?: readonly boolean[]
): { positions: Point3[]; applied: boolean; rmsDisplacement: number | null } {
	if ([...points, ...reference].some((point) => point.some((value) => !Number.isFinite(value)))) {
		throw new Error('Alignment coordinates must be finite.');
	}
	if (points.length !== reference.length) {
		return { positions: points.map((point) => [...point]), applied: false, rmsDisplacement: null };
	}
	const ids = points.flatMap((_, i) => (include?.[i] === false ? [] : [i]));
	if (ids.length < 2) {
		return { positions: points.map((point) => [...point]), applied: false, rmsDisplacement: null };
	}
	const center = ZERO();
	const targetCenter = ZERO();
	for (const i of ids) {
		for (let a = 0; a < 3; a++) {
			center[a] += points[i][a] / ids.length;
			targetCenter[a] += reference[i][a] / ids.length;
		}
	}
	const cross = [ZERO(), ZERO(), ZERO()];
	for (const i of ids) {
		for (let a = 0; a < 3; a++) {
			for (let b = 0; b < 3; b++) {
				cross[a][b] += (points[i][a] - center[a]) * (reference[i][b] - targetCenter[b]);
			}
		}
	}
	const ctC = Array.from({ length: 3 }, (_, a) =>
		Array.from({ length: 3 }, (_, b) => cross.reduce((sum, row) => sum + row[a] * row[b], 0))
	);
	const eigen = symmetricEigen(ctC);
	const right = [0, 1, 2].map((a) => eigen.vectors.map((row) => row[a]));
	const left: number[][] = [];
	const largest = Math.sqrt(Math.max(0, eigen.values[0]));
	for (let a = 0; a < 3; a++) {
		const singular = Math.sqrt(Math.max(0, eigen.values[a]));
		const candidate =
			singular > largest * 1e-8 && singular > 0
				? cross.map((row) => dot(row, right[a]) / singular)
				: [0, 0, 0];
		left.push(completeAxis(candidate, left));
	}
	// R = U V^T, acting on row-vector positions from the right.
	const rotation = Array.from({ length: 3 }, (_, a) =>
		Array.from({ length: 3 }, (_, b) =>
			left.reduce((sum, column, k) => sum + column[a] * right[k][b], 0)
		)
	);
	const positions: Point3[] = points.map((point) => {
		const aligned = ZERO();
		for (let a = 0; a < 3; a++) {
			aligned[a] = targetCenter[a];
			for (let b = 0; b < 3; b++) aligned[a] += (point[b] - center[b]) * rotation[b][a];
		}
		return aligned;
	});
	const rmsDisplacement = Math.sqrt(
		ids.reduce((sum, i) => sum + squaredDistance(positions[i], reference[i]), 0) / ids.length
	);
	return { positions, applied: true, rmsDisplacement };
}

export function getNeighbors(
	geometry: GeometryResult,
	index: number,
	count = geometry.neighborCount,
	space: 'original' | 'projected' = 'original'
): Neighbor[] {
	if (!geometry.valid[index] || geometry.rank === 0) return [];
	const matrix = space === 'original' ? geometry.originalDistances : geometry.projectedDistances;
	return matrix[index]
		.flatMap((distance, other) =>
			other !== index && geometry.valid[other] ? [{ index: other, distance }] : []
		)
		.sort((a, b) => a.distance - b.distance || a.index - b.index)
		.slice(0, Math.max(0, Math.floor(count)));
}

/**
 * Center activation histories across probes, then L2 normalize each neuron.
 * Effect rows are assumed to contain measured, already centered-logit changes.
 * PCA centers the resulting neuron cloud across neurons. Invalid/zero fingerprints
 * receive no neighbors and are kept at the origin for an explicit unresolved view.
 */
export function buildGeometry(
	input: readonly number[][],
	options: GeometryOptions = { kind: 'activation' }
): GeometryResult {
	const dimensions = input[0]?.length ?? 0;
	if (input.some((row) => row.length !== dimensions)) {
		throw new Error('Fingerprints must all use the same ordered probe dimensions.');
	}
	if (input.some((row) => row.some((value) => !Number.isFinite(value)))) {
		throw new Error('Fingerprints must contain finite measurements.');
	}
	if (options.neighbors !== undefined && !Number.isFinite(options.neighbors)) {
		throw new Error('The neighbor count must be finite.');
	}
	const magnitudes: number[] = [];
	const valid: boolean[] = [];
	const fingerprints = input.map((row) => {
		const mean =
			options.kind === 'activation' && dimensions
				? row.reduce((sum, value) => sum + value / dimensions, 0)
				: 0;
		const centered = row.map((value) => value - mean);
		const norm = Math.sqrt(dot(centered, centered));
		if (!Number.isFinite(norm))
			throw new Error('Fingerprint magnitudes exceed the supported numeric range.');
		const isValid = norm > EPS;
		// RMS measures activation variation / effect strength independently of normalization.
		magnitudes.push(dimensions ? norm / Math.sqrt(dimensions) : 0);
		valid.push(isValid);
		const normalize = options.kind === 'activation' || options.normalizeEffects !== false;
		return centered.map((value) => (isValid ? value / (normalize ? norm : 1) : 0));
	});
	const ids = valid.flatMap((isValid, i) => (isValid ? [i] : []));
	const mean = Array<number>(dimensions).fill(0);
	for (const i of ids)
		for (let d = 0; d < dimensions; d++) mean[d] += fingerprints[i][d] / ids.length;
	const centered = ids.map((i) => fingerprints[i].map((value, d) => value - mean[d]));
	const gram = centered.map((row, i) => centered.map((other, j) => (i <= j ? dot(row, other) : 0)));
	for (let i = 0; i < gram.length; i++) for (let j = 0; j < i; j++) gram[i][j] = gram[j][i];
	const { values, vectors } = symmetricEigen(gram);
	if (values.some((value) => !Number.isFinite(value))) {
		throw new Error('Fingerprint covariance exceeds the supported numeric range.');
	}
	const positive = values.map((value) => Math.max(0, value));
	const total = positive.reduce((sum, value) => sum + value, 0);
	const inputEnergy = fingerprints.reduce((sum, row) => sum + dot(row, row), 0);
	if (!Number.isFinite(total) || !Number.isFinite(inputEnergy)) {
		throw new Error('Fingerprint covariance exceeds the supported numeric range.');
	}
	const cutoff = Math.max(inputEnergy * EPS * EPS, (positive[0] ?? 0) * 1e-10);
	const rank = positive.filter((value) => value > cutoff).length;
	const explainedVarianceByAxis: Point3 = [0, 1, 2].map((axis) =>
		total > 0 && (positive[axis] ?? 0) > cutoff ? (positive[axis] ?? 0) / total : 0
	) as Point3;
	let positions: Point3[] = input.map(ZERO);
	for (let row = 0; row < ids.length; row++) {
		for (let axis = 0; axis < Math.min(3, rank); axis++) {
			positions[ids[row]][axis] = vectors[row][axis] * Math.sqrt(positive[axis]);
		}
	}
	let alignment: GeometryResult['alignment'] = { applied: false, rmsDisplacement: null };
	if (
		options.previous?.kind === options.kind &&
		options.previous.positions.length === input.length
	) {
		const aligned = alignPoints(
			positions,
			options.previous.positions,
			valid.map((value, i) => value && options.previous!.valid[i])
		);
		positions = aligned.positions.map((point, i) => (valid[i] ? point : ZERO()));
		alignment = { applied: aligned.applied, rmsDisplacement: aligned.rmsDisplacement };
	}
	const result: GeometryResult = {
		kind: options.kind,
		positions,
		edges: [],
		magnitudes,
		valid,
		fingerprints,
		originalDistances: distances(fingerprints),
		projectedDistances: distances(positions),
		neighborCount: Math.min(
			Math.max(0, Math.floor(options.neighbors ?? 6)),
			Math.max(0, ids.length - 1)
		),
		neighborRetention: null,
		perNodeRetention: input.map(() => null),
		explainedVariance: explainedVarianceByAxis.reduce((sum, value) => sum + value, 0),
		explainedVarianceByAxis,
		rank,
		boundaryDegenerate: rank > 3 && Math.abs(positive[2] - positive[3]) <= positive[0] * 1e-8,
		alignment
	};
	const edges = new Set<string>();
	for (const i of ids) {
		if (rank === 0) continue;
		const original = getNeighbors(result, i);
		for (const neighbor of original) {
			const a = Math.min(i, neighbor.index);
			const b = Math.max(i, neighbor.index);
			const key = `${a}:${b}`;
			if (!edges.has(key)) {
				edges.add(key);
				result.edges.push([a, b]);
			}
		}
		if (!original.length || rank === 0) continue;
		const radius = original[original.length - 1].distance;
		const tieTolerance = Math.max(...result.originalDistances[i]) * 1e-9;
		const projected = getNeighbors(result, i, result.neighborCount, 'projected');
		result.perNodeRetention[i] =
			projected.filter(({ index }) => result.originalDistances[i][index] <= radius + tieTolerance)
				.length / projected.length;
	}
	const retentions = result.perNodeRetention.filter((value): value is number => value !== null);
	result.neighborRetention = retentions.length
		? retentions.reduce((sum, value) => sum + value, 0) / retentions.length
		: null;
	return result;
}
