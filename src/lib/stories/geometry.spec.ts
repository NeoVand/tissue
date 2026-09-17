import { describe, expect, it, vi, afterEach } from 'vitest';
import { buildGeometry } from '../lab/geometry';
import {
	buildStoryGeometry,
	prepareStoryGeometryIndex,
	storyAuditIds,
	storyNearestNeighbors
} from './geometry';
import { StoryGeometryEngine } from './geometry-engine';
import type { StoryAtlas } from './protocol';

function atlas(rows: number[][], hidden = rows.length): StoryAtlas {
	return {
		version: 1,
		modelId: 'geometry-fixture',
		corpusId: 'fixed-probes',
		seed: 42,
		step: 0,
		backend: 'cpu',
		capturedAt: '2026-09-16T00:00:00Z',
		elapsedMs: 0,
		config: {
			layers: rows.length / hidden,
			hidden,
			width: 8,
			heads: 1,
			context: 128,
			batchSize: 1,
			vocabularySize: 96,
			learningRate: 0.001
		},
		unitCount: rows.length,
		dimensions: rows[0].length,
		fingerprints: Float32Array.from(rows.flat()),
		positions: rows[0].map((_, i) => i),
		examples: ['fixed calibration window']
	};
}
function rows(n = 30, p = 12) {
	return Array.from({ length: n }, (_, i) =>
		Array.from({ length: p }, (_, d) =>
			Math.fround(3 + Math.sin((i + 0.3) * (d + 1.7)) + Math.cos(i * 1.9 + d))
		)
	);
}
const distance = (a: number[], b: number[]) => Math.hypot(...a.map((v, i) => v - b[i]));

describe('scalable fixed-probe activation geometry', () => {
	it('agrees with an independent exact neuron-Gram PCA fit on a small nondegenerate cloud', () => {
		const input = rows();
		const exact = buildGeometry(input, { kind: 'activation' });
		const { geometry } = buildStoryGeometry(atlas(input));
		expect(geometry.pca.converged).toBe(true);
		expect(geometry.pca.relativeResidual).toBeLessThanOrEqual(1e-8);
		expect(geometry.explainedVariance).toBeCloseTo(exact.explainedVariance, 9);
		for (let i = 0; i < input.length; i++)
			for (let j = i + 1; j < input.length; j++)
				expect(distance(geometry.positions[i], geometry.positions[j])).toBeCloseTo(
					exact.projectedDistances[i][j],
					6
				);
	});

	it('keeps centered RMS separate from signed correlation and excludes constant rows', () => {
		const { geometry, index } = buildStoryGeometry(
			atlas([
				[1, 2, 4, 8],
				[12, 14, 18, 26],
				[8, 7, 5, 1],
				[3, 3, 3, 3]
			])
		);
		expect(geometry.valid).toEqual([true, true, true, false]);
		expect(geometry.magnitudes[1]).toBeCloseTo(geometry.magnitudes[0] * 2);
		const neighbors = storyNearestNeighbors(index, 0);
		expect(neighbors.map((v) => v.index)).toEqual([1, 2]);
		expect(neighbors[0].distance).toBeCloseTo(0);
		expect(neighbors[1].distance).toBeCloseTo(2);
		expect(neighbors[1].similarity).toBeCloseTo(-1);
		expect(storyNearestNeighbors(index, 3)).toEqual([]);
		expect(geometry.edges.every(([a, b]) => a !== 3 && b !== 3)).toBe(true);
		expect(geometry.audit.resolvedFocals).toBe(3);
	});

	it('performs exact selected-unit scans with explicit layer filtering and deterministic ties', () => {
		const input = rows(24, 11);
		input[1] = [...input[0]];
		input[2] = [...input[0]];
		const { index } = prepareStoryGeometryIndex(atlas(input, 8));
		const brute = Array.from({ length: 8 }, (_, i) => i + 8)
			.map((id) => ({
				index: id,
				distance: distance(
					Array.from(index.normalized.slice(0, 11)),
					Array.from(index.normalized.slice(id * 11, (id + 1) * 11))
				)
			}))
			.sort((a, b) => a.distance - b.distance || a.index - b.index)
			.slice(0, 6);
		const selected = storyNearestNeighbors(index, 0, 6, 1);
		expect(selected.map((v) => v.index)).toEqual(brute.map((v) => v.index));
		selected.forEach((v, i) => expect(v.distance).toBeCloseTo(brute[i].distance, 12));
		expect(storyNearestNeighbors(index, 0, 2).map((v) => v.index)).toEqual([1, 2]);
		expect(() => storyNearestNeighbors(index, 0, 6, 3)).toThrow();
	});

	it('fixes the audit schedule before seeing values and reports unresolved samples without replacement', () => {
		const input = rows(180, 5),
			ids = storyAuditIds(180);
		for (const id of ids.slice(0, 20)) input[id].fill(1);
		const { geometry } = buildStoryGeometry(atlas(input));
		expect(new Set(ids).size).toBe(128);
		expect(geometry.audit.ids).toEqual(ids);
		expect(geometry.audit.planned).toBe(128);
		expect(geometry.audit.resolvedFocals).toBe(108);
		expect(geometry.audit.scoredFocals).toBe(108);
		expect(geometry.audit.perFocal.filter((v) => v.retention === null).map((v) => v.id)).toEqual(
			ids.slice(0, 20)
		);
		expect(geometry.edges.length).toBeLessThanOrEqual(108 * 6);
		expect(geometry.edgeScope).toBe('audit-sample');
	});

	it('retains all neighbors for a cloud contained in three dimensions and handles ties', () => {
		const input = rows(20, 4);
		input[1] = [...input[0]];
		const { geometry } = buildStoryGeometry(atlas(input));
		expect(geometry.explainedVariance).toBeCloseTo(1, 12);
		expect(geometry.audit.neighborRetention).toBe(1);
		expect(geometry.audit.perFocal.every((row) => row.neighborCount === 6)).toBe(true);
	});

	it('reports unresolved or direction-identical clouds without inventing projection quality', () => {
		for (const input of [
			Array.from({ length: 4 }, () => [0, 0, 0]),
			Array.from({ length: 4 }, () => [1, 2, 4])
		]) {
			const { geometry } = buildStoryGeometry(atlas(input));
			expect(geometry.positions.flat().every(Number.isFinite)).toBe(true);
			expect(geometry.explainedVariance).toBe(0);
			expect(geometry.pca.totalVariance).toBe(0);
			expect(geometry.pca.boundaryUncertain).toBe(true);
			expect(geometry.audit.neighborRetention).toBeNull();
			expect(geometry.audit.scoredFocals).toBe(0);
		}
	});

	it('flags a degenerate third/fourth eigenvalue boundary and exposes incomplete convergence', () => {
		const symmetric = Array.from({ length: 5 }, (_, i) =>
			Array.from({ length: 5 }, (_, j) => +(i === j))
		);
		const { geometry: tied } = buildStoryGeometry(atlas(symmetric));
		expect(tied.pca.boundaryUncertain).toBe(true);
		expect(tied.explainedVariance).toBeCloseTo(0.75);
		const { geometry: interrupted } = buildStoryGeometry(atlas(rows(40, 32)), {
			maxIterations: 1,
			tolerance: 1e-12
		});
		expect(interrupted.pca.converged).toBe(false);
		expect(interrupted.pca.iterations).toBe(1);
		expect(interrupted.pca.relativeResidual).toBeGreaterThan(1e-12);
	});

	it('is deterministic and permutation equivariant for a separated PCA subspace', () => {
		const input = rows(20, 10);
		const original = buildStoryGeometry(atlas(input)).geometry;
		expect(buildStoryGeometry(atlas(input)).geometry.positions).toEqual(original.positions);
		const permutation = input.map((_, i) => (i * 7) % input.length);
		const permuted = buildStoryGeometry(atlas(permutation.map((i) => input[i]))).geometry;
		for (let i = 0; i < input.length; i++)
			for (let j = i + 1; j < input.length; j++)
				expect(distance(permuted.positions[i], permuted.positions[j])).toBeCloseTo(
					distance(original.positions[permutation[i]], original.positions[permutation[j]]),
					7
				);
	});

	it('aligns matching probe identities with reflections, without dilation, and resets on a different model', () => {
		const input = atlas(rows(20, 6));
		const first = buildStoryGeometry(input).geometry;
		const reference = {
			...first,
			positions: first.positions.map(([x, y, z]): [number, number, number] => [
				-y + 2,
				x - 3,
				z + 4
			])
		};
		const aligned = buildStoryGeometry({ ...input, step: 10 }, { previous: reference }).geometry;
		expect(aligned.alignment.applied).toBe(true);
		expect(aligned.alignment.rmsDisplacement).toBeLessThan(1e-7);
		expect(distance(aligned.positions[0], aligned.positions[1])).toBeCloseTo(
			distance(first.positions[0], first.positions[1]),
			12
		);
		const other = buildStoryGeometry(
			{ ...input, modelId: 'another-model' },
			{ previous: reference }
		).geometry;
		expect(other.alignment.applied).toBe(false);
		expect(other.positions).toEqual(first.positions);
	});

	it('rejects malformed shapes and nonfinite values before fitting', () => {
		const input = atlas(rows(4, 5));
		expect(() => buildStoryGeometry({ ...input, dimensions: 8 })).toThrow();
		expect(() => buildStoryGeometry({ ...input, fingerprints: new Float32Array(1) })).toThrow();
		input.fingerprints[0] = Infinity;
		expect(() => buildStoryGeometry(input)).toThrow(/Nonfinite/);
		expect(() => buildStoryGeometry(atlas(rows()), { rmsFloor: 0 })).toThrow();
	});

	it('handles 9,216 units with a compact result and O(NP) retained index', () => {
		const input = atlas(rows(9216, 128), 1536);
		const { geometry, index } = buildStoryGeometry(input);
		expect(geometry.positions.length).toBe(9216);
		expect(index.normalized.length).toBe(9216 * 128);
		expect(index.normalized.byteLength).toBe(9_437_184);
		expect(geometry.audit.planned).toBe(128);
		expect(geometry.edges.length).toBeLessThanOrEqual(768);
		expect(JSON.stringify(geometry).length).toBeLessThan(2_000_000);
		expect(geometry.positions.flat().every(Number.isFinite)).toBe(true);
		expect(storyNearestNeighbors(index, 9215, 6, 5)).toHaveLength(6);
	}, 15_000);
});

describe('story geometry worker lifecycle', () => {
	afterEach(() => vi.unstubAllGlobals());
	it('copies raw data for transfer, rejects canceled requests and ignores stale generations', async () => {
		class WorkerMock {
			static instances: WorkerMock[] = [];
			onmessage: ((event: { data: unknown }) => void) | null = null;
			onerror = null;
			onmessageerror = null;
			requests: Array<{ id: number; generation: number; atlas: StoryAtlas }> = [];
			terminated = false;
			constructor() {
				WorkerMock.instances.push(this);
			}
			postMessage(request: { id: number; generation: number; atlas: StoryAtlas }) {
				this.requests.push(request);
			}
			terminate() {
				this.terminated = true;
			}
		}
		vi.stubGlobal('Worker', WorkerMock);
		const engine = new StoryGeometryEngine(),
			input = atlas(rows(4, 4));
		const pending = engine.build(input);
		const rejection = expect(pending).rejects.toThrow(/reset/);
		const first = WorkerMock.instances[0];
		expect(first.requests[0].atlas.fingerprints).not.toBe(input.fingerprints);
		expect(Array.from(first.requests[0].atlas.fingerprints)).toEqual(
			Array.from(input.fingerprints)
		);
		engine.reset();
		await rejection;
		expect(first.terminated).toBe(true);
		const current = engine.setAtlas(input),
			second = WorkerMock.instances[1];
		first.onmessage?.({ data: { ...first.requests[0], ok: true, result: null } });
		second.onmessage?.({ data: { ...second.requests[0], ok: true, result: null } });
		await current;
		const neighbors = engine.neighbors(0);
		const disposed = expect(neighbors).rejects.toThrow(/disposed/);
		engine.destroy();
		await disposed;
		await expect(engine.build(input)).rejects.toThrow(/disposed/);
		expect(second.terminated).toBe(true);
	});
});
