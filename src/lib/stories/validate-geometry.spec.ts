import { describe, expect, it } from 'vitest';
import { buildStoryGeometry, type StoryGeometryResult } from './geometry';
import { validateStoryGeometry } from './validate-geometry';
import type { StoryAtlas } from './protocol';

function fixture(n = 24, p = 12): StoryAtlas {
	return {
		version: 1,
		modelId: 'validation-fixture',
		corpusId: 'fixed-calibration',
		seed: 42,
		step: 10,
		backend: 'cpu',
		capturedAt: '2026-09-16T00:00:00Z',
		elapsedMs: 0,
		config: {
			layers: 1,
			hidden: n,
			width: 8,
			heads: 1,
			context: 128,
			batchSize: 1,
			vocabularySize: 96,
			learningRate: 0.001
		},
		unitCount: n,
		dimensions: p,
		fingerprints: Float32Array.from(
			{ length: n * p },
			(_, i) =>
				3 +
				Math.sin((Math.floor(i / p) + 0.3) * ((i % p) + 1.7)) +
				Math.cos(Math.floor(i / p) * 1.9 + (i % p))
		),
		positions: Array.from({ length: p }, (_, i) => i),
		examples: ['fixed calibration window']
	};
}

describe('archived story geometry validation', () => {
	it('accepts real fits, aligned fits, declared options, and unresolved/identical clouds', () => {
		const atlas = fixture();
		const first = buildStoryGeometry(atlas).geometry;
		expect(() => validateStoryGeometry(first, atlas)).not.toThrow();
		const moved = {
			...first,
			positions: first.positions.map(([x, y, z]): [number, number, number] => [
				-y + 2,
				x - 3,
				z + 4
			])
		};
		const aligned = buildStoryGeometry(atlas, { previous: moved }).geometry;
		expect(() => validateStoryGeometry(aligned, atlas)).not.toThrow();
		const custom = buildStoryGeometry(atlas, {
			auditSize: 7,
			k: 3,
			rmsFloor: 1e-7,
			maxIterations: 1,
			tolerance: 1e-12
		}).geometry;
		expect(() => validateStoryGeometry(custom, atlas)).not.toThrow();
		for (const identical of [false, true]) {
			atlas.fingerprints = Float32Array.from(
				{ length: atlas.unitCount * atlas.dimensions },
				(_, i) => (identical ? i % atlas.dimensions : 0)
			);
			expect(() => validateStoryGeometry(buildStoryGeometry(atlas).geometry, atlas)).not.toThrow();
		}
	});

	it('rejects missing nested records and nonfinite or inconsistent PCA diagnostics', () => {
		const atlas = fixture(),
			original = buildStoryGeometry(atlas).geometry;
		const mutations: Array<(g: StoryGeometryResult) => void> = [
			(g) => {
				Reflect.deleteProperty(g, 'pca');
			},
			(g) => {
				g.pca.eigenvalues = [];
			},
			(g) => {
				g.pca.relativeResidual = Infinity;
			},
			(g) => {
				g.pca.converged = !g.pca.converged;
			},
			(g) => {
				g.pca.boundaryUncertain = !g.pca.boundaryUncertain;
			},
			(g) => {
				g.pca.totalVariance *= 2;
			},
			(g) => {
				g.explainedVarianceByAxis[0] = 0;
			},
			(g) => {
				g.positions[0][0] += 2;
			}
		];
		for (const mutate of mutations) {
			const g = structuredClone(original);
			mutate(g);
			expect(() => validateStoryGeometry(g, atlas)).toThrow(/Invalid story geometry/);
		}
	});

	it('rejects changed probe order, wrong raw magnitudes and invalid floor flags', () => {
		const atlas = fixture(),
			original = buildStoryGeometry(atlas).geometry;
		expect(() =>
			validateStoryGeometry(original, { ...atlas, positions: [...atlas.positions].reverse() })
		).toThrow(/identity/);
		for (const mutate of [
			(g: StoryGeometryResult) => {
				g.identity = 'wrong';
			},
			(g: StoryGeometryResult) => {
				g.rmsFloor = 0;
			},
			(g: StoryGeometryResult) => {
				g.magnitudes[0] *= 2;
			},
			(g: StoryGeometryResult) => {
				g.valid[0] = false;
			}
		]) {
			const g = structuredClone(original);
			mutate(g);
			expect(() => validateStoryGeometry(g, atlas)).toThrow(/Invalid story geometry/);
		}
	});

	it('rejects malformed per-focal rows and forged audit coverage, schedules, counts or means', () => {
		const atlas = fixture(),
			original = buildStoryGeometry(atlas).geometry;
		const mutations: Array<(g: StoryGeometryResult) => void> = [
			(g) => {
				g.audit.perFocal[0] = null as unknown as StoryGeometryResult['audit']['perFocal'][number];
			},
			(g) => {
				g.audit.ids.reverse();
			},
			(g) => {
				g.audit.planned++;
			},
			(g) => {
				g.audit.validPopulation--;
			},
			(g) => {
				g.audit.resolvedFocals--;
			},
			(g) => {
				g.audit.scoredFocals--;
			},
			(g) => {
				g.audit.perFocal[0].neighborCount++;
			},
			(g) => {
				g.audit.perFocal[0].retention = null;
			},
			(g) => {
				g.audit.perFocal[0].retention = 0.123;
			},
			(g) => {
				g.audit.neighborRetention = (g.audit.neighborRetention! + 0.25) % 1;
			}
		];
		for (const mutate of mutations) {
			const g = structuredClone(original);
			mutate(g);
			expect(() => validateStoryGeometry(g, atlas)).toThrow(/Invalid story geometry/);
		}
	});

	it('rejects duplicate/missing/invalid edges and inconsistent alignment metadata', () => {
		const atlas = fixture(),
			original = buildStoryGeometry(atlas).geometry;
		const mutations: Array<(g: StoryGeometryResult) => void> = [
			(g) => {
				g.edges.push(g.edges[0]);
			},
			(g) => {
				g.edges = [];
			},
			(g) => {
				g.edges[0] = [0, 0];
			},
			(g) => {
				g.alignment.rmsDisplacement = 0;
			},
			(g) => {
				g.alignment = { applied: true, rmsDisplacement: NaN };
			}
		];
		for (const mutate of mutations) {
			const g = structuredClone(original);
			mutate(g);
			expect(() => validateStoryGeometry(g, atlas)).toThrow(/Invalid story geometry/);
		}
	});

	it('validates large sampled geometry and keeps invalid selected audit IDs in its coverage', () => {
		const atlas = fixture(2048, 128);
		atlas.fingerprints.fill(0, 0, 128 * 300);
		const geometry = buildStoryGeometry(atlas).geometry;
		expect(geometry.audit.resolvedFocals).toBeLessThan(128);
		expect(() => validateStoryGeometry(geometry, atlas)).not.toThrow();
	});
});
