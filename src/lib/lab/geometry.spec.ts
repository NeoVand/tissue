import { describe, expect, it } from 'vitest';
import { alignPoints, buildGeometry, getNeighbors, type Point3 } from './geometry';

function expectDistanceMatricesClose(a: number[][], b: number[][], digits = 8): void {
	expect(a.length).toBe(b.length);
	for (let i = 0; i < a.length; i++) {
		for (let j = 0; j < a.length; j++) expect(a[i][j]).toBeCloseTo(b[i][j], digits);
	}
}

describe('functional geometry', () => {
	it('preserves pairwise distances when the measured cloud fits in three dimensions', () => {
		const points = [
			[2, 1, 0],
			[0, 3, 0],
			[-2, 0, 1],
			[0, -1, -2],
			[1, 1, 3]
		];
		const geometry = buildGeometry(points, {
			kind: 'effect',
			normalizeEffects: false,
			neighbors: 2
		});
		expect(geometry.rank).toBe(3);
		expect(geometry.explainedVariance).toBeCloseTo(1);
		expect(geometry.neighborRetention).toBe(1);
		expectDistanceMatricesClose(geometry.projectedDistances, geometry.originalDistances);
	});

	it('reports the analytically known amount of variance discarded by a 4D cloud', () => {
		const points = [4, 3, 2, 1].flatMap((scale, axis) =>
			[-1, 1].map((sign) => [0, 1, 2, 3].map((d) => (d === axis ? scale * sign : 0)))
		);
		const geometry = buildGeometry(points, {
			kind: 'effect',
			normalizeEffects: false,
			neighbors: 2
		});
		expect(geometry.rank).toBe(4);
		expect(geometry.explainedVariance).toBeCloseTo(58 / 60);
		expect(geometry.explainedVarianceByAxis[0]).toBeCloseTo(32 / 60);
		expect(geometry.explainedVarianceByAxis[1]).toBeCloseTo(18 / 60);
		expect(geometry.explainedVarianceByAxis[2]).toBeCloseTo(8 / 60);
		expect(geometry.boundaryDegenerate).toBe(false);
		for (let i = 0; i < points.length; i++) {
			for (let j = 0; j < points.length; j++) {
				expect(geometry.projectedDistances[i][j]).toBeLessThanOrEqual(
					geometry.originalDistances[i][j] + 1e-9
				);
			}
		}
	});

	it('defines activation similarity by centered correlation, keeping amplitude separate', () => {
		const geometry = buildGeometry([
			[1, 2, 4, 8],
			[12, 14, 18, 26],
			[-1, -2, -4, -8],
			[3, 3, 3, 3]
		]);
		expect(geometry.originalDistances[0][1]).toBeCloseTo(0);
		expect(geometry.originalDistances[0][2]).toBeCloseTo(2);
		expect(geometry.magnitudes[1]).toBeCloseTo(2 * geometry.magnitudes[0]);
		expect(geometry.valid).toEqual([true, true, true, false]);
		expect(getNeighbors(geometry, 0, 20).map((neighbor) => neighbor.index)).toEqual([1, 2]);
		expect(getNeighbors(geometry, 3)).toEqual([]);
		expect(geometry.edges.every(([a, b]) => a !== 3 && b !== 3)).toBe(true);
	});

	it('lets effect magnitude participate only when explicitly requested', () => {
		const input = [
			[1, -1],
			[2, -2],
			[-1, 1]
		];
		const direction = buildGeometry(input, { kind: 'effect' });
		const raw = buildGeometry(input, { kind: 'effect', normalizeEffects: false });
		expect(direction.originalDistances[0][1]).toBeCloseTo(0);
		expect(raw.originalDistances[0][1]).toBeCloseTo(Math.sqrt(2));
		expect(direction.magnitudes).toEqual([1, 2, 1]);
	});

	it('retains the same PCA rank and variance fractions for small measured effects', () => {
		const input = [
			[2, 1, 0],
			[0, 3, 0],
			[-2, 0, 1],
			[0, -1, -2],
			[1, 1, 3]
		];
		const ordinary = buildGeometry(input, { kind: 'effect', normalizeEffects: false });
		const tiny = buildGeometry(
			input.map((row) => row.map((value) => value * 1e-8)),
			{
				kind: 'effect',
				normalizeEffects: false
			}
		);
		expect(tiny.rank).toBe(ordinary.rank);
		expect(tiny.explainedVariance).toBeCloseTo(ordinary.explainedVariance);
		for (let i = 0; i < input.length; i++) {
			for (let j = 0; j < input.length; j++) {
				expect(tiny.projectedDistances[i][j] * 1e8).toBeCloseTo(ordinary.projectedDistances[i][j]);
			}
		}
	});

	it('is deterministic and permutation equivariant in distances for a distinct PCA subspace', () => {
		const input = Array.from({ length: 12 }, (_, i) =>
			Array.from({ length: 7 }, (_, j) => Math.sin((i + 0.3) * (j + 1.7)) + Math.cos(i * 1.9 + j))
		);
		const original = buildGeometry(input, { kind: 'effect', normalizeEffects: false });
		const repeated = buildGeometry(input, { kind: 'effect', normalizeEffects: false });
		expect(original.positions).toEqual(repeated.positions);
		const order = [9, 3, 1, 7, 11, 6, 4, 0, 8, 2, 10, 5];
		const permuted = buildGeometry(
			order.map((i) => input[i]),
			{ kind: 'effect', normalizeEffects: false }
		);
		const restore = input.map((_, i) => order.indexOf(i));
		expectDistanceMatricesClose(
			original.projectedDistances,
			restore.map((i) => restore.map((j) => permuted.projectedDistances[i][j]))
		);
		expect(permuted.explainedVariance).toBeCloseTo(original.explainedVariance);
	});

	it('explicitly flags a degenerate third/fourth PCA boundary', () => {
		const points = [0, 1, 2, 3].flatMap((axis) =>
			[-1, 1].map((sign) => [0, 1, 2, 3].map((d) => (d === axis ? sign : 0)))
		);
		const geometry = buildGeometry(points, { kind: 'effect' });
		expect(geometry.boundaryDegenerate).toBe(true);
		expect(geometry.explainedVariance).toBeCloseTo(0.75);
	});

	it('does not inflate neighborhood retention when raw effects are small', () => {
		const input = Array.from({ length: 24 }, (_, i) =>
			Array.from({ length: 7 }, (_, j) => Math.sin((i + 0.3) * (j + 1.7)) + Math.cos(i * 1.9 + j))
		);
		const ordinary = buildGeometry(input, {
			kind: 'effect',
			normalizeEffects: false,
			neighbors: 3
		});
		const tiny = buildGeometry(
			input.map((row) => row.map((value) => value * 1e-11)),
			{
				kind: 'effect',
				normalizeEffects: false,
				neighbors: 3
			}
		);
		expect(ordinary.neighborRetention).toBeLessThan(1);
		expect(tiny.neighborRetention).toBeCloseTo(ordinary.neighborRetention!);
	});

	it.each(
		[
			[],
			[[], []],
			[
				[0, 0],
				[0, 0]
			],
			[
				[1, 1],
				[2, 2]
			],
			[[1, 2]],
			[
				[1, 2],
				[1, 2]
			]
		].map((input) => ({ input }))
	)(
		'keeps degenerate histories finite without claiming informative neighborhoods: %j',
		({ input }) => {
			const geometry = buildGeometry(input);
			expect(geometry.positions.flat().every(Number.isFinite)).toBe(true);
			expect(geometry.magnitudes.every(Number.isFinite)).toBe(true);
			expect(geometry.rank).toBe(0);
			expect(geometry.neighborRetention).toBeNull();
			expect(geometry.explainedVariance).toBe(0);
			expect(geometry.edges).toEqual([]);
			expect(getNeighbors(geometry, 0)).toEqual([]);
		}
	);

	it('rejects incompatible and unmeasured probe data rather than silently inventing zeros', () => {
		expect(() => buildGeometry([[1], [1, 2]])).toThrow('same ordered probe dimensions');
		expect(() => buildGeometry([[1, NaN]])).toThrow('finite measurements');
		expect(() => buildGeometry([[1, Infinity]])).toThrow('finite measurements');
		expect(() => buildGeometry([[1e308, -1e308]])).toThrow('numeric range');
	});

	it('returns measured neighbor distances from the requested space', () => {
		const geometry = buildGeometry(
			[
				[1, 0],
				[1, 0.1],
				[-1, 0],
				[0, 1]
			],
			{ kind: 'effect', neighbors: 2 }
		);
		const neighbors = getNeighbors(geometry, 0);
		expect(neighbors.map(({ index }) => index)).toEqual([1, 3]);
		expect(neighbors[0].distance).toBe(geometry.originalDistances[0][1]);
		expect(getNeighbors(geometry, 0, 0)).toEqual([]);
		expect(getNeighbors(geometry, -1)).toEqual([]);
	});
});

describe('temporal frame alignment', () => {
	const reference: Point3[] = [
		[1, 2, -1],
		[-2, 1, 0],
		[1, -2, 3],
		[4, 0, -2],
		[0, -1, -2]
	];

	it('recovers a reflected, rotated, translated coordinate frame', () => {
		const moved: Point3[] = reference.map(([x, y, z]) => [y + 12, z - 7, -x + 3]);
		const aligned = alignPoints(moved, reference);
		expect(aligned.applied).toBe(true);
		expect(aligned.rmsDisplacement).toBeCloseTo(0, 9);
		for (let i = 0; i < reference.length; i++) {
			for (let d = 0; d < 3; d++) expect(aligned.positions[i][d]).toBeCloseTo(reference[i][d], 9);
		}
	});

	it('handles planar and linear clouds without introducing NaNs', () => {
		const planar: Point3[] = [
			[1, 2, 0],
			[0, 2, 0],
			[-2, 1, 0],
			[1, -3, 0]
		];
		const linear: Point3[] = [
			[1, 0, 0],
			[2, 0, 0],
			[-3, 0, 0]
		];
		for (const points of [planar, linear]) {
			const moved: Point3[] = points.map(([x, y, z]) => [z + 7, x - 2, y + 9]);
			const aligned = alignPoints(moved, points);
			expect(aligned.positions.flat().every(Number.isFinite)).toBe(true);
			expect(aligned.rmsDisplacement).toBeCloseTo(0, 8);
		}
	});

	it('preserves actual size change instead of fitting it away', () => {
		const scaled: Point3[] = reference.map(([x, y, z]) => [x * 2, y * 2, z * 2]);
		const aligned = alignPoints(scaled, reference);
		expect(aligned.rmsDisplacement).toBeGreaterThan(1);
		const originalDistance = Math.hypot(...reference[0].map((value, d) => value - reference[1][d]));
		const alignedDistance = Math.hypot(
			...aligned.positions[0].map((value, d) => value - aligned.positions[1][d])
		);
		expect(alignedDistance).toBeCloseTo(originalDistance * 2);
	});

	it('fits only shared valid neuron identities when a previous map exists', () => {
		const previous = buildGeometry(reference, { kind: 'effect', normalizeEffects: false });
		const current = buildGeometry(reference, { kind: 'effect', normalizeEffects: false, previous });
		expect(current.alignment.applied).toBe(true);
		expect(current.alignment.rmsDisplacement).toBeCloseTo(0);
		expectDistanceMatricesClose(current.projectedDistances, previous.projectedDistances);
	});
});
