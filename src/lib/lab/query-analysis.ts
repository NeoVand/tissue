import { buildGeometry, type GeometryResult } from './geometry';
import { MODEL_CONFIG, NEURON_COUNT } from './protocol';
import type { QueryMeasurement } from './query-protocol';

export const QUERY_ANALYSIS_DESIGN = {
	version: 'paired-query-transfer-v1',
	groupsPerSplit: 16,
	queries: 3,
	answerCoordinates: 8,
	neighbors: 6,
	candidatePool: 32,
	rmsFloor: 1e-8,
	shuffleSalt: 0x71534846
} as const;

export const QUERY_METHODS = [
	{ id: 'query-effect', label: 'Query effects', primary: true },
	{ id: 'full-effect', label: 'Full effects', primary: false },
	{ id: 'query-activation', label: 'Query activations', primary: false },
	{ id: 'outgoing-weight', label: 'Outgoing weights', primary: false }
] as const;
export type QueryMethod = (typeof QUERY_METHODS)[number]['id'];
type ByMethod<T> = Record<QueryMethod, T>;

export interface QueryMethodScore {
	neighbors: number[];
	calibrationMeanCosine: number | null;
	score: number | null;
	delta: number | null;
	resolvedTestNeighbors: number;
	coverage: number | null;
	shuffledScore: number | null;
	shuffledDelta: number | null;
}

export interface QueryUnitAnalysis {
	id: number;
	layer: number;
	channel: number;
	calibrationEligible: boolean;
	unresolvedCalibrationMethods: QueryMethod[];
	status: 'calibration-unresolved' | 'insufficient-candidates' | 'test-unresolved' | 'scored';
	strengths: {
		calibrationFullEffectRms: number;
		calibrationQueryEffectRms: number;
		calibrationQueryActivationRms: number;
		outgoingWeightRms: number;
		testFullEffectRms: number;
		testQueryEffectRms: number;
	};
	candidatePool: number[];
	candidatePoolSize: number;
	randomScore: number | null;
	randomResolvedNeighbors: number;
	randomCoverage: number | null;
	shuffledRandomScore: number | null;
	methods: ByMethod<QueryMethodScore>;
}

export interface QueryMethodSummary {
	meanScore: number | null;
	meanDelta: number | null;
	meanCoverage: number | null;
	selectedNeighbors: number;
	resolvedTestNeighbors: number;
	meanShuffledScore: number | null;
	meanShuffledDelta: number | null;
}

export interface QueryAnalysisSummary {
	totalUnits: number;
	calibrationEligibleUnits: number;
	candidateEligibleUnits: number;
	scoredUnits: number;
	calibrationUnresolvedUnits: number;
	insufficientCandidateUnits: number;
	testUnresolvedFocals: number;
	randomMeanScore: number | null;
	randomMeanCoverage: number | null;
	shuffledRandomMeanScore: number | null;
	methods: ByMethod<QueryMethodSummary>;
}

export type QueryGeometry = Omit<
	GeometryResult,
	'fingerprints' | 'originalDistances' | 'projectedDistances' | 'perNodeRetention'
>;

export interface QueryAnalysis {
	version: 1;
	design: typeof QUERY_ANALYSIS_DESIGN;
	seed: number;
	step: number;
	checkpointHash: string;
	measurementCapturedAt: string;
	units: QueryUnitAnalysis[];
	overall: QueryAnalysisSummary;
	layers: (QueryAnalysisSummary & { layer: number })[];
	/** Maps each unit to the held-out donor identity used by the descriptive shuffle. */
	shuffledTestIdentities: number[];
	/** Calibration contrasts only; global geometric links are not the matched test selections. */
	calibrationGeometry: QueryGeometry;
}

export interface QueryCalibrationInput {
	layers: number[];
	fullEffects: number[][];
	queryEffects: number[][];
	queryActivations: number[][];
	outgoingWeights: number[][];
}

export interface QueryCalibrationUnit {
	id: number;
	layer: number;
	calibrationEligible: boolean;
	unresolvedCalibrationMethods: QueryMethod[];
	strengths: Omit<QueryUnitAnalysis['strengths'], 'testFullEffectRms' | 'testQueryEffectRms'>;
	candidatePool: number[];
	selections: ByMethod<{ neighbors: number[]; meanCosine: number | null }>;
}

export interface QueryCalibrationPlan {
	units: QueryCalibrationUnit[];
}

function requireCondition(condition: unknown, message: string): asserts condition {
	if (!condition) throw new Error(`Invalid query analysis input: ${message}`);
}

function validateMatrix(rows: number[][], count: number, dimensions: number, label: string) {
	requireCondition(Array.isArray(rows) && rows.length === count, `${label} row count`);
	for (const row of rows) {
		requireCondition(Array.isArray(row) && row.length === dimensions, `${label} dimensions`);
		requireCondition(
			row.every((value) => Number.isFinite(value) && Math.abs(value) <= 3.4028234663852886e38),
			`${label} must contain finite Float32-range values`
		);
	}
}

/** Stable RMS; a zero-length vector has no measured direction. */
export function queryRms(row: readonly number[]): number {
	return row.length ? Math.hypot(...row) / Math.sqrt(row.length) : 0;
}

/** Group-major input (a coordinates, b coordinates, c coordinates), output (h1, h2). */
export function helmertQueryContrasts(
	rows: number[][],
	groupCount: number,
	coordinates: number
): number[][] {
	requireCondition(Number.isInteger(groupCount) && groupCount > 0, 'positive group count');
	requireCondition(Number.isInteger(coordinates) && coordinates > 0, 'positive coordinate count');
	validateMatrix(rows, rows.length, groupCount * 3 * coordinates, 'query contrasts');
	return rows.map((row) => {
		const result: number[] = [];
		for (let group = 0; group < groupCount; group++) {
			const start = group * 3 * coordinates;
			for (let coordinate = 0; coordinate < coordinates; coordinate++) {
				result.push((row[start + coordinates + coordinate] - row[start + coordinate]) / Math.SQRT2);
			}
			for (let coordinate = 0; coordinate < coordinates; coordinate++) {
				result.push(
					(2 * row[start + 2 * coordinates + coordinate] -
						row[start + coordinate] -
						row[start + coordinates + coordinate]) /
						Math.sqrt(6)
				);
			}
		}
		return result;
	});
}

function byMethod<T>(create: (method: QueryMethod) => T): ByMethod<T> {
	return Object.fromEntries(QUERY_METHODS.map(({ id }) => [id, create(id)])) as ByMethod<T>;
}

function normalized(row: number[]): number[] | null {
	if (queryRms(row) <= QUERY_ANALYSIS_DESIGN.rmsFloor) return null;
	const norm = Math.hypot(...row);
	return row.map((value) => value / norm);
}

function cosine(a: number[] | null, b: number[] | null): number {
	if (!a || !b) return 0;
	let sum = 0;
	for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
	return Math.max(-1, Math.min(1, sum));
}

const mean = (values: number[]) =>
	values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;

/** This function cannot access held-out data: all selection decisions stop at this boundary. */
export function buildQueryCalibrationPlan(input: QueryCalibrationInput): QueryCalibrationPlan {
	const count = input.layers.length;
	requireCondition(
		input.layers.every((layer) => Number.isInteger(layer) && layer >= 0),
		'layer IDs'
	);
	const raw = {
		'query-effect': input.queryEffects,
		'full-effect': input.fullEffects,
		'query-activation': input.queryActivations,
		'outgoing-weight': input.outgoingWeights
	};
	for (const { id } of QUERY_METHODS) {
		const dimensions = raw[id][0]?.length ?? 0;
		requireCondition(count === 0 || dimensions > 0, `${id} needs measured coordinates`);
		validateMatrix(raw[id], count, dimensions, id);
	}
	const rms = byMethod((method) => raw[method].map(queryRms));
	const directions = byMethod((method) => raw[method].map(normalized));
	const eligible = input.layers.map((_, id) =>
		QUERY_METHODS.every(({ id: method }) => directions[method][id] !== null)
	);
	return {
		units: input.layers.map((layer, id) => {
			const candidatePool = eligible[id]
				? input.layers
						.flatMap((otherLayer, other) =>
							other !== id && otherLayer === layer && eligible[other] ? [other] : []
						)
						.map((other) => ({
							id: other,
							difference: Math.abs(
								Math.log(rms['full-effect'][other]) - Math.log(rms['full-effect'][id])
							)
						}))
						.sort((a, b) => a.difference - b.difference || a.id - b.id)
						.slice(0, QUERY_ANALYSIS_DESIGN.candidatePool)
						.map((candidate) => candidate.id)
				: [];
			const enough = candidatePool.length >= QUERY_ANALYSIS_DESIGN.neighbors;
			return {
				id,
				layer,
				calibrationEligible: eligible[id],
				unresolvedCalibrationMethods: QUERY_METHODS.flatMap(({ id: method }) =>
					directions[method][id] === null ? [method] : []
				),
				strengths: {
					calibrationFullEffectRms: rms['full-effect'][id],
					calibrationQueryEffectRms: rms['query-effect'][id],
					calibrationQueryActivationRms: rms['query-activation'][id],
					outgoingWeightRms: rms['outgoing-weight'][id]
				},
				candidatePool,
				selections: byMethod((method) => {
					const selected = enough
						? candidatePool
								.map((other) => ({
									id: other,
									similarity: cosine(directions[method][id], directions[method][other])
								}))
								.sort((a, b) => b.similarity - a.similarity || a.id - b.id)
								.slice(0, QUERY_ANALYSIS_DESIGN.neighbors)
						: [];
					return {
						neighbors: selected.map((other) => other.id),
						meanCosine: mean(selected.map((other) => other.similarity))
					};
				})
			};
		})
	};
}

function shuffledIdentities(
	plan: QueryCalibrationPlan,
	resolved: boolean[],
	seed: number
): number[] {
	let state = (seed ^ QUERY_ANALYSIS_DESIGN.shuffleSalt) >>> 0;
	if (state === 0) state = 0x9e3779b9;
	const random = () => {
		state ^= state << 13;
		state ^= state >>> 17;
		state ^= state << 5;
		return (state >>> 0) / 0x100000000;
	};
	const identities = plan.units.map((unit) => unit.id);
	const layers = [...new Set(plan.units.map((unit) => unit.layer))].sort((a, b) => a - b);
	for (const layer of layers) {
		for (const status of [false, true]) {
			const members = plan.units
				.filter(
					(unit) => unit.layer === layer && unit.calibrationEligible && resolved[unit.id] === status
				)
				.map((unit) => unit.id);
			const donors = [...members];
			for (let i = donors.length - 1; i > 0; i--) {
				const j = Math.floor(random() * (i + 1));
				[donors[i], donors[j]] = [donors[j], donors[i]];
			}
			members.forEach((id, index) => (identities[id] = donors[index]));
		}
	}
	return identities;
}

function summarizeUnits(units: QueryUnitAnalysis[]): QueryAnalysisSummary {
	const scored = units.filter((unit) => unit.status === 'scored');
	return {
		totalUnits: units.length,
		calibrationEligibleUnits: units.filter((unit) => unit.calibrationEligible).length,
		candidateEligibleUnits: units.filter(
			(unit) => unit.candidatePoolSize >= QUERY_ANALYSIS_DESIGN.neighbors
		).length,
		scoredUnits: scored.length,
		calibrationUnresolvedUnits: units.filter((unit) => unit.status === 'calibration-unresolved')
			.length,
		insufficientCandidateUnits: units.filter((unit) => unit.status === 'insufficient-candidates')
			.length,
		testUnresolvedFocals: units.filter((unit) => unit.status === 'test-unresolved').length,
		randomMeanScore: mean(scored.map((unit) => unit.randomScore!)),
		randomMeanCoverage: mean(scored.map((unit) => unit.randomCoverage!)),
		shuffledRandomMeanScore: mean(scored.map((unit) => unit.shuffledRandomScore!)),
		methods: byMethod((method) => ({
			meanScore: mean(scored.map((unit) => unit.methods[method].score!)),
			meanDelta: mean(scored.map((unit) => unit.methods[method].delta!)),
			meanCoverage: mean(scored.map((unit) => unit.methods[method].coverage!)),
			selectedNeighbors: scored.reduce(
				(sum, unit) => sum + unit.methods[method].neighbors.length,
				0
			),
			resolvedTestNeighbors: scored.reduce(
				(sum, unit) => sum + unit.methods[method].resolvedTestNeighbors,
				0
			),
			meanShuffledScore: mean(scored.map((unit) => unit.methods[method].shuffledScore!)),
			meanShuffledDelta: mean(scored.map((unit) => unit.methods[method].shuffledDelta!))
		}))
	};
}

/** Scoring never changes the frozen calibration plan, even when held-out vectors are zero. */
export function scoreQueryCalibrationPlan(
	plan: QueryCalibrationPlan,
	testQueryEffects: number[][],
	testFullEffects: number[][],
	seed: number
): Pick<QueryAnalysis, 'units' | 'overall' | 'layers' | 'shuffledTestIdentities'> {
	const count = plan.units.length;
	validateMatrix(
		testQueryEffects,
		count,
		testQueryEffects[0]?.length ?? 0,
		'held-out query effects'
	);
	validateMatrix(testFullEffects, count, testFullEffects[0]?.length ?? 0, 'held-out full effects');
	const directions = testQueryEffects.map(normalized);
	const resolved = directions.map((direction) => direction !== null);
	const shuffledTestIdentities = shuffledIdentities(plan, resolved, seed);
	const similarity = (id: number, other: number) => cosine(directions[id], directions[other]);
	const shuffledSimilarity = (id: number, other: number) =>
		similarity(shuffledTestIdentities[id], shuffledTestIdentities[other]);
	const units: QueryUnitAnalysis[] = plan.units.map((unit) => {
		const enough = unit.candidatePool.length >= QUERY_ANALYSIS_DESIGN.neighbors;
		const scored = unit.calibrationEligible && enough && resolved[unit.id];
		const randomScore = scored
			? mean(unit.candidatePool.map((other) => similarity(unit.id, other)))
			: null;
		const shuffledRandomScore = scored
			? mean(unit.candidatePool.map((other) => shuffledSimilarity(unit.id, other)))
			: null;
		const randomResolvedNeighbors = unit.candidatePool.filter((other) => resolved[other]).length;
		return {
			id: unit.id,
			layer: unit.layer,
			channel: unit.id % MODEL_CONFIG.hidden,
			calibrationEligible: unit.calibrationEligible,
			unresolvedCalibrationMethods: [...unit.unresolvedCalibrationMethods],
			status: !unit.calibrationEligible
				? 'calibration-unresolved'
				: !enough
					? 'insufficient-candidates'
					: !resolved[unit.id]
						? 'test-unresolved'
						: 'scored',
			strengths: {
				...unit.strengths,
				testFullEffectRms: queryRms(testFullEffects[unit.id]),
				testQueryEffectRms: queryRms(testQueryEffects[unit.id])
			},
			candidatePool: [...unit.candidatePool],
			candidatePoolSize: unit.candidatePool.length,
			randomScore,
			randomResolvedNeighbors,
			randomCoverage: unit.candidatePool.length
				? randomResolvedNeighbors / unit.candidatePool.length
				: null,
			shuffledRandomScore,
			methods: byMethod((method) => {
				const selection = unit.selections[method];
				const score = scored
					? mean(selection.neighbors.map((other) => similarity(unit.id, other)))
					: null;
				const shuffledScore = scored
					? mean(selection.neighbors.map((other) => shuffledSimilarity(unit.id, other)))
					: null;
				const resolvedTestNeighbors = selection.neighbors.filter((other) => resolved[other]).length;
				return {
					neighbors: [...selection.neighbors],
					calibrationMeanCosine: selection.meanCosine,
					score,
					delta: score === null ? null : score - randomScore!,
					resolvedTestNeighbors,
					coverage: selection.neighbors.length
						? resolvedTestNeighbors / selection.neighbors.length
						: null,
					shuffledScore,
					shuffledDelta: shuffledScore === null ? null : shuffledScore - shuffledRandomScore!
				};
			})
		};
	});
	return {
		units,
		overall: summarizeUnits(units),
		layers: [...new Set(units.map((unit) => unit.layer))]
			.sort((a, b) => a - b)
			.map((layer) => ({ layer, ...summarizeUnits(units.filter((unit) => unit.layer === layer)) })),
		shuffledTestIdentities
	};
}

function validateMeasurement(measurement: QueryMeasurement) {
	requireCondition(
		measurement.version === 1 && measurement.design === 'paired-query-v1',
		'measurement version'
	);
	requireCondition(
		measurement.checkpointPreserved === true &&
			measurement.intervention === 'zero-all-token-positions',
		'measurement provenance'
	);
	requireCondition(
		Number.isInteger(measurement.seed) &&
			Number.isInteger(measurement.step) &&
			measurement.step >= 0,
		'seed and checkpoint'
	);
	const assignments = new Set<string>();
	for (const split of ['calibration', 'test'] as const) {
		const groups = measurement.groups[split];
		requireCondition(
			groups.length === QUERY_ANALYSIS_DESIGN.groupsPerSplit,
			`${split} group count`
		);
		for (const group of groups) {
			const key = JSON.stringify(group.assignment);
			requireCondition(!assignments.has(key), 'assignment groups must be unique and disjoint');
			assignments.add(key);
			requireCondition(group.examples.length === 3, 'three queries per group');
			const prefix = group.examples[0].tokenIds.slice(0, -1).join(',');
			group.examples.forEach((example, query) => {
				requireCondition(
					example.query === ['a', 'b', 'c'][query] && example.split === split,
					'fixed query order and split'
				);
				requireCondition(
					example.tokenIds.length === MODEL_CONFIG.sequenceLength &&
						example.tokenIds.slice(0, -1).join(',') === prefix,
					'identical query prefixes'
				);
				requireCondition(example.tokenIds.at(-1) === query, 'final query token');
				requireCondition(JSON.stringify(example.assignment) === key, 'shared query assignment');
			});
		}
		const raw = measurement.splits[split];
		validateMatrix(raw.activations, NEURON_COUNT, groups.length * 3, `${split} activations`);
		validateMatrix(raw.effects, NEURON_COUNT, groups.length * 3 * 8, `${split} effects`);
	}
	validateMatrix(measurement.outgoingWeights, NEURON_COUNT, MODEL_CONFIG.width, 'outgoing weights');
}

export function analyzeQueryShifts(measurement: QueryMeasurement): QueryAnalysis {
	validateMeasurement(measurement);
	const calibration = measurement.splits.calibration;
	const test = measurement.splits.test;
	const groups = QUERY_ANALYSIS_DESIGN.groupsPerSplit;
	const calibrationEffects = helmertQueryContrasts(calibration.effects, groups, 8);
	const testEffects = helmertQueryContrasts(test.effects, groups, 8);
	const plan = buildQueryCalibrationPlan({
		layers: Array.from({ length: NEURON_COUNT }, (_, id) => Math.floor(id / MODEL_CONFIG.hidden)),
		fullEffects: calibration.effects,
		queryEffects: calibrationEffects,
		queryActivations: helmertQueryContrasts(calibration.activations, groups, 1),
		outgoingWeights: measurement.outgoingWeights
	});
	const geometry = buildGeometry(
		calibrationEffects.map((row) =>
			queryRms(row) > QUERY_ANALYSIS_DESIGN.rmsFloor ? row : row.map(() => 0)
		),
		{ kind: 'effect', neighbors: QUERY_ANALYSIS_DESIGN.neighbors, normalizeEffects: true }
	);
	const calibrationGeometry: QueryGeometry = {
		kind: geometry.kind,
		positions: geometry.positions,
		edges: geometry.edges,
		magnitudes: geometry.magnitudes,
		valid: geometry.valid,
		neighborCount: geometry.neighborCount,
		neighborRetention: geometry.neighborRetention,
		explainedVariance: geometry.explainedVariance,
		explainedVarianceByAxis: geometry.explainedVarianceByAxis,
		rank: geometry.rank,
		boundaryDegenerate: geometry.boundaryDegenerate,
		alignment: geometry.alignment
	};
	return {
		version: 1,
		design: { ...QUERY_ANALYSIS_DESIGN },
		seed: measurement.seed,
		step: measurement.step,
		checkpointHash: measurement.checkpointHash,
		measurementCapturedAt: measurement.capturedAt,
		...scoreQueryCalibrationPlan(plan, testEffects, test.effects, measurement.seed),
		calibrationGeometry
	};
}
