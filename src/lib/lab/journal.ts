import type { Atlas, Checkpoint, Initialization, Metrics, Probe, RepairResult } from './protocol';
import { MODEL_CONFIG as cfg, NEURON_COUNT } from './protocol';
import { CALIBRATION_EXAMPLES, TEST_EXAMPLES, assignmentSplits } from './model/dataset';

export type Point3 = [number, number, number];

export interface MapFrame {
	positions: Point3[];
	edges: [number, number][];
	magnitudes: number[];
	valid: boolean[];
	neighbors: number[][];
	neighborRetention: number | null;
	explainedVariance: number;
	rank?: number;
	boundaryDegenerate?: boolean;
	alignment?: { applied: boolean; rmsDisplacement: number | null };
}

export interface Observation {
	id: string;
	time: string;
	step: number;
	kind: 'training' | 'measurement' | 'intervention' | 'note' | 'failure';
	title: string;
	detail: string;
	probe?: Probe;
}

export interface Snapshot {
	step: number;
	metrics: Metrics;
	activation: MapFrame;
	effect?: MapFrame;
	probe: Probe;
}

export interface RunRecord {
	version: 1;
	id: string;
	createdAt: string;
	updatedAt: string;
	title: string;
	seed: number;
	initialization: Initialization;
	metrics: Metrics[];
	snapshots: Snapshot[];
	observations: Observation[];
	repairs?: RepairResult[];
	checkpoint?: Checkpoint;
	/** Raw measurements retained for independent reanalysis of the latest atlas. */
	atlas?: Atlas;
	provenance: { source: 'browser' | 'reference'; appVersion: string; userAgent: string };
}

const DATABASE = 'tissue-lab';
const STORE = 'runs';
let journalWarnings: string[] = [];

/** Invalid rows stay in IndexedDB for recovery, but never enter the rendered archive. */
export function getJournalWarnings(): string[] {
	return [...journalWarnings];
}

function openDatabase(): Promise<IDBDatabase> {
	return new Promise((resolve, reject) => {
		const request = indexedDB.open(DATABASE, 1);
		request.onupgradeneeded = () => {
			if (!request.result.objectStoreNames.contains(STORE)) {
				request.result.createObjectStore(STORE, { keyPath: 'id' });
			}
		};
		request.onsuccess = () => resolve(request.result);
		request.onerror = () =>
			reject(request.error ?? new Error('Could not open the experiment journal.'));
	});
}

export async function saveRun(run: RunRecord): Promise<void> {
	validateRun(run);
	const database = await openDatabase();
	try {
		await new Promise<void>((resolve, reject) => {
			const transaction = database.transaction(STORE, 'readwrite');
			transaction.objectStore(STORE).put(run);
			transaction.oncomplete = () => resolve();
			transaction.onerror = () =>
				reject(transaction.error ?? new Error('Could not save this run.'));
			transaction.onabort = () => reject(transaction.error ?? new Error('Saving was interrupted.'));
		});
	} finally {
		database.close();
	}
}

export async function readRuns(): Promise<RunRecord[]> {
	journalWarnings = [];
	const database = await openDatabase();
	try {
		return await new Promise((resolve, reject) => {
			const request = database.transaction(STORE).objectStore(STORE).getAll();
			request.onsuccess = () => {
				const valid: RunRecord[] = [];
				for (const [index, candidate] of (request.result as unknown[]).entries()) {
					try {
						valid.push(validateRun(candidate));
					} catch (error) {
						journalWarnings.push(
							`Local record ${index + 1} was excluded from the archive: ${error instanceof Error ? error.message : String(error)}. Its stored data was preserved.`
						);
					}
				}
				resolve(valid.sort((a, b) => Date.parse(b.createdAt) - Date.parse(a.createdAt)));
			};
			request.onerror = () => reject(request.error ?? new Error('Could not read saved runs.'));
		});
	} finally {
		database.close();
	}
}

export function exportRun(run: RunRecord): void {
	validateRun(run);
	const blob = new Blob([JSON.stringify(run)], { type: 'application/json' });
	const url = URL.createObjectURL(blob);
	const link = document.createElement('a');
	link.href = url;
	link.download = `tissue-seed-${run.seed}-step-${run.checkpoint?.step ?? 0}.json`;
	link.click();
	setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function parseRun(text: string): RunRecord {
	if (text.length > 100_000_000) throw new Error('This run exceeds the 100 MB import limit.');
	return validateRun(JSON.parse(text));
}

type RecordValue = Record<string, unknown>;
const ANSWERS = cfg.vocabulary.length - 6;
const UINT32 = 0xffffffff;

function requireValue(condition: unknown, path: string, message: string): asserts condition {
	if (!condition) throw new Error(`Invalid experiment at ${path}: ${message}`);
}
function object(value: unknown, path: string): RecordValue {
	requireValue(
		value !== null && typeof value === 'object' && !Array.isArray(value),
		path,
		'expected an object'
	);
	return value as RecordValue;
}
function array(value: unknown, path: string, length?: number): unknown[] {
	requireValue(Array.isArray(value), path, 'expected an array');
	if (length !== undefined)
		requireValue(value.length === length, path, `expected ${length} entries`);
	return value;
}
function number(value: unknown, path: string, min = -Infinity, max = Infinity): number {
	requireValue(
		typeof value === 'number' && Number.isFinite(value),
		path,
		'expected a finite number'
	);
	requireValue(value >= min && value <= max, path, `outside the allowed range [${min}, ${max}]`);
	return value;
}
function integer(value: unknown, path: string, min = 0, max = Number.MAX_SAFE_INTEGER): number {
	const result = number(value, path, min, max);
	requireValue(Number.isSafeInteger(result), path, 'expected an integer');
	return result;
}
function string(value: unknown, path: string, allowEmpty = false): string {
	requireValue(
		typeof value === 'string' && (allowEmpty || value.length > 0),
		path,
		'expected text'
	);
	return value;
}
function date(value: unknown, path: string): void {
	const text = string(value, path);
	const parts =
		/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/.exec(text);
	requireValue(parts && Number.isFinite(Date.parse(text)), path, 'expected an ISO timestamp');
	const year = Number(parts[1]);
	const month = Number(parts[2]);
	const day = Number(parts[3]);
	const leap = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
	const days = [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
	requireValue(
		month >= 1 &&
			month <= 12 &&
			day >= 1 &&
			day <= days[month - 1] &&
			Number(parts[4]) <= 23 &&
			Number(parts[5]) <= 59 &&
			Number(parts[6]) <= 59,
		path,
		'timestamp has invalid calendar fields'
	);
}
function choice(value: unknown, choices: readonly string[], path: string): string {
	const result = string(value, path);
	requireValue(choices.includes(result), path, 'unrecognized value');
	return result;
}
function matrix(
	value: unknown,
	rows: number,
	columns: number,
	path: string,
	min = -Infinity,
	max = Infinity
): void {
	array(value, path, rows).forEach((row, i) =>
		array(row, `${path}[${i}]`, columns).forEach((entry, j) =>
			number(entry, `${path}[${i}][${j}]`, min, max)
		)
	);
}
function unique(values: readonly unknown[], path: string): void {
	requireValue(new Set(values).size === values.length, path, 'duplicate identities');
}
function orderedSteps(values: number[], path: string): void {
	requireValue(
		values.every((step, i) => i === 0 || step > values[i - 1]),
		path,
		'steps must be unique and increasing'
	);
}

/** Shapes follow initializeParameters insertion order, the v1 serialized tree contract. */
const parameterShapes: number[][] = [
	[cfg.vocabulary.length, cfg.width],
	[cfg.sequenceLength, cfg.width],
	[cfg.width, cfg.vocabulary.length],
	...Array.from({ length: cfg.layers }, () => [
		[cfg.width, cfg.width],
		[cfg.width, cfg.width],
		[cfg.width, cfg.width],
		[cfg.width, cfg.width],
		[cfg.width, cfg.hidden],
		[cfg.hidden, cfg.width]
	]).flat()
];
const parameterCount = parameterShapes.reduce((sum, shape) => sum + shape[0] * shape[1], 0);

function validateMetrics(value: unknown, path: string): number {
	const metric = object(value, path);
	const step = integer(metric.step, `${path}.step`);
	if (metric.trainLoss !== null) number(metric.trainLoss, `${path}.trainLoss`, 0);
	number(metric.validationLoss, `${path}.validationLoss`, 0);
	number(metric.accuracy, `${path}.accuracy`, 0, 1);
	number(metric.elapsedMs, `${path}.elapsedMs`, 0);
	choice(metric.backend, ['webgpu', 'wasm', 'cpu'], `${path}.backend`);
	requireValue(
		metric.validationExamples === TEST_EXAMPLES.length,
		`${path}.validationExamples`,
		'evaluation set does not match architecture v1'
	);
	requireValue(
		metric.chanceAccuracy === 1 / ANSWERS,
		`${path}.chanceAccuracy`,
		'baseline does not match answer vocabulary'
	);
	return step;
}

function validateExample(value: unknown, index: number, path: string): void {
	const example = object(value, path);
	const expected = CALIBRATION_EXAMPLES[index];
	requireValue(expected !== undefined, path, 'calibration index is out of range');
	for (const key of ['id', 'answer', 'answerId', 'query', 'split'] as const) {
		requireValue(
			example[key] === expected[key],
			`${path}.${key}`,
			'does not match the fixed calibration prompt'
		);
	}
	for (const key of ['tokens', 'tokenIds', 'assignment'] as const) {
		const entries = array(example[key], `${path}.${key}`, expected[key].length);
		requireValue(
			entries.every((entry, i) => entry === expected[key][i]),
			`${path}.${key}`,
			'does not match the fixed calibration prompt'
		);
	}
}

function validateProbe(value: unknown, path: string, expectedStep?: number): void {
	const probe = object(value, path);
	const step = integer(probe.step, `${path}.step`);
	if (expectedStep !== undefined)
		requireValue(
			step === expectedStep,
			`${path}.step`,
			'probe and observation checkpoint disagree'
		);
	const index = integer(
		probe.exampleIndex,
		`${path}.exampleIndex`,
		0,
		CALIBRATION_EXAMPLES.length - 1
	);
	validateExample(probe.example, index, `${path}.example`);
	const probabilities = array(probe.probabilities, `${path}.probabilities`, ANSWERS).map((p, i) =>
		number(p, `${path}.probabilities[${i}]`, 0, 1)
	);
	const other = number(probe.otherProbability, `${path}.otherProbability`, 0, 1);
	requireValue(
		Math.abs(probabilities.reduce((sum, p) => sum + p, other) - 1) <= 0.001,
		`${path}.probabilities`,
		'probability mass must sum to one'
	);
	choice(probe.predictedAnswer, cfg.vocabulary, `${path}.predictedAnswer`);
	matrix(probe.activations, cfg.sequenceLength, NEURON_COUNT, `${path}.activations`, 0);
	if (probe.lesionNeuron !== null)
		integer(probe.lesionNeuron, `${path}.lesionNeuron`, 0, NEURON_COUNT - 1);
}

function validateMap(value: unknown, path: string): void {
	const map = object(value, path);
	matrix(map.positions, NEURON_COUNT, 3, `${path}.positions`);
	array(map.magnitudes, `${path}.magnitudes`, NEURON_COUNT).forEach((v, i) =>
		number(v, `${path}.magnitudes[${i}]`, 0)
	);
	const valid = array(map.valid, `${path}.valid`, NEURON_COUNT);
	valid.forEach((v, i) =>
		requireValue(typeof v === 'boolean', `${path}.valid[${i}]`, 'expected a boolean')
	);
	array(map.edges, `${path}.edges`).forEach((value, i) => {
		const pair = array(value, `${path}.edges[${i}]`, 2).map((v) =>
			integer(v, `${path}.edges[${i}]`, 0, NEURON_COUNT - 1)
		);
		requireValue(
			pair[0] !== pair[1] && valid[pair[0]] && valid[pair[1]],
			`${path}.edges[${i}]`,
			'edges must join distinct resolved neurons'
		);
	});
	array(map.neighbors, `${path}.neighbors`, NEURON_COUNT).forEach((value, i) => {
		const neighbors = array(value, `${path}.neighbors[${i}]`);
		unique(neighbors, `${path}.neighbors[${i}]`);
		neighbors.forEach((v) => {
			const index = integer(v, `${path}.neighbors[${i}]`, 0, NEURON_COUNT - 1);
			requireValue(
				valid[i] && valid[index] && index !== i,
				`${path}.neighbors[${i}]`,
				'neighbors must be distinct resolved neurons'
			);
		});
	});
	if (map.neighborRetention !== null)
		number(map.neighborRetention, `${path}.neighborRetention`, 0, 1);
	number(map.explainedVariance, `${path}.explainedVariance`, 0, 1 + 1e-10);
	if (map.rank !== undefined) integer(map.rank, `${path}.rank`, 0, NEURON_COUNT - 1);
	if (map.boundaryDegenerate !== undefined)
		requireValue(
			typeof map.boundaryDegenerate === 'boolean',
			`${path}.boundaryDegenerate`,
			'expected a boolean'
		);
	if (map.alignment !== undefined) {
		const alignment = object(map.alignment, `${path}.alignment`);
		requireValue(
			typeof alignment.applied === 'boolean',
			`${path}.alignment.applied`,
			'expected a boolean'
		);
		if (alignment.rmsDisplacement !== null)
			number(alignment.rmsDisplacement, `${path}.alignment.rmsDisplacement`, 0);
	}
}

function validateLeaves(value: unknown, path: string, secondMoment = false): void {
	array(value, path, parameterShapes.length).forEach((entry, i) => {
		const leaf = object(entry, `${path}[${i}]`);
		const shape = array(leaf.shape, `${path}[${i}].shape`, 2);
		requireValue(
			shape.every((v, j) => v === parameterShapes[i][j]),
			`${path}[${i}].shape`,
			'does not match architecture v1'
		);
		array(
			leaf.values,
			`${path}[${i}].values`,
			parameterShapes[i][0] * parameterShapes[i][1]
		).forEach((v, j) => {
			number(v, `${path}[${i}].values[${j}]`, secondMoment ? 0 : -Infinity);
			requireValue(
				Number.isFinite(Math.fround(v as number)),
				`${path}[${i}].values[${j}]`,
				'not representable as a finite Float32'
			);
		});
	});
}

function validateCheckpoint(value: unknown, seed: number): number {
	const checkpoint = object(value, 'checkpoint');
	requireValue(
		checkpoint.version === 1 && checkpoint.architecture === 'binding-transformer-v1',
		'checkpoint',
		'unsupported checkpoint architecture'
	);
	requireValue(checkpoint.seed === seed, 'checkpoint.seed', 'does not match run seed');
	const step = integer(checkpoint.step, 'checkpoint.step');
	integer(checkpoint.randomState, 'checkpoint.randomState', 0, UINT32);
	number(checkpoint.elapsedMs, 'checkpoint.elapsedMs', 0);
	if (checkpoint.trainLoss !== null) number(checkpoint.trainLoss, 'checkpoint.trainLoss', 0);
	validateLeaves(checkpoint.parameters, 'checkpoint.parameters');
	const optimizer = object(checkpoint.optimizer, 'checkpoint.optimizer');
	requireValue(
		optimizer.t === step,
		'checkpoint.optimizer.t',
		'optimizer and model steps disagree'
	);
	validateLeaves(optimizer.m, 'checkpoint.optimizer.m');
	validateLeaves(optimizer.v, 'checkpoint.optimizer.v', true);
	return step;
}

function validateAtlas(value: unknown, seed: number, checkpointStep: number): void {
	const atlas = object(value, 'atlas');
	requireValue(
		atlas.seed === seed && atlas.step === checkpointStep,
		'atlas',
		'raw measurements and durable checkpoint disagree'
	);
	date(atlas.capturedAt, 'atlas.capturedAt');
	array(atlas.neurons, 'atlas.neurons', NEURON_COUNT).forEach((value, i) => {
		const neuron = object(value, `atlas.neurons[${i}]`);
		requireValue(
			neuron.id === i &&
				neuron.layer === Math.floor(i / cfg.hidden) &&
				neuron.channel === i % cfg.hidden,
			`atlas.neurons[${i}]`,
			'neuron order does not match architecture'
		);
	});
	array(atlas.examples, 'atlas.examples', CALIBRATION_EXAMPLES.length).forEach((example, i) =>
		validateExample(example, i, `atlas.examples[${i}]`)
	);
	matrix(
		atlas.activationFingerprints,
		NEURON_COUNT,
		CALIBRATION_EXAMPLES.length * cfg.sequenceLength,
		'atlas.activationFingerprints',
		0
	);
	matrix(atlas.outgoingWeights, NEURON_COUNT, cfg.width, 'atlas.outgoingWeights');
	number(atlas.calibrationAccuracy, 'atlas.calibrationAccuracy', 0, 1);
	if (atlas.effectFingerprints !== undefined) {
		requireValue(
			atlas.intervention === 'zero-all-token-positions',
			'atlas.intervention',
			'measured effects need their intervention definition'
		);
		matrix(
			atlas.effectFingerprints,
			NEURON_COUNT,
			CALIBRATION_EXAMPLES.length * ANSWERS,
			'atlas.effectFingerprints',
			-1,
			1
		);
	} else
		requireValue(
			atlas.intervention === 'not-measured',
			'atlas.intervention',
			'effect measurements are missing'
		);
}

function validateRepair(value: unknown, seed: number, path: string, checkpointStep?: number): void {
	const repair = object(value, path);
	const step = integer(repair.step, `${path}.step`);
	requireValue(
		checkpointStep !== undefined && step <= checkpointStep && repair.seed === seed,
		path,
		'repair and durable checkpoint disagree'
	);
	const lesion = integer(repair.lesionNeuron, `${path}.lesionNeuron`, 0, NEURON_COUNT - 1);
	const steps = integer(repair.steps, `${path}.steps`, 1, 250);
	const count = integer(repair.neighborCount, `${path}.neighborCount`, 1, 32);
	const observation = (value: unknown, path: string) => {
		const entry = object(value, path);
		integer(entry.update, `${path}.update`, 0, steps);
		number(entry.loss, `${path}.loss`, 0);
		number(entry.accuracy, `${path}.accuracy`, 0, 1);
		number(entry.meanAnswerProbability, `${path}.meanAnswerProbability`, 0, 1);
		return entry.update as number;
	};
	requireValue(
		observation(repair.intact, `${path}.intact`) === 0,
		`${path}.intact`,
		'baseline update must be zero'
	);
	requireValue(
		observation(repair.lesioned, `${path}.lesioned`) === 0,
		`${path}.lesioned`,
		'baseline update must be zero'
	);
	const methods = ['activation', 'intervention', 'outgoing-weight', 'random'];
	const arms = array(repair.arms, `${path}.arms`, methods.length);
	const actualMethods = arms.map((value, i) => {
		const armPath = `${path}.arms[${i}]`;
		const arm = object(value, armPath);
		const method = choice(arm.method, methods, `${armPath}.method`);
		const neurons = array(arm.neurons, `${armPath}.neurons`, count).map((n) =>
			integer(n, `${armPath}.neurons`, 0, NEURON_COUNT - 1)
		);
		unique(neurons, `${armPath}.neurons`);
		requireValue(
			neurons.every(
				(n) => n !== lesion && Math.floor(n / cfg.hidden) === Math.floor(lesion / cfg.hidden)
			),
			`${armPath}.neurons`,
			'repair neurons must be surviving units in the lesioned layer'
		);
		requireValue(
			arm.trainableParameters === count * cfg.width * 2,
			`${armPath}.trainableParameters`,
			'repair budgets do not match'
		);
		const updates = array(arm.curve, `${armPath}.curve`).map((entry, j) =>
			observation(entry, `${armPath}.curve[${j}]`)
		);
		requireValue(
			updates[0] === 0 && updates.at(-1) === steps,
			`${armPath}.curve`,
			'repair curve must include both endpoints'
		);
		orderedSteps(updates, `${armPath}.curve`);
		return method;
	});
	unique(actualMethods, `${path}.arms`);
	requireValue(
		repair.testExamples === TEST_EXAMPLES.length,
		`${path}.testExamples`,
		'test set does not match architecture'
	);
	array(repair.testPredictions, `${path}.testPredictions`, TEST_EXAMPLES.length).forEach(
		(value, i) => {
			const predictionPath = `${path}.testPredictions[${i}]`;
			const prediction = object(value, predictionPath);
			const example = TEST_EXAMPLES[i];
			requireValue(
				prediction.id === example.id &&
					prediction.prompt === example.tokens.join('') &&
					prediction.answer === example.answer,
				predictionPath,
				'does not match the fixed evaluation prompt'
			);
			number(prediction.intactProbability, `${predictionPath}.intactProbability`, 0, 1);
			number(prediction.lesionedProbability, `${predictionPath}.lesionedProbability`, 0, 1);
			const predictions = array(
				prediction.armProbabilities,
				`${predictionPath}.armProbabilities`,
				methods.length
			);
			const predictionMethods = predictions.map((value, j) => {
				const item = object(value, `${predictionPath}.armProbabilities[${j}]`);
				number(item.probability, `${predictionPath}.armProbabilities[${j}].probability`, 0, 1);
				return choice(item.method, methods, `${predictionPath}.armProbabilities[${j}].method`);
			});
			unique(predictionMethods, `${predictionPath}.armProbabilities`);
		}
	);
	date(repair.completedAt, `${path}.completedAt`);
	string(repair.caveat, `${path}.caveat`);
}

/** Complete structural checks without loading JaxJS or allocating model tensors. */
export function validateRun(value: unknown): RunRecord {
	const run = object(value, 'run');
	requireValue(run.version === 1, 'version', 'unsupported experiment version');
	string(run.id, 'id');
	string(run.title, 'title');
	date(run.createdAt, 'createdAt');
	date(run.updatedAt, 'updatedAt');
	const seed = integer(run.seed, 'seed', 0, UINT32);
	const initialization = object(run.initialization, 'initialization');
	requireValue(initialization.seed === seed, 'initialization.seed', 'does not match run seed');
	const backend = choice(
		initialization.backend,
		['webgpu', 'wasm', 'cpu'],
		'initialization.backend'
	);
	requireValue(
		initialization.parameterCount === parameterCount,
		'initialization.parameterCount',
		'does not match architecture v1'
	);
	requireValue(
		validateMetrics(initialization.metrics, 'initialization.metrics') === 0,
		'initialization.metrics.step',
		'initial observation must be step zero'
	);
	requireValue(
		object(initialization.metrics, 'initialization.metrics').backend === backend,
		'initialization.backend',
		'does not match initial metrics'
	);
	for (const split of ['train', 'calibration', 'test'] as const) {
		requireValue(
			initialization[`${split}Assignments`] === assignmentSplits[split].length,
			`initialization.${split}Assignments`,
			'does not match assignment splits'
		);
	}
	const metrics = array(run.metrics, 'metrics');
	requireValue(metrics.length > 0, 'metrics', 'at least one measured checkpoint is required');
	const metricSteps = metrics.map((metric, i) => validateMetrics(metric, `metrics[${i}]`));
	orderedSteps(metricSteps, 'metrics');
	const provenance = object(run.provenance, 'provenance');
	choice(provenance.source, ['browser', 'reference'], 'provenance.source');
	string(provenance.appVersion, 'provenance.appVersion');
	string(provenance.userAgent, 'provenance.userAgent', true);
	const checkpointStep =
		run.checkpoint === undefined ? undefined : validateCheckpoint(run.checkpoint, seed);
	if (checkpointStep !== undefined)
		requireValue(
			metricSteps.includes(checkpointStep),
			'metrics',
			'durable checkpoint has no matching metric'
		);
	const snapshots = array(run.snapshots, 'snapshots');
	const snapshotSteps = snapshots.map((value, i) => {
		const path = `snapshots[${i}]`;
		const snapshot = object(value, path);
		const step = integer(snapshot.step, `${path}.step`);
		requireValue(
			checkpointStep !== undefined && step <= checkpointStep,
			`${path}.step`,
			'snapshot is newer than durable model weights'
		);
		requireValue(
			metricSteps.includes(step) && validateMetrics(snapshot.metrics, `${path}.metrics`) === step,
			`${path}.metrics`,
			'snapshot and metrics disagree'
		);
		validateMap(snapshot.activation, `${path}.activation`);
		if (snapshot.effect !== undefined) validateMap(snapshot.effect, `${path}.effect`);
		validateProbe(snapshot.probe, `${path}.probe`, step);
		requireValue(
			object(snapshot.probe, `${path}.probe`).lesionNeuron === null,
			`${path}.probe`,
			'checkpoint probe must be intact'
		);
		return step;
	});
	orderedSteps(snapshotSteps, 'snapshots');
	if (run.atlas !== undefined) {
		requireValue(
			checkpointStep !== undefined,
			'atlas',
			'raw measurements need durable model weights'
		);
		validateAtlas(run.atlas, seed, checkpointStep);
	}
	const observations = array(run.observations, 'observations');
	const observationIds = observations.map((value, i) => {
		const path = `observations[${i}]`;
		const observation = object(value, path);
		const id = string(observation.id, `${path}.id`);
		date(observation.time, `${path}.time`);
		const step = integer(observation.step, `${path}.step`);
		choice(
			observation.kind,
			['training', 'measurement', 'intervention', 'note', 'failure'],
			`${path}.kind`
		);
		string(observation.title, `${path}.title`);
		string(observation.detail, `${path}.detail`, true);
		if (observation.probe !== undefined) validateProbe(observation.probe, `${path}.probe`, step);
		return id;
	});
	unique(observationIds, 'observations');
	if (run.repairs !== undefined)
		array(run.repairs, 'repairs').forEach((repair, i) =>
			validateRepair(repair, seed, `repairs[${i}]`, checkpointStep)
		);
	// Later metrics and failure notes are observations, not resumable weights. They
	// remain evidence; the resume flow must explicitly reconcile to checkpoint.step.
	return value as RunRecord;
}
