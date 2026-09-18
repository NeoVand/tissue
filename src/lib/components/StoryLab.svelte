<script lang="ts">
	import type { Snippet } from 'svelte';
	import ResearchWorkspace from './ResearchWorkspace.svelte';
	import { onMount } from 'svelte';
	import { publicAsset } from '$lib/deployment/public-assets';
	import { StoryEngine } from '$lib/stories/engine';
	import { StoryGeometryEngine } from '$lib/stories/geometry-engine';
	import type { StoryNeighbor } from '$lib/stories/geometry';
	import {
		STORY_PRESETS,
		STORY_CHARACTERS,
		storyParameterCount,
		storyUnitCount,
		type StoryBackend,
		type StoryPreset,
		type StoryProbe,
		type StoryGeneration,
		type StoryMetrics,
		type StoryEvent
	} from '$lib/stories/protocol';
	import {
		listStoryRuns,
		loadStoryRun,
		saveStoryRun,
		importStoryRun,
		exportStoryRun,
		loadStoryReference,
		getStoryArchiveWarnings,
		type StoryRunRecord,
		type StoryRunSummary,
		type StoryReference
	} from '$lib/stories/archive';
	import Icon from './Icon.svelte';
	import NeuralField from './NeuralField.svelte';
	import StoryMetricsPanel from './StoryMetrics.svelte';
	import StoryProbePanel from './StoryProbe.svelte';

	let {
		theme,
		active = true,
		disabled = false,
		onbusy,
		representation
	}: {
		representation?: Snippet;
		theme: 'dark' | 'light';
		active?: boolean;
		disabled?: boolean;
		onbusy?: (busy: boolean) => void;
	} = $props();
	let toolsCollapsed = $state(false);
	let setupOpen = $state(false);
	let toolPanel = $state<'model' | 'inspector'>('model');
	type Phase =
		| 'idle'
		| 'initializing'
		| 'ready'
		| 'archived'
		| 'training'
		| 'measuring'
		| 'probing'
		| 'generating'
		| 'loading'
		| 'saving'
		| 'error';
	let phase = $state<Phase>('idle');
	let status = $state('Choose a model size, then initialize a new run or open a saved specimen.');
	let error = $state('');
	let storageWarning = $state('');
	let preset = $state<StoryPreset>('small');
	let seed = $state(42);
	let backend = $state<StoryBackend | 'auto'>('auto');
	let budget = $state(100);
	let record = $state.raw<StoryRunRecord | null>(null);
	let runs = $state.raw<StoryRunSummary[]>([]);
	let references = $state.raw<StoryReference[]>([]);
	let runtimeReady = $state(false);
	let residentModelId = $state<string | null>(null);
	let savedCheckpointStep = $state<number | null>(null);
	let snapshotIndex = $state<number | null>(null);
	let selected = $state<number | null>(null);
	let layerFilter = $state<number | null>(null);
	let viewMode = $state<'functional' | 'architecture'>('functional');
	let token = $state(0);
	let probePrompt = $state('Once upon a time, there was a little');
	let probe = $state.raw<StoryProbe | null>(null);
	let lesioned = $state.raw<StoryProbe | null>(null);
	let nearest = $state.raw<StoryNeighbor[]>([]);
	let samplePrompt = $state('Once upon a time');
	let samplingSeed = $state(7);
	let temperature = $state(0.8);
	let sampleLength = $state(64);
	let sample = $state.raw<StoryGeneration | null>(null);
	let trainingTarget = $state<number | null>(null);
	let stopRequested = $state(false);
	let measurementProgress = $state({ completed: 0, total: 0 });
	let footerTab = $state<'samples' | 'history'>('samples');
	let importInput: HTMLInputElement | undefined;
	let engine: StoryEngine | null = null;
	let geometryEngine: StoryGeometryEngine | null = null;
	let mounted = false;
	let lifecycle = 0;
	let neighborRevision = 0;
	let trainCallActive = false;

	const presetIds: StoryPreset[] = ['small', 'medium', 'large'];
	let busy = $derived(!['idle', 'ready', 'archived', 'error'].includes(phase));
	let blocked = $derived(busy || disabled);
	let config = $derived(STORY_PRESETS[record?.presetId ?? preset]);
	let unitCount = $derived(storyUnitCount(config));
	let latestMetric = $derived(record?.metrics.at(-1));
	let snapshot = $derived(
		record ? record.snapshots[snapshotIndex ?? record.snapshots.length - 1] : undefined
	);
	let historical = $derived(snapshotIndex !== null && snapshot?.atlas.step !== latestMetric?.step);
	let matchingProbe = $derived(
		probe &&
			snapshot &&
			probe.step === snapshot.atlas.step &&
			probe.modelId === snapshot.atlas.modelId
			? probe
			: null
	);
	let liveProbe = $derived(
		!historical &&
			runtimeReady &&
			probe?.modelId === residentModelId &&
			probe?.step === latestMetric?.step
			? probe
			: null
	);
	let ready = $derived(runtimeReady && !blocked && !historical);
	let points = $derived.by(() => {
		if (!snapshot) return [];
		const { atlas, geometry } = snapshot;
		const columns = 32;
		const rows = Math.ceil(config.hidden / columns);
		return Array.from({ length: atlas.unitCount }, (_, id) => {
			if (viewMode === 'functional' && !geometry.valid[id]) return null;
			const layer = Math.floor(id / config.hidden);
			const channel = id % config.hidden;
			const position: [number, number, number] =
				viewMode === 'functional'
					? geometry.positions[id]
					: [
							(layer / Math.max(1, config.layers - 1) - 0.5) * 2.8,
							(0.5 - Math.floor(channel / columns) / Math.max(1, rows - 1)) * 1.35,
							((channel % columns) / (columns - 1) - 0.5) * 1.1
						];
			const activation = matchingProbe
				? matchingProbe.activations[
						Math.min(token, matchingProbe.prompt.tokenIds.length - 1) * unitCount + id
					]
				: atlas.fingerprints[id * atlas.dimensions];
			return { id, position, layer, channel, activation };
		}).filter((point) => point !== null);
	});
	let edges = $derived<[number, number][]>(
		viewMode === 'functional' && selected !== null
			? nearest.map((neighbor) => [selected!, neighbor.index])
			: []
	);
	let corpus = $derived(record?.checkpoint?.corpus);
	let residentStep = $derived(runtimeReady ? latestMetric?.step : null);
	function setPhase(next: Phase): void {
		phase = next;
		onbusy?.(!['idle', 'ready', 'archived', 'error'].includes(next));
	}
	function current(ticket: number): boolean {
		return mounted && ticket === lifecycle;
	}
	function observe(title: string, detail: string): void {
		if (!record) return;
		record = {
			...record,
			observations: [
				...record.observations,
				{ time: new Date().toISOString(), step: latestMetric?.step ?? 0, title, detail }
			]
		};
	}
	function appendMetric(metric: StoryMetrics): void {
		if (!record) return;
		if (probe && probe.step !== metric.step) invalidateProbe();
		record = {
			...record,
			metrics: [...record.metrics.filter((entry) => entry.step !== metric.step), metric].sort(
				(a, b) => a.step - b.step
			)
		};
	}
	function handleEvent(event: StoryEvent, ticket: number): void {
		if (!current(ticket)) return;
		if (event.type === 'status') status = event.message;
		else if (event.type === 'metrics') appendMetric(event.metrics);
		else measurementProgress = { completed: event.completed, total: event.total };
	}
	async function refreshRuns(): Promise<void> {
		const next = await listStoryRuns();
		if (mounted) {
			runs = next;
			const warnings = getStoryArchiveWarnings();
			if (warnings.length) storageWarning = warnings.join(' ');
		}
	}
	async function persist(): Promise<boolean> {
		if (!record) return false;
		const next = { ...record, updatedAt: new Date().toISOString() };
		record = next;
		try {
			await saveStoryRun(next);
			if (!mounted) return false;
			savedCheckpointStep = next.checkpoint?.step ?? null;
			storageWarning = '';
			await refreshRuns();
			return true;
		} catch (reason) {
			if (mounted)
				storageWarning = `Local save failed: ${reason instanceof Error ? reason.message : String(reason)}. The in-memory evidence can still be exported.`;
			return false;
		}
	}
	async function disposeRuntime(): Promise<void> {
		const previous = engine;
		engine = null;
		runtimeReady = false;
		residentModelId = null;
		invalidateProbe();
		if (previous) await previous.dispose();
	}
	function getGeometry(): StoryGeometryEngine {
		return (geometryEngine ??= new StoryGeometryEngine());
	}
	async function updateNeighbors(): Promise<void> {
		const revision = ++neighborRevision;
		nearest = [];
		if (selected === null || !snapshot || viewMode !== 'functional') return;
		const atlas = snapshot.atlas;
		try {
			const result = await getGeometry().neighbors(selected, 6, layerFilter ?? undefined);
			if (
				mounted &&
				revision === neighborRevision &&
				snapshot?.atlas === atlas &&
				snapshot.atlas.modelId === atlas.modelId &&
				snapshot.atlas.step === atlas.step
			)
				nearest = result;
		} catch (reason) {
			if (mounted && revision === neighborRevision)
				error = `Neighbor lookup failed: ${reason instanceof Error ? reason.message : String(reason)}`;
		}
	}
	function selectUnit(id: number): void {
		if (id < 0 || id >= unitCount) return;
		selected = id;
		void updateNeighbors();
	}
	function selectLayer(layer: number | null): void {
		layerFilter = layer;
		if (layer !== null && (selected === null || Math.floor(selected / config.hidden) !== layer))
			selected = layer * config.hidden;
		void updateNeighbors();
	}
	function selectView(next: 'functional' | 'architecture'): void {
		viewMode = next;
		void updateNeighbors();
	}
	function invalidateProbe(): void {
		probe = null;
		lesioned = null;
	}
	async function probeCurrent(ticket: number): Promise<void> {
		invalidateProbe();
		if (!engine || !probePrompt.length) return;
		const next = await engine.probe(probePrompt);
		if (!current(ticket)) return;
		if (next.modelId !== residentModelId || next.step !== latestMetric?.step)
			throw new Error('Prompt probe does not match the resident checkpoint. Run the prompt again.');
		probe = next;
		lesioned = null;
		token = next.prompt.tokenIds.length - 1;
	}
	async function capture(ticket: number, includeWeights: boolean): Promise<void> {
		if (!engine || !record) return;
		neighborRevision++;
		nearest = [];
		status = 'Capturing all units on the fixed calibration windows…';
		const atlas = await engine.captureAtlas();
		if (!current(ticket)) return;
		status = 'Fitting measured activation geometry…';
		const geometry = await getGeometry().build(atlas);
		if (!current(ticket) || !record) return;
		neighborRevision++;
		nearest = [];
		record = {
			...record,
			snapshots: [
				...record.snapshots.filter((entry) => entry.atlas.step !== atlas.step),
				{ capturedAt: atlas.capturedAt, atlas, geometry }
			].sort((a, b) => a.atlas.step - b.atlas.step)
		};
		snapshotIndex = null;
		if (includeWeights) {
			const checkpoint = await engine.exportCheckpoint();
			if (!current(ticket) || !record) return;
			record = { ...record, checkpoint };
		}
		try {
			await probeCurrent(ticket);
		} catch (reason) {
			if (current(ticket))
				error = `Atlas recorded; prompt probe failed: ${reason instanceof Error ? reason.message : String(reason)}`;
		}
		if (!current(ticket)) return;
		await updateNeighbors();
		if (!current(ticket)) return;
		await persist();
	}
	async function initialize(): Promise<void> {
		if (blocked || !Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) return;
		const ticket = ++lifecycle;
		setPhase('initializing');
		error = '';
		storageWarning = '';
		stopRequested = false;
		try {
			await disposeRuntime();
			if (!current(ticket)) return;
			geometryEngine?.destroy();
			geometryEngine = null;
			neighborRevision++;
			const owner = new StoryEngine((event) => handleEvent(event, ticket));
			engine = owner;
			status = `Initializing ${(storyParameterCount(STORY_PRESETS[preset]) / 1e6).toFixed(3)}M parameters…`;
			const init = await owner.initialize(preset, seed, backend);
			if (!current(ticket)) return;
			const now = new Date().toISOString();
			record = {
				version: 1,
				kind: 'tissue-story-run',
				id: crypto.randomUUID(),
				createdAt: now,
				updatedAt: now,
				title: `TinyStories ${(init.parameterCount / 1e6).toFixed(3)}M · seed ${seed}`,
				seed,
				presetId: preset,
				metrics: [init.metrics],
				snapshots: [],
				checkpoint: null,
				observations: [],
				samples: [],
				provenance: { source: 'browser', browser: navigator.userAgent }
			};
			probe = null;
			lesioned = null;
			sample = null;
			snapshotIndex = null;
			selected = null;
			layerFilter = null;
			nearest = [];
			savedCheckpointStep = null;
			residentModelId = init.modelId;
			runtimeReady = true;
			observe(
				'Model initialized',
				`${init.parameterCount.toLocaleString()} parameters; ${init.unitCount.toLocaleString()} MLP channels; ${init.backend}. Corpus ${init.corpus.id}.`
			);
			setPhase('measuring');
			await capture(ticket, true);
			if (!current(ticket)) return;
			status = `Ready · ${init.backend.toUpperCase()} · step ${init.metrics.step}`;
			setPhase('ready');
		} catch (reason) {
			if (current(ticket)) await failTraining(reason);
		}
	}
	async function failTraining(reason: unknown): Promise<void> {
		error = reason instanceof Error ? reason.message : String(reason);
		observe(
			'Operation interrupted',
			`${error} Latest durable checkpoint: ${savedCheckpointStep ?? 'none'}. Later measurements remain in this record.`
		);
		await persist();
		await disposeRuntime();
		if (!mounted) return;
		status = 'Operation interrupted; recorded evidence retained';
		setPhase('error');
	}
	async function train(): Promise<void> {
		if (!ready || !engine || !record) return;
		const ticket = lifecycle;
		error = '';
		stopRequested = false;
		trainingTarget = (latestMetric?.step ?? 0) + budget;
		setPhase('training');
		try {
			while (!stopRequested && (latestMetric?.step ?? 0) < trainingTarget) {
				status = `Training toward step ${trainingTarget}…`;
				trainCallActive = true;
				const next = await engine
					.train(Math.min(25, trainingTarget - (latestMetric?.step ?? 0)))
					.finally(() => {
						trainCallActive = false;
					});
				if (!current(ticket)) return;
				appendMetric(next);
				await capture(ticket, record.presetId !== 'large');
				if (!current(ticket)) return;
			}
			if (record.presetId === 'large') {
				status = 'Saving model and optimizer checkpoint…';
				const checkpoint = await engine.exportCheckpoint();
				if (!current(ticket) || !record) return;
				record = { ...record, checkpoint };
			}
			observe(
				stopRequested ? 'Training paused' : 'Training interval completed',
				`At step ${latestMetric?.step}. Calibration atlases measured at burst boundaries, at most 25 updates apart.`
			);
			await persist();
			if (!current(ticket)) return;
			status = `Ready · step ${latestMetric?.step} · checkpoint ${record.checkpoint?.step ?? 'unavailable'}`;
			setPhase('ready');
		} catch (reason) {
			if (current(ticket)) await failTraining(reason);
		} finally {
			if (current(ticket)) trainingTarget = null;
		}
	}
	async function pause(): Promise<void> {
		stopRequested = true;
		status =
			phase === 'generating'
				? 'Stopping after the current sampled character…'
				: 'Pausing after the current worker operation, then recording evidence…';
		try {
			if (trainCallActive || phase === 'generating') await engine?.pause();
		} catch (reason) {
			if (mounted) error = String(reason);
		}
	}
	async function inspectPrompt(): Promise<void> {
		if (!ready) return;
		const ticket = lifecycle;
		setPhase('probing');
		error = '';
		try {
			await probeCurrent(ticket);
			if (current(ticket)) status = `Prompt measured at step ${probe?.step}`;
		} catch (reason) {
			if (current(ticket)) error = reason instanceof Error ? reason.message : String(reason);
		} finally {
			if (current(ticket)) setPhase('ready');
		}
	}
	async function silenceUnit(): Promise<void> {
		const intact = liveProbe;
		if (!ready || !engine || !intact || selected === null) return;
		const ticket = lifecycle;
		const id = selected;
		setPhase('probing');
		error = '';
		lesioned = null;
		status = `Measuring exact all-position ablation of unit ${id}…`;
		try {
			const result = await engine.probe(intact.prompt.original, id);
			if (!current(ticket)) return;
			if (
				result.modelId !== intact.modelId ||
				result.modelId !== residentModelId ||
				result.step !== intact.step ||
				result.step !== latestMetric?.step ||
				result.prompt.original !== intact.prompt.original ||
				result.prompt.text !== intact.prompt.text ||
				result.lesionNeuron !== id
			)
				throw new Error(
					'Ablation and intact probe do not match the same checkpoint and prompt. Nothing was recorded. Run the prompt again.'
				);
			lesioned = result;
			if (record)
				record = {
					...record,
					interventions: [
						...(record.interventions ?? []),
						{
							step: result.step,
							modelId: result.modelId,
							neuron: id,
							prompt: intact.prompt.text,
							capturedAt: new Date().toISOString(),
							probabilities: intact.probabilities.slice(),
							lesionedProbabilities: result.probabilities.slice()
						}
					]
				};
			observe(
				'Single-unit ablation measured',
				`Unit ${id}, L${Math.floor(id / config.hidden) + 1} channel ${id % config.hidden}, set to zero at all positions of prompt ${JSON.stringify(intact.prompt.text)}. Results shown beside intact predictions at step ${result.step}.`
			);
			await persist();
			if (current(ticket)) status = `Unit ${id} ablation measured at step ${result.step}`;
		} catch (reason) {
			if (current(ticket)) error = reason instanceof Error ? reason.message : String(reason);
		} finally {
			if (current(ticket)) setPhase('ready');
		}
	}
	async function generate(): Promise<void> {
		if (!ready || !engine || !record) return;
		const ticket = lifecycle;
		error = '';
		setPhase('generating');
		stopRequested = false;
		status = 'Sampling characters from the current model…';
		try {
			const result = await engine.generate(samplePrompt, {
				maxTokens: sampleLength,
				temperature,
				seed: samplingSeed,
				topK: 20
			});
			if (!current(ticket) || !record) return;
			sample = result;
			record = { ...record, samples: [...(record.samples ?? []), result] };
			observe(
				'Text sampled',
				`Step ${result.step}; sampling seed ${result.samplingSeed}; temperature ${result.temperature}; top-k ${result.topK}; ${result.tokenIds.length} generated characters${result.cancelled ? '; cancelled' : ''}.`
			);
			await persist();
			if (current(ticket)) status = `Sample recorded at step ${result.step}`;
		} catch (reason) {
			if (current(ticket)) error = reason instanceof Error ? reason.message : String(reason);
		} finally {
			if (current(ticket)) setPhase('ready');
		}
	}
	async function openRecord(next: StoryRunRecord): Promise<void> {
		const ticket = ++lifecycle;
		setPhase('loading');
		error = '';
		status = 'Opening recorded measurements; model remains unloaded…';
		try {
			await disposeRuntime();
			if (!current(ticket)) return;
			record = next;
			preset = next.presetId;
			seed = next.seed;
			snapshotIndex = null;
			selected = null;
			layerFilter = null;
			nearest = [];
			probe = null;
			lesioned = null;
			sample = next.samples?.at(-1) ?? null;
			savedCheckpointStep = next.checkpoint?.step ?? null;
			geometryEngine?.destroy();
			geometryEngine = null;
			if (next.snapshots.length) await getGeometry().setAtlas(next.snapshots.at(-1)!.atlas);
			if (!current(ticket)) return;
			status = 'Saved measurements open · resume explicitly to train or probe';
			setPhase('archived');
		} catch (reason) {
			if (current(ticket)) {
				error = String(reason);
				setPhase('error');
			}
		}
	}
	async function openSaved(id: string): Promise<void> {
		if (blocked) return;
		setPhase('loading');
		try {
			const next = await loadStoryRun(id);
			if (!mounted) return;
			if (!next) throw new Error('This saved story run is unavailable.');
			await openRecord(next);
		} catch (reason) {
			if (mounted) {
				error = String(reason);
				setPhase('error');
			}
		}
	}
	async function openReference(reference: StoryReference): Promise<void> {
		if (blocked) return;
		setPhase('loading');
		status = 'Loading and verifying recorded specimen…';
		try {
			const original = await loadStoryReference(reference);
			if (!mounted) return;
			const now = new Date().toISOString();
			const next = {
				...original,
				id: crypto.randomUUID(),
				createdAt: now,
				updatedAt: now,
				provenance: {
					...original.provenance,
					source: 'reference' as const,
					parentRunId: original.id
				}
			};
			await saveStoryRun(next);
			if (!mounted) return;
			await refreshRuns();
			await openRecord(next);
		} catch (reason) {
			if (mounted) {
				error = String(reason);
				setPhase('error');
			}
		}
	}
	async function importFile(event: Event): Promise<void> {
		const input = event.currentTarget as HTMLInputElement;
		const file = input.files?.[0];
		input.value = '';
		if (!file || blocked) return;
		setPhase('loading');
		try {
			const original = await importStoryRun(file);
			if (!mounted) return;
			const now = new Date().toISOString();
			const next = {
				...original,
				id: crypto.randomUUID(),
				createdAt: now,
				updatedAt: now,
				provenance: {
					...original.provenance,
					source: original.provenance?.source ?? ('browser' as const),
					parentRunId: original.id
				}
			};
			await saveStoryRun(next);
			if (!mounted) return;
			await refreshRuns();
			await openRecord(next);
		} catch (reason) {
			if (mounted) {
				error = String(reason);
				setPhase('error');
			}
		}
	}
	async function resume(): Promise<void> {
		if (blocked || !record?.checkpoint) return;
		const ticket = ++lifecycle;
		setPhase('initializing');
		error = '';
		status = 'Restoring checkpoint, optimizer and training random state…';
		try {
			await disposeRuntime();
			if (!current(ticket) || !record?.checkpoint) return;
			const source = record;
			const checkpoint = source.checkpoint!;
			if ((source.metrics.at(-1)?.step ?? 0) > checkpoint.step) {
				const now = new Date().toISOString();
				record = {
					...source,
					id: crypto.randomUUID(),
					title: `${source.title} · resumed ${checkpoint.step}`,
					createdAt: now,
					updatedAt: now,
					metrics: source.metrics.filter((metric) => metric.step <= checkpoint.step),
					snapshots: source.snapshots.filter((entry) => entry.atlas.step <= checkpoint.step),
					samples: source.samples?.filter((entry) => entry.step <= checkpoint.step),
					interventions: source.interventions?.filter((entry) => entry.step <= checkpoint.step),
					provenance: { ...source.provenance, source: 'browser', parentRunId: source.id },
					observations: [
						{
							time: now,
							step: checkpoint.step,
							title: 'Resumed durable checkpoint in a new run',
							detail: `Source ${source.id} retains later observations through step ${source.metrics.at(-1)?.step}. This continuation starts from durable weights at ${checkpoint.step}.`
						}
					]
				};
			}
			const owner = new StoryEngine((event) => handleEvent(event, ticket));
			engine = owner;
			const restored = await owner.loadCheckpoint(checkpoint, backend);
			if (!current(ticket)) return;
			const metric = restored.metrics;
			appendMetric(metric);
			residentModelId = restored.modelId;
			runtimeReady = true;
			snapshotIndex = null;
			sample = record?.samples?.at(-1) ?? null;
			if (snapshot?.atlas.step === metric.step) {
				await getGeometry().setAtlas(snapshot.atlas);
				if (!current(ticket)) return;
				await probeCurrent(ticket);
				await updateNeighbors();
				await persist();
			} else {
				await capture(ticket, true);
			}
			if (!current(ticket)) return;
			status = `Restored step ${metric.step} · ${metric.backend.toUpperCase()}`;
			setPhase('ready');
		} catch (reason) {
			if (current(ticket)) await failTraining(reason);
		}
	}
	async function selectSnapshot(index: number | null): Promise<void> {
		if (blocked || !record) return;
		neighborRevision++;
		snapshotIndex = index;
		nearest = [];
		const entry = record.snapshots[index ?? record.snapshots.length - 1];
		if (!entry) return;
		setPhase('measuring');
		status = `Opening measured map at step ${entry.atlas.step}…`;
		try {
			await getGeometry().setAtlas(entry.atlas);
			if (!mounted) return;
			await updateNeighbors();
			status = historical
				? `Historical map at step ${entry.atlas.step}; historical weights are not stored`
				: 'Latest measured map';
		} catch (reason) {
			if (mounted) error = String(reason);
		} finally {
			if (mounted) setPhase(runtimeReady ? 'ready' : 'archived');
		}
	}
	async function saveNow(): Promise<void> {
		if (blocked || !record) return;
		setPhase('saving');
		try {
			if (runtimeReady && engine) {
				const checkpoint = await engine.exportCheckpoint();
				if (!mounted || !record) return;
				record = { ...record, checkpoint };
			}
			const saved = await persist();
			if (mounted)
				status = saved
					? 'Evidence and latest available checkpoint saved locally'
					: 'Local save failed; evidence remains available for export';
		} catch (reason) {
			if (mounted) error = String(reason);
		} finally {
			if (mounted) setPhase(runtimeReady ? 'ready' : 'archived');
		}
	}
	function exportNow(): void {
		if (!record) return;
		try {
			exportStoryRun(record);
		} catch (reason) {
			error = `Export failed: ${reason instanceof Error ? reason.message : String(reason)}`;
		}
	}
	function number(value: number | undefined, digits = 1): string {
		return value === undefined
			? '—'
			: value !== 0 && Math.abs(value) < 0.001
				? value.toExponential(2)
				: value.toFixed(digits);
	}
	function time(value: string): string {
		return new Date(value).toLocaleString(undefined, {
			month: 'short',
			day: 'numeric',
			hour: '2-digit',
			minute: '2-digit'
		});
	}
	function archivedEffects(entry: NonNullable<StoryRunRecord['interventions']>[number]) {
		return Array.from(entry.probabilities, (probability, id) => ({
			id,
			probability,
			lesioned: entry.lesionedProbabilities[id],
			delta: entry.lesionedProbabilities[id] - probability
		}))
			.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))
			.slice(0, 8);
	}
	onMount(() => {
		mounted = true;
		void refreshRuns().catch((reason) => {
			if (mounted) storageWarning = String(reason);
		});
		void fetch(publicAsset('/experiments/stories-index.json'))
			.then(async (response) => {
				if (!response.ok) return;
				const data = (await response.json()) as { version?: number; references?: StoryReference[] };
				if (mounted && data.version === 1 && Array.isArray(data.references))
					references = data.references;
			})
			.catch(() => {});
		return () => {
			mounted = false;
			lifecycle++;
			neighborRevision++;
			onbusy?.(false);
			void engine?.dispose();
			geometryEngine?.destroy();
		};
	});
</script>

<input
	type="file"
	accept=".tissue,application/octet-stream"
	{@attach (input) => {
		importInput = input;
		return () => {
			importInput = undefined;
		};
	}}
	onchange={importFile}
	class="file-input"
	aria-label="Import TinyStories specimen"
/>
<div class="story-lab" data-active={active}>
	<header class="story-toolbar">
		<div class="lab-title">
			<Icon name="book" size={21} />
			<div>
				<h1>TinyStories</h1>
				<span
					>{record
						? `${(storyParameterCount(config) / 1e6).toFixed(2)}M parameters · ${config.layers} layers · measured step ${snapshot?.atlas.step ?? '—'}`
						: 'Character transformer · measured MLP activations'}</span
				>
			</div>
			<span class="study-label">Study 003</span>
		</div>
		{#if phase === 'training'}<button
				class="secondary training-progress"
				onclick={pause}
				disabled={stopRequested}
				><Icon name="pause" size={14} />Pause training · step {latestMetric?.step ?? 0}</button
			>{/if}
		{@render representation?.()}
		<div class="file-controls">
			<button
				class="icon-button"
				onclick={saveNow}
				disabled={!record || blocked}
				aria-label="Save story checkpoint"
				title="Save latest evidence and checkpoint"><Icon name="check" size={15} /></button
			><button
				class="icon-button"
				onclick={exportNow}
				disabled={!record || busy}
				aria-label="Export story run"
				title="Export binary .tissue specimen"><Icon name="download" size={15} /></button
			><button
				class="icon-button"
				onclick={() => importInput?.click()}
				disabled={blocked}
				aria-label="Import story run"
				title="Import binary .tissue specimen"><Icon name="upload" size={15} /></button
			>
		</div>
	</header>
	{#if error || storageWarning}<div class="notice" role="alert">
			<Icon name="info" size={14} /><span
				>{error}{error && storageWarning ? ' · ' : ''}{storageWarning}</span
			><button
				class="icon-button"
				aria-label="Dismiss story message"
				onclick={() => {
					error = '';
					storageWarning = '';
				}}><Icon name="close" size={13} /></button
			>
		</div>{/if}
	{#if disabled}<div class="external-busy">
			Another lab worker is active. Model operations here are paused until it finishes.
		</div>{/if}
	<ResearchWorkspace name="story" bind:panel={toolPanel} bind:collapsed={toolsCollapsed}>
		{#snippet model()}
			<aside class="setup-panel">
				<div class="resume-row">
					{#if record?.checkpoint && !runtimeReady}<button
							class="primary initialize"
							onclick={resume}
							disabled={blocked}
							><Icon name="play" size={13} />Resume step {record.checkpoint.step}</button
						>{/if}
				</div>
				<details class="lab-disclosure" bind:open={setupOpen}>
					<summary>New model <small>Architecture & initialization</small></summary>
					<div class="section-label">
						<Icon name="layers" size={13} />
						<h2>Model scale</h2>
						<span>Next initialization</span>
					</div>
					<div class="preset-list">
						{#each presetIds as id (id)}{@const option = STORY_PRESETS[id]}<button
								class:chosen={preset === id}
								onclick={() => {
									preset = id;
									if (id !== 'small' && backend === 'wasm') backend = 'auto';
								}}
								disabled={blocked}
								aria-pressed={preset === id}
								><span
									><strong>{(storyParameterCount(option) / 1e6).toFixed(3)}M</strong><small
										>{id === 'small' ? 'Compact' : id === 'medium' ? 'Expanded' : 'Large'}</small
									></span
								><span>{option.layers} layers · {option.width} width · {option.heads} heads</span
								><span
									>{storyUnitCount(option).toLocaleString()} MLP channels · {option.context} context</span
								></button
							>{/each}
					</div>
					<div class="initialize-options">
						<label for="story-seed">Initialization seed</label><input
							id="story-seed"
							type="number"
							min="0"
							max="4294967295"
							bind:value={seed}
							disabled={blocked}
						/><span class="control-caption">Backend</span>
						<div class="backend-options">
							{#each ['auto', 'webgpu', 'wasm'] as device (device)}<button
									class:chosen={backend === device}
									onclick={() => (backend = device as StoryBackend | 'auto')}
									disabled={blocked || (device === 'wasm' && preset !== 'small')}
									>{device === 'auto' ? 'Auto' : device === 'webgpu' ? 'GPU' : 'WASM'}</button
								>{/each}
						</div>
						<button
							class="secondary initialize"
							onclick={initialize}
							disabled={blocked || !Number.isInteger(seed) || seed < 0 || seed > 0xffffffff}
							><Icon name="network" size={13} />{record
								? 'Initialize new run'
								: 'Initialize model'}</button
						>
						<p>
							Models start only when initialized or resumed. WASM is available for the compact
							preset; larger models need a supported GPU and sufficient memory.
						</p>
						{#if record && !record.checkpoint}<p>
								This specimen contains measurements only. Initialize a new run to train; no
								resumable weights are included.
							</p>{/if}
					</div>
				</details>
				<details class="lab-disclosure" open={runtimeReady}>
					<summary>Training <small>Updates & checkpoints</small></summary>
					<div class="transport">
						<span>Updates</span>
						<div class="segmented">
							{#each [25, 100, 500] as steps (steps)}<button
									class:chosen={budget === steps}
									onclick={() => (budget = steps)}
									disabled={blocked}>{steps}</button
								>{/each}
						</div>
						{#if phase === 'training'}<button
								class="primary"
								onclick={pause}
								disabled={stopRequested}
								><Icon name="pause" size={13} />{stopRequested ? 'Pausing…' : 'Pause'}</button
							>{:else if phase === 'generating'}<button
								class="secondary"
								onclick={pause}
								disabled={stopRequested}
								><Icon name="pause" size={13} />{stopRequested
									? 'Stopping…'
									: 'Stop sampling'}</button
							>{:else}<button class="primary" onclick={train} disabled={!ready}
								><Icon name="play" size={13} />Train</button
							>{/if}
					</div>
				</details>
				<div class="archive-panel">
					<div class="section-label">
						<Icon name="flask" size={13} />
						<h2>Recorded specimens</h2>
					</div>
					{#each references as reference (reference.id)}<button
							class="archive-item reference"
							onclick={() => openReference(reference)}
							disabled={blocked}
							><span>{reference.title}</span><small
								>Reference · step {reference.step} · {(reference.bytes / 1e6).toFixed(1)} MB</small
							><small
								>Held-out {reference.validationLoss.toFixed(3)} / unigram {reference.unigramLoss.toFixed(
									3
								)} nats</small
							></button
						>{/each}{#each runs as saved (saved.id)}<button
							class="archive-item"
							class:current={record?.id === saved.id}
							onclick={() => openSaved(saved.id)}
							disabled={blocked}
							><span>{saved.title}</span><small
								>Weights {saved.checkpointStep ?? '—'} · {saved.snapshotCount} measured maps</small
							><small>{time(saved.updatedAt)}</small></button
						>{:else}<p class="archive-empty">
							Completed measurements and checkpoints will be saved here.
						</p>{/each}
				</div>
				<details class="lab-disclosure">
					<summary>Dataset & provenance</summary>
					<div class="corpus-card">
						<div class="section-label">
							<Icon name="book" size={13} />
							<h2>Corpus</h2>
						</div>
						<strong>TinyStories · character level</strong>
						<p>
							{corpus
								? `${corpus.trainStories.toLocaleString()} training stories; ${corpus.calibrationStories} calibration and ${corpus.evaluationStories} evaluation stories.`
								: '2,097 training stories and 184 held-out stories. Calibration and evaluation are separated by story.'}
						</p>
						<span>96-character vocabulary · ASCII + newline</span><a
							href="https://huggingface.co/datasets/roneneldan/TinyStories"
							target="_blank"
							rel="noreferrer">Dataset & provenance <Icon name="right" size={10} /></a
						>
					</div>
				</details>
			</aside>
		{/snippet}
		<main class="story-main">
			<div class="view-toolbar">
				<div class="segmented">
					<button class:chosen={viewMode === 'functional'} onclick={() => selectView('functional')}
						><Icon name="cube" size={12} />Functional</button
					><button
						class:chosen={viewMode === 'architecture'}
						onclick={() => selectView('architecture')}
						><Icon name="network" size={12} />Model layout</button
					>
				</div>
				<div class="layer-options">
					<span>Layers</span><button
						class:chosen={layerFilter === null}
						onclick={() => selectLayer(null)}>All</button
					>{#each Array.from({ length: config.layers }, (_, i) => i) as layer (layer)}<button
							class:chosen={layerFilter === layer}
							onclick={() => selectLayer(layer)}>L{layer + 1}</button
						>{/each}
				</div>
				<span class="architecture-label"
					>{config.layers} × {config.width} · {unitCount.toLocaleString()} units</span
				>
			</div>
			<div class="story-field">
				{#if snapshot}<NeuralField
						{points}
						{edges}
						{selected}
						onselect={(id) => {
							selectUnit(id);
							toolPanel = 'inspector';
							toolsCollapsed = false;
						}}
						{theme}
						{layerFilter}
						mode="activation"
						geometryLabel={viewMode === 'functional'
							? `Calibration activation geometry · step ${snapshot.atlas.step}`
							: 'Layer / channel coordinates'}
						edgeLabel={viewMode === 'functional'
							? 'Selected neighbors · full fingerprint space'
							: 'Architectural coordinates · no graph edges'}
					/>{:else}<div class="field-empty">
						<Icon name="cube" size={38} />
						<h2>
							{busy ? 'Preparing the first measurement' : 'Explore a character model'}
						</h2>
						<p>
							{busy
								? status
								: 'Initialize a model or open a recorded specimen. Coordinates will come from real activation measurements across fixed calibration text.'}
						</p>
						{#if !busy}<div class="welcome-actions">
								<button
									class="primary"
									disabled={!references.length || blocked}
									onclick={() => {
										const reference = references[0];
										if (reference) void openReference(reference);
									}}>Open recorded model</button
								>
								<button
									class="secondary"
									onclick={() => {
										toolPanel = 'model';
										toolsCollapsed = false;
										setupOpen = true;
									}}>Set up a model</button
								>
							</div>{/if}
						<div>
							{unitCount.toLocaleString()} MLP channels <span>×</span> 128 calibration coordinates
						</div>
					</div>{/if}
			</div>
			{#if snapshot}<div class="map-context">
					<span
						>{viewMode === 'functional'
							? `3D PCA · ${(snapshot.geometry.explainedVariance * 100).toFixed(1)}% variance retained`
							: 'Layer / channel coordinates'}</span
					><span
						>{viewMode === 'functional'
							? 'Similarity is not connectivity'
							: 'Spacing chosen for display'}</span
					>
				</div>{/if}

			<details class="lab-disclosure">
				<summary
					>Measurement details <small>Evaluation, projection quality & map history</small></summary
				>
				<StoryMetricsPanel metrics={record?.metrics ?? []} compact />
				<div class="geometry-footer">
					<span
						>{snapshot
							? `${snapshot.atlas.unitCount.toLocaleString()} / ${snapshot.atlas.unitCount.toLocaleString()} channels captured`
							: 'All channels will be captured'}</span
					><span
						>{snapshot
							? `${snapshot.atlas.dimensions} fixed calibration coordinates`
							: '8 windows × 16 positions'}</span
					>{#if snapshot && viewMode === 'functional'}<span
							>{snapshot.atlas.unitCount - snapshot.geometry.audit.validPopulation} unresolved directions
							omitted</span
						><span
							>Variance <strong>{(snapshot.geometry.explainedVariance * 100).toFixed(1)}%</strong
							></span
						><span
							>Audited retention <strong
								>{snapshot.geometry.audit.neighborRetention === null
									? '—'
									: `${(snapshot.geometry.audit.neighborRetention * 100).toFixed(1)}%`}</strong
							></span
						>{/if}
				</div>
				<div class="view-note">
					{#if snapshot}{#if viewMode === 'architecture'}Positions label the true layer and channel;
							spacing is chosen for display.{:else}All resolved unit fingerprints enter PCA.
							Neighbor retention is scored on {snapshot.geometry.audit.scoredFocals} / {snapshot
								.geometry.audit.planned} preselected focal units ({snapshot.geometry.audit
								.resolvedFocals} resolved; {snapshot.geometry.audit.requested} requested). This sample
							does not establish all-unit retention. Links are selected exact nearest neighbors, not model
							connections.{/if}<span
							>{matchingProbe
								? `Activity: prompt character ${token + 1}, measured step ${matchingProbe.step}.`
								: `Activity: calibration window 1, position ${(snapshot.atlas.positions[0] ?? 0) + 1}.`}</span
						>{:else}Coordinates will be fitted from every channel’s fixed calibration responses.
						Measured captures form the timeline.{/if}
				</div>
				{#if snapshot && viewMode === 'functional' && (!snapshot.geometry.pca.converged || snapshot.geometry.pca.boundaryUncertain)}<div
						class="geometry-warning"
					>
						{#if !snapshot.geometry.pca.converged}Approximate PCA did not reach its requested
							tolerance after {snapshot.geometry.pca.iterations} iterations (relative residual {snapshot.geometry.pca.relativeResidual.toExponential(
								2
							)}).
						{/if}{#if snapshot.geometry.pca.boundaryUncertain}The third/fourth component boundary is
							numerically uncertain; treat the selected 3D subspace cautiously.{/if}
					</div>{/if}
				<div class="timeline">
					<span><Icon name="activity" size={11} />Measured maps</span
					>{#each record?.snapshots ?? [] as frame, i (frame.atlas.step)}<button
							class:chosen={(snapshotIndex ?? (record?.snapshots.length ?? 1) - 1) === i}
							onclick={() => selectSnapshot(i)}
							disabled={blocked}>{frame.atlas.step}</button
						>{/each}<button
						class="latest"
						onclick={() => selectSnapshot(null)}
						disabled={blocked || !record?.snapshots.length}>Latest</button
					><span class="checkpoint-label">Durable weights {savedCheckpointStep ?? '—'}</span>
				</div>
				{#if historical}<div class="historical-note">
						Viewing step {snapshot?.atlas.step}. Historical maps retain activations and coordinates;
						only the latest saved checkpoint retains weights. Return to Latest to use the resident
						model.
					</div>{/if}
				{#if selected !== null && nearest.length && viewMode === 'functional'}<div
						class="neighbor-row"
					>
						<span>Nearest units in fingerprint space</span
						>{#each nearest as neighbor (neighbor.index)}<button
								onclick={() => selectUnit(neighbor.index)}
								title={`Cosine ${neighbor.similarity.toFixed(4)}; distance ${neighbor.distance.toFixed(4)}`}
								>L{Math.floor(neighbor.index / config.hidden) + 1}/{neighbor.index %
									config.hidden}</button
							>{/each}
					</div>{/if}
			</details>
		</main>
		{#snippet inspector()}
			<details class="lab-disclosure prompt-probe">
				<summary>Prompt probe <small>Inspect every input position</small></summary>
				<div class="probe-input">
					<label for="story-probe-text">Probe context</label><textarea
						id="story-probe-text"
						bind:value={probePrompt}
						rows="2"
						spellcheck="false"
						disabled={blocked}
						placeholder="Enter a prompt using the 96-character vocabulary"></textarea><button
						class="secondary"
						onclick={inspectPrompt}
						disabled={!ready || !probePrompt.length}
						><Icon name="activity" size={13} />Run prompt</button
					><span
						>{probe
							? `Measured ${probe.prompt.tokenIds.length} characters at step ${probe.step}`
							: 'Activations are measured after a real forward pass'}</span
					>
				</div>
			</details>
			<div class="story-inspector">
				<StoryProbePanel
					probe={liveProbe}
					{lesioned}
					{selected}
					{token}
					{config}
					busy={blocked || !runtimeReady || historical}
					onselect={selectUnit}
					ontoken={(position) => (token = position)}
					onlesion={silenceUnit}
				/>
			</div>
		{/snippet}
	</ResearchWorkspace>
	<details class="lab-disclosure evidence-drawer">
		<summary
			>Samples & evidence <small>Batch generation, learning curves and experiment history</small
			></summary
		>
		<div class="story-lower">
			<section class="sample-panel">
				<div class="lower-tabs">
					<button class:chosen={footerTab === 'samples'} onclick={() => (footerTab = 'samples')}
						><Icon name="book" size={13} />Generate & compare</button
					><button class:chosen={footerTab === 'history'} onclick={() => (footerTab = 'history')}
						><Icon name="activity" size={13} />Experiment history
						<span>{record?.observations.length ?? 0}</span></button
					>
				</div>
				{#if footerTab === 'samples'}<div class="sample-controls">
						<label for="story-sample-prompt">Sampling prompt</label><input
							id="story-sample-prompt"
							bind:value={samplePrompt}
							disabled={blocked}
						/><label for="story-sampling-seed">Seed</label><input
							id="story-sampling-seed"
							type="number"
							min="0"
							max="4294967295"
							bind:value={samplingSeed}
							disabled={blocked}
						/><label for="story-temperature">Temperature</label><input
							id="story-temperature"
							type="number"
							min="0.1"
							max="2"
							step="0.1"
							bind:value={temperature}
							disabled={blocked}
						/>
						<div class="segmented">
							{#each [32, 64, 128] as length (length)}<button
									class:chosen={sampleLength === length}
									onclick={() => (sampleLength = length)}
									disabled={blocked}>{length}</button
								>{/each}
						</div>
						<button
							class="secondary"
							onclick={generate}
							disabled={!ready ||
								!samplePrompt.length ||
								!Number.isInteger(samplingSeed) ||
								samplingSeed < 0 ||
								samplingSeed > 0xffffffff ||
								!Number.isFinite(temperature) ||
								temperature < 0.1 ||
								temperature > 2}><Icon name="play" size={12} />Sample</button
						>
					</div>
					<div class="sample-output">
						{#if sample}<div class="sample-meta">
								<span
									>Step {sample.step} · sampling seed {sample.samplingSeed} · temperature {sample.temperature}
									· top-k {sample.topK}</span
								><span
									>{sample.tokenIds.length} / {sample.requestedTokens} characters{sample.cancelled
										? ' · cancelled'
										: ''}</span
								>
							</div>
							<p>
								<span class="sample-prefix">{sample.prompt.text}</span>{sample.completion}
							</p>{:else}<p class="empty-sample">
								Sampled text will appear here. Samples are observations of the current checkpoint,
								not evidence of story understanding.
							</p>{/if}
					</div>
					{#if (record?.samples?.length ?? 0) > 1}<div class="sample-history">
							<span>Recorded samples</span
							>{#each record?.samples ?? [] as past, i (`${past.step}-${i}`)}<button
									onclick={() => (sample = past)}
									class:chosen={sample === past}>Step {past.step} · #{i + 1}</button
								>{/each}
						</div>{/if}{:else}<div class="observation-list">
						{#each record?.interventions ?? [] as intervention, i (`${intervention.capturedAt}-${i}`)}<details
								class="archived-intervention"
							>
								<summary
									>Recorded ablation · unit {intervention.neuron} · step {intervention.step}</summary
								>
								<p>
									All prompt positions silenced. The table shows the eight largest absolute
									probability changes; the archive retains all 96 values.
								</p>
								<code class="intervention-prompt">{intervention.prompt}</code>
								<table>
									<thead
										><tr
											><th>Character</th><th>Intact</th><th>Lesioned</th><th>Δ percentage points</th
											></tr
										></thead
									><tbody
										>{#each archivedEffects(intervention) as effect (effect.id)}<tr
												><td
													>{effect.id === 0
														? '↵'
														: effect.id === 1
															? '␣'
															: STORY_CHARACTERS[effect.id]}</td
												><td>{(effect.probability * 100).toFixed(3)}%</td><td
													>{(effect.lesioned * 100).toFixed(3)}%</td
												><td>{effect.delta > 0 ? '+' : ''}{number(effect.delta * 100, 3)}</td></tr
											>{/each}</tbody
									>
								</table>
							</details>{/each}
						{#each [...(record?.observations ?? [])].reverse() as observation, i (`${observation.time}-${i}`)}<article
							>
								<time datetime={observation.time}
									>{time(observation.time)} · step {observation.step}</time
								>
								<h3>{observation.title}</h3>
								<p>{observation.detail}</p>
							</article>{:else}<p class="empty-sample">
								Initialization, training, samples and interventions will be recorded here.
							</p>{/each}
					</div>{/if}
			</section>
			<section class="learning-panel">
				<StoryMetricsPanel metrics={record?.metrics ?? []} />
			</section>
		</div>
	</details>
	<footer class="story-status" role="status">
		<span class:working={busy}></span>
		<p>{status}</p>
		{#if phase === 'training' && trainingTarget !== null}<span
				>{latestMetric?.step ?? 0} / {trainingTarget} updates</span
			>{/if}{#if measurementProgress.total > 0 && busy && phase !== 'generating'}<span
				>Calibration {measurementProgress.completed} / {measurementProgress.total}</span
			>{/if}<span>{latestMetric?.backend.toUpperCase() ?? 'No backend allocated'}</span><span
			>Resident step {residentStep ?? '—'}</span
		>{#if latestMetric?.stepMs}<span>{number(latestMetric.stepMs)} ms / update</span>{/if}
	</footer>
</div>

<style>
	.story-lab {
		color: var(--ink);
	}
	.file-input {
		display: none;
	}
	.story-toolbar {
		display: flex;
		align-items: center;
		gap: 20px;
		padding: 12px 17px;
		background: var(--surface);
		border-bottom: 1px solid var(--line);
	}
	.lab-title {
		display: flex;
		align-items: center;
		gap: 10px;
		margin-right: auto;
	}
	h1,
	h2,
	h3,
	p {
		margin: 0;
	}
	h1 {
		font-size: 15px;
		font-weight: 600;
		letter-spacing: -0.3px;
	}
	.lab-title > div > span {
		display: block;
		margin-top: 5px;
		color: var(--muted);
		font: 12px var(--mono);
	}
	.study-label {
		border: 1px solid var(--line);
		border-radius: 4px;
		color: var(--muted);
		padding: 4px 6px;
		font: 12px var(--mono);
	}
	.transport,
	.file-controls {
		display: flex;
		align-items: center;
		gap: 8px;
	}
	.transport > span {
		color: var(--muted);
		font: 12px var(--mono);
	}
	.transport > button {
		min-width: 83px;
	}
	.file-controls {
		gap: 3px;
		padding-left: 12px;
		border-left: 1px solid var(--line);
	}
	.segmented {
		display: flex;
		gap: 2px;
		border: 1px solid var(--line);
		border-radius: 5px;
		padding: 2px;
	}
	.segmented button {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 5px;
		border: 0;
		border-radius: 3px;
		background: transparent;
		color: var(--muted);
		padding: 5px 9px;
		font-size: 12px;
	}
	.segmented button.chosen {
		background: var(--surface-hover);
		color: var(--ink);
	}
	.notice {
		display: flex;
		align-items: center;
		gap: 10px;
		margin: 10px 14px;
		padding: 9px 12px;
		border: 1px solid color-mix(in srgb, var(--warning) 45%, var(--line));
		border-radius: 5px;
		color: var(--warning);
		font-size: 13px;
		line-height: 1.5;
	}
	.notice > span {
		flex: 1;
	}
	.external-busy {
		padding: 9px 17px;
		color: var(--muted);
		font-size: 12px;
		background: var(--surface-raised);
		border-bottom: 1px solid var(--line);
	}
	.setup-panel {
		background: var(--surface);
		border-right: 1px solid var(--line);
		min-width: 0;
	}
	.section-label {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 13px 12px;
	}
	.section-label h2 {
		font-size: 12px;
		font-weight: 550;
	}
	.section-label > span {
		color: var(--faint);
		font: 12px var(--mono);
		margin-left: auto;
	}
	.preset-list {
		padding: 0 10px;
		display: flex;
		flex-direction: column;
		gap: 7px;
	}
	.preset-list button {
		display: flex;
		flex-direction: column;
		gap: 7px;
		text-align: left;
		padding: 12px 10px;
		border: 1px solid var(--line);
		border-radius: 6px;
		background: var(--bg);
	}
	.preset-list button.chosen {
		border-color: color-mix(in srgb, var(--accent) 65%, var(--line));
		background: color-mix(in srgb, var(--accent) 4%, var(--surface));
	}
	.preset-list button > span:first-child {
		display: flex;
		align-items: center;
		justify-content: space-between;
	}
	.preset-list strong {
		font: 20px var(--mono);
		letter-spacing: -0.6px;
	}
	.preset-list small {
		color: var(--muted);
		font-size: 12px;
	}
	.preset-list button > span:not(:first-child) {
		font: 12px/1.5 var(--mono);
		color: var(--muted);
	}
	.initialize-options {
		display: grid;
		grid-template-columns: 1fr 65px;
		gap: 9px;
		padding: 14px 12px;
		border-bottom: 1px solid var(--line);
		align-items: center;
	}
	.initialize-options > label,
	.control-caption {
		font: 12px var(--mono);
		color: var(--muted);
	}
	.initialize-options input {
		width: 65px;
		padding: 5px 7px;
		font: 12px var(--mono);
	}
	.backend-options {
		display: flex;
		grid-column: 1/-1;
		gap: 4px;
	}
	.backend-options button {
		flex: 1;
		padding: 6px;
		font-size: 12px;
		color: var(--muted);
		border: 1px solid var(--line);
		border-radius: 4px;
		background: transparent;
	}
	.backend-options button.chosen {
		border-color: var(--accent);
		color: var(--ink);
	}
	.initialize-options .initialize {
		grid-column: 1/-1;
		width: 100%;
		font-size: 12px;
	}
	.initialize-options p {
		grid-column: 1/-1;
		color: var(--muted);
		font-size: 12px;
		line-height: 1.6;
	}
	.corpus-card {
		padding: 0 12px 15px;
		border-bottom: 1px solid var(--line);
	}
	.corpus-card .section-label {
		padding: 13px 0 10px;
	}
	.corpus-card > strong {
		font-size: 12px;
		font-weight: 500;
	}
	.corpus-card p {
		margin: 7px 0;
		font-size: 12px;
		line-height: 1.65;
		color: var(--muted);
	}
	.corpus-card > span {
		font: 12px/1.6 var(--mono);
		color: var(--muted);
	}
	.corpus-card a {
		display: flex;
		align-items: center;
		gap: 4px;
		margin-top: 10px;
		color: var(--muted);
		font-size: 12px;
		text-decoration: none;
	}
	.archive-panel {
		max-height: 300px;
		overflow: auto;
		padding-bottom: 12px;
	}
	.archive-item {
		display: flex;
		flex-direction: column;
		gap: 5px;
		text-align: left;
		width: calc(100% - 20px);
		margin: 0 10px 7px;
		padding: 10px;
		border: 1px solid var(--line);
		border-radius: 5px;
		background: transparent;
	}
	.archive-item.current,
	.archive-item.reference {
		border-color: color-mix(in srgb, var(--accent) 38%, var(--line));
	}
	.archive-item > span {
		font-size: 12px;
	}
	.archive-item small {
		color: var(--muted);
		font: 12px/1.5 var(--mono);
	}
	.archive-empty {
		font-size: 12px;
		line-height: 1.6;
		color: var(--muted);
		padding: 0 13px;
	}
	.story-main {
		min-width: 0;
	}
	.view-toolbar {
		display: flex;
		align-items: center;
		gap: 12px;
		padding: 9px 12px;
		border-bottom: 1px solid var(--line);
		background: var(--surface);
	}
	.layer-options {
		display: flex;
		gap: 3px;
		align-items: center;
	}
	.layer-options > span {
		font: 12px var(--mono);
		color: var(--muted);
		margin-right: 3px;
	}
	.layer-options button {
		font: 12px var(--mono);
		padding: 5px;
		border: 1px solid transparent;
		border-radius: 3px;
		color: var(--muted);
		background: transparent;
	}
	.layer-options button.chosen {
		border-color: var(--line);
		background: var(--surface-hover);
		color: var(--ink);
	}
	.architecture-label {
		margin-left: auto;
		color: var(--faint);
		font: 12px var(--mono);
	}
	.story-field {
		height: 480px;
		min-width: 0;
	}
	.field-empty {
		display: flex;
		height: 100%;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		text-align: center;
		padding: 35px;
		gap: 18px;
		color: var(--accent);
		background: radial-gradient(
			ellipse at center,
			color-mix(in srgb, var(--accent) 3%, transparent),
			transparent 65%
		);
	}
	.field-empty h2 {
		font-size: 20px;
		font-weight: 450;
		letter-spacing: -0.4px;
		color: var(--ink);
	}
	.field-empty p {
		color: var(--muted);
		font-size: 13px;
		line-height: 1.75;
		max-width: 440px;
	}
	.field-empty > div {
		font: 12px var(--mono);
		color: var(--faint);
	}
	.field-empty > div span {
		padding: 0 10px;
	}
	.geometry-footer {
		display: flex;
		align-items: center;
		flex-wrap: wrap;
		gap: 12px;
		padding: 10px 12px;
		border-top: 1px solid var(--line);
		font: 12px var(--mono);
		color: var(--muted);
	}
	.geometry-footer span:nth-child(3) {
		margin-left: auto;
	}
	.geometry-footer strong {
		font-weight: 400;
		color: var(--ink);
	}
	.view-note {
		padding: 0 12px 10px;
		color: var(--muted);
		font-size: 12px;
		line-height: 1.65;
	}
	.view-note > span {
		display: block;
		color: var(--faint);
		margin-top: 4px;
		font: 12px/1.5 var(--mono);
	}
	.geometry-warning {
		margin: 0 12px 11px;
		padding: 8px 10px;
		border: 1px solid color-mix(in srgb, var(--warning) 35%, var(--line));
		border-radius: 5px;
		color: var(--warning);
		font-size: 12px;
		line-height: 1.6;
	}
	.timeline {
		display: flex;
		align-items: center;
		gap: 5px;
		padding: 9px 12px;
		border-top: 1px solid var(--line);
		border-bottom: 1px solid var(--line);
		flex-wrap: wrap;
	}
	.timeline > span:first-child {
		display: flex;
		align-items: center;
		gap: 5px;
		color: var(--muted);
		font: 12px var(--mono);
		margin-right: 4px;
	}
	.timeline button {
		border: 1px solid transparent;
		border-radius: 3px;
		padding: 4px 6px;
		color: var(--muted);
		font: 12px var(--mono);
		background: transparent;
	}
	.timeline button.chosen {
		border-color: var(--line);
		background: var(--surface-hover);
		color: var(--ink);
	}
	.timeline .latest {
		color: var(--accent);
	}
	.checkpoint-label {
		margin-left: auto;
		font: 12px var(--mono);
		color: var(--faint);
	}
	.historical-note {
		padding: 10px 13px;
		border-bottom: 1px solid var(--line);
		color: var(--warning);
		font-size: 12px;
		line-height: 1.6;
	}
	.probe-input {
		display: grid;
		grid-template-columns: 1fr auto;
		gap: 8px;
		padding: 13px 12px;
		background: var(--surface);
	}
	.probe-input label {
		grid-column: 1/-1;
		color: var(--muted);
		font: 12px var(--mono);
	}
	.probe-input textarea {
		resize: vertical;
		min-height: 54px;
		font: 13px/1.6 var(--mono);
	}
	.probe-input button {
		align-self: start;
		margin-top: 2px;
		font-size: 12px;
	}
	.probe-input > span {
		grid-column: 1/-1;
		color: var(--faint);
		font: 12px var(--mono);
	}
	.neighbor-row {
		display: flex;
		gap: 5px;
		flex-wrap: wrap;
		align-items: center;
		padding: 7px 12px 13px;
		background: var(--surface);
	}
	.neighbor-row > span {
		font: 12px var(--mono);
		color: var(--muted);
		margin-right: 6px;
	}
	.neighbor-row > button {
		border: 1px solid var(--line);
		background: transparent;
		border-radius: 3px;
		padding: 4px 5px;
		font: 12px var(--mono);
	}
	.story-inspector {
		min-width: 0;
		border-left: 1px solid var(--line);
		background: var(--surface);
	}
	.story-lower {
		display: grid;
		grid-template-columns: minmax(0, 1.6fr) minmax(350px, 1fr);
		border-bottom: 1px solid var(--line);
	}
	.sample-panel {
		min-width: 0;
	}
	.lower-tabs {
		display: flex;
		gap: 16px;
		padding: 0 15px;
		border-bottom: 1px solid var(--line);
		background: var(--surface);
	}
	.lower-tabs button {
		display: flex;
		gap: 6px;
		align-items: center;
		border: 0;
		border-bottom: 1px solid transparent;
		padding: 12px 0;
		background: transparent;
		color: var(--muted);
		font-size: 12px;
	}
	.lower-tabs button.chosen {
		border-bottom-color: var(--accent);
		color: var(--ink);
	}
	.lower-tabs button > span {
		font: 12px var(--mono);
		color: var(--faint);
	}
	.sample-controls {
		display: flex;
		gap: 8px;
		align-items: center;
		flex-wrap: wrap;
		padding: 14px 15px;
	}
	.sample-controls label {
		color: var(--muted);
		font: 12px var(--mono);
	}
	.sample-controls input {
		padding: 6px 8px;
		font: 12px var(--mono);
	}
	.sample-controls input[type='number'] {
		width: 60px;
	}
	.sample-controls > input:first-of-type {
		flex: 1;
		min-width: 160px;
	}
	.sample-controls > button {
		font-size: 12px;
	}
	.sample-controls .segmented button {
		font: 12px var(--mono);
		padding: 5px 6px;
	}
	.sample-output {
		padding: 0 15px 15px;
	}
	.sample-meta {
		display: flex;
		justify-content: space-between;
		gap: 10px;
		flex-wrap: wrap;
		font: 12px var(--mono);
		color: var(--muted);
		padding-bottom: 11px;
	}
	.sample-output > p {
		min-height: 110px;
		border: 1px solid var(--line);
		border-radius: 6px;
		background: var(--surface);
		padding: 18px;
		font: 15px/1.9 var(--mono);
		white-space: pre-wrap;
		overflow-wrap: anywhere;
	}
	.sample-prefix {
		color: var(--muted);
	}
	.sample-output > p.empty-sample {
		display: flex;
		align-items: center;
		justify-content: center;
		text-align: center;
		font: 13px/1.7 var(--mono);
		color: var(--faint);
	}
	.sample-history {
		display: flex;
		align-items: center;
		gap: 6px;
		flex-wrap: wrap;
		padding: 0 15px 14px;
	}
	.sample-history > span {
		font: 12px var(--mono);
		color: var(--muted);
		margin-right: 5px;
	}
	.sample-history > button {
		border: 1px solid var(--line);
		border-radius: 4px;
		background: transparent;
		font: 12px var(--mono);
		padding: 5px 7px;
	}
	.sample-history > button.chosen {
		border-color: var(--accent);
	}
	.learning-panel {
		border-left: 1px solid var(--line);
		background: var(--surface);
	}
	.learning-panel :global(.metric-strip) {
		display: none;
	}
	.learning-panel :global(.learning-curve) {
		border-top: 0;
		padding-top: 17px;
	}
	.observation-list {
		max-height: 330px;
		overflow: auto;
		padding: 8px 15px 15px;
	}
	.observation-list article {
		padding: 13px 0;
		border-bottom: 1px solid var(--line);
	}
	.observation-list time {
		color: var(--muted);
		font: 12px var(--mono);
	}
	.observation-list h3 {
		font-size: 13px;
		font-weight: 500;
		margin: 7px 0;
	}
	.observation-list p {
		font-size: 12px;
		line-height: 1.7;
		color: var(--muted);
		overflow-wrap: anywhere;
	}
	.observation-list > p {
		padding: 30px;
		text-align: center;
	}
	.archived-intervention {
		margin-top: 10px;
		border: 1px solid color-mix(in srgb, var(--warning) 30%, var(--line));
		border-radius: 5px;
		padding: 10px;
	}
	.archived-intervention summary {
		font-size: 12px;
		cursor: pointer;
	}
	.archived-intervention p {
		padding: 10px 0;
	}
	.intervention-prompt {
		display: block;
		font: 12px/1.6 var(--mono);
		white-space: pre-wrap;
		overflow-wrap: anywhere;
	}
	.archived-intervention table {
		width: 100%;
		border-collapse: collapse;
		margin-top: 10px;
		font: 12px var(--mono);
	}
	.archived-intervention th,
	.archived-intervention td {
		text-align: right;
		padding: 5px;
		border-bottom: 1px solid var(--line);
	}
	.archived-intervention th {
		color: var(--muted);
		font-weight: 400;
	}
	.archived-intervention th:first-child,
	.archived-intervention td:first-child {
		text-align: left;
	}
	.story-status {
		display: flex;
		align-items: center;
		gap: 15px;
		padding: 9px 17px;
		font: 12px var(--mono);
		color: var(--muted);
		background: var(--surface);
		flex-wrap: wrap;
	}
	.story-status > span:first-child {
		width: 4px;
		height: 4px;
		border-radius: 50%;
		background: var(--faint);
	}
	.story-status > span.working {
		background: var(--warning);
	}
	.story-status p {
		flex: 1;
	}
	.story-toolbar {
		padding: 22px 24px;
		background: var(--bg);
		border-bottom: 0;
		flex-wrap: wrap;
		gap: 16px;
	}
	h1 {
		font-size: 22px;
		letter-spacing: -0.5px;
	}
	.lab-title > div > span {
		font:
			13px/1.5 'DM Sans',
			sans-serif;
	}
	.study-label {
		display: none;
	}
	.setup-panel {
		border: 0;
	}
	.archive-panel {
		max-height: none;
		padding: 4px 16px 16px;
	}
	.section-label {
		padding: 16px 0 12px;
		gap: 8px;
	}
	.section-label h2 {
		font-size: 14px;
	}
	.section-label > span {
		display: none;
	}
	.archive-item {
		padding: 12px;
		margin-bottom: 8px;
		border-radius: 8px;
	}
	.archive-item small {
		font:
			12px/1.6 'DM Sans',
			sans-serif;
	}
	.preset-list {
		padding: 16px;
		gap: 10px;
	}
	.preset-list button {
		padding: 14px;
		border-radius: 8px;
	}
	.initialize-options {
		padding: 0 16px 18px;
		grid-template-columns: 1fr 100px;
		gap: 12px;
	}
	.initialize-options p {
		font:
			13px/1.65 'DM Sans',
			sans-serif;
	}
	.resume-row:not(:empty) {
		padding: 16px;
		border-bottom: 1px solid var(--line);
	}
	.resume-row .initialize {
		width: 100%;
	}
	.transport {
		padding: 18px 16px;
		flex-wrap: wrap;
		gap: 12px;
	}
	.transport > span {
		width: 100%;
		font:
			13px 'DM Sans',
			sans-serif;
	}
	.transport .segmented {
		flex-wrap: wrap;
	}
	.corpus-card {
		padding: 0 16px 18px;
		border: 0;
	}
	.story-inspector {
		border: 0;
	}
	.view-toolbar {
		padding: 12px 16px;
		gap: 12px;
		flex-wrap: wrap;
		min-height: 60px;
	}
	.segmented button {
		min-height: 32px;
		padding: 6px 10px;
	}
	.layer-options {
		flex-wrap: wrap;
		gap: 5px;
	}
	.layer-options button {
		min-width: 30px;
		min-height: 30px;
	}
	.architecture-label {
		display: none;
	}
	.story-field {
		height: clamp(390px, 48vh, 640px);
	}
	.story-lower {
		grid-template-columns: minmax(0, 1.4fr) minmax(280px, 1fr);
		border: 0;
	}
	.view-note,
	.geometry-footer,
	.timeline {
		padding: 14px 18px;
	}
	.field-empty p {
		font-size: 15px;
		max-width: 480px;
	}
	.story-status {
		padding: 12px 24px 20px;
		flex-wrap: wrap;
		gap: 10px;
	}
	@media (max-width: 1000px) {
		.story-inspector {
			grid-column: auto;
		}
		.story-lower {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	@media (max-width: 600px) {
		.story-toolbar {
			padding: 18px 12px;
		}
		.story-field {
			height: 400px;
		}
		.view-toolbar {
			padding: 10px;
			gap: 8px;
		}
		.lab-title > :global(.icon) {
			display: none;
		}
	}

	.map-context {
		padding: 10px 18px;
		display: flex;
		gap: 12px;
		justify-content: space-between;
		flex-wrap: wrap;
		color: var(--muted);
		font-size: 12px;
		border-top: 1px solid var(--line);
	}
	.setup-panel > .lab-disclosure > .section-label {
		padding-left: 16px;
	}
	@media (max-width: 700px) {
		.setup-panel {
			display: block;
		}
		.initialize-options {
			display: grid;
			grid-template-columns: minmax(0, 1fr) 100px;
		}
		.initialize-options p,
		.backend-options,
		.control-caption,
		.initialize {
			grid-column: 1 / -1;
		}
		.initialize-options > label {
			grid-column: 1;
		}
	}

	.welcome-actions {
		display: flex;
		justify-content: center;
		gap: 10px;
		flex-wrap: wrap;
		margin: 20px 0;
	}
	.training-progress {
		margin-left: auto;
	}

	@media (max-width: 700px) {
		.preset-list {
			flex-direction: column;
		}
		.archive-panel {
			display: block;
		}
		.archive-item {
			width: 100%;
		}
		.backend-options {
			grid-row: auto;
		}
		.sample-controls {
			padding: 14px;
			flex-wrap: wrap;
		}
		.sample-controls > input:first-of-type {
			flex: 1 1 100%;
			width: 100%;
		}
		.corpus-card {
			display: block;
		}
	}
</style>
