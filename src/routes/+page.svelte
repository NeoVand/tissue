<script lang="ts">
	import ResearchWorkspace from '$lib/components/ResearchWorkspace.svelte';
	let toolsCollapsed = $state(false);
	let toolPanel = $state<'model' | 'inspector'>('model');
	let bindingTrainingOpen = $state(false);
	let repairSetup = $state(false);
	let commandHeight = $state(92);
	function openBindingTools(panel: 'model' | 'inspector') {
		toolPanel = panel;
		toolsCollapsed = false;
	}
	import { onMount } from 'svelte';
	import { resolve } from '$app/paths';
	import { publicAsset } from '$lib/deployment/public-assets';
	import NeuralField from '$lib/components/NeuralField.svelte';
	import Icon from '$lib/components/Icon.svelte';
	import ModelAnatomy from '$lib/components/ModelAnatomy.svelte';
	import ModelInspector from '$lib/components/ModelInspector.svelte';
	import ActivationHeatmap from '$lib/components/ActivationHeatmap.svelte';
	import ResearchJournal from '$lib/components/ResearchJournal.svelte';
	import ResearchMethods from '$lib/components/ResearchMethods.svelte';
	import QueryShiftStudy from '$lib/components/QueryShiftStudy.svelte';
	import TinyStoriesWorkbench from '$lib/components/TinyStoriesWorkbench.svelte';
	import { QueryAnalysisEngine } from '$lib/lab/query-analysis-engine';
	import type { QueryAnalysis } from '$lib/lab/query-analysis';
	import {
		readQueryStudies,
		saveQueryStudy,
		parseQueryStudy,
		exportQueryStudy,
		type QueryStudyRecord
	} from '$lib/lab/query-journal';
	import FingerprintComparison from '$lib/components/FingerprintComparison.svelte';
	import LearningCurve from '$lib/components/LearningCurve.svelte';
	import { Engine } from '$lib/lab/engine';
	import { getNeighbors, type GeometryResult } from '$lib/lab/geometry';
	import { GeometryEngine } from '$lib/lab/geometry-engine';
	import {
		MODEL_CONFIG,
		NEURON_COUNT,
		type EngineEvent,
		type Metrics,
		type Probe
	} from '$lib/lab/protocol';
	import {
		exportRun,
		parseRun,
		readRuns,
		getJournalWarnings,
		saveRun,
		type MapFrame,
		type RunRecord,
		type Snapshot
	} from '$lib/lab/journal';
	import { notebook } from '$lib/lab/notebook';
	import { references } from '$lib/lab/references';

	let tab = $state<'observatory' | 'journal' | 'methods' | 'queries' | 'stories'>('observatory');
	let storyVisited = $state(false);
	let storyBusy = $state(false);
	let phase = $state<
		'booting' | 'ready' | 'training' | 'measuring' | 'probing' | 'repairing' | 'querying' | 'error'
	>('booting');
	let status = $state('Initializing the binding model…');
	let error = $state('');
	let storageError = $state('');
	let seed = $state(42);
	let budget = $state(2000);
	let mode = $state<'activation' | 'effect'>('activation');
	let selected = $state<number | null>(null);
	let theme = $state<'dark' | 'light'>('dark');
	let layout = $state<'functional' | 'model'>('functional');
	let layerFilter = $state<number | null>(null);
	let showEdges = $state(true);
	let bottomTab = $state<'activations' | 'repairs' | 'findings'>('activations');
	let renderStats = $state<{
		nodes: number;
		edges: number;
		drawCalls: number;
		frameMs: number;
	} | null>(null);
	let queryRecords = $state.raw<QueryStudyRecord[]>([]);
	let queryRecord = $state.raw<QueryStudyRecord | null>(null);
	let queryAnalysis = $state.raw<QueryAnalysis | null>(null);
	let queryBusy = $state(false);
	let queryActivity = $state<'measurement' | 'analysis' | 'loading'>('loading');
	let queryStatus = $state('Ready to measure a paired-query study.');
	let queryError = $state('');
	let queryProgress = $state({ completed: 0, total: 512 });
	let querySelected = $state<number | null>(0);
	let queryInput: HTMLInputElement;
	let queryAnalysisEngine: QueryAnalysisEngine | undefined;
	let queryGeneration = 0;
	let token = $state(13);
	let example = $state(0);
	let cursor = $state(-1);
	let interventionSetup = $state(false);
	let run = $state.raw<RunRecord | null>(null);
	let runs = $state.raw<RunRecord[]>([]);
	let currentProbe = $state.raw<Probe | null>(null);
	let lesionProbe = $state.raw<Probe | null>(null);
	let progress = $state({ completed: 0, total: 256 });
	let note = $state('');
	let savedAt = $state('');
	let importInput: HTMLInputElement;
	let engine: Engine | undefined;
	let geometryEngine: GeometryEngine | undefined;
	let activationGeometry = $state.raw<GeometryResult | undefined>(undefined);
	let effectGeometry = $state.raw<GeometryResult | undefined>(undefined);
	let stopRequested = false;
	let generation = 0;
	let mounted = false;

	let busy = $derived((phase !== 'ready' && phase !== 'error') || storyBusy);
	let ready = $derived(phase === 'ready' && !storyBusy);
	let lastSnapshot = $derived(run?.snapshots.at(-1));
	let snapshot = $derived(cursor < 0 ? lastSnapshot : run?.snapshots[cursor]);
	let historical = $derived(!!snapshot && snapshot.step !== lastSnapshot?.step);
	let metrics = $derived(historical ? snapshot?.metrics : run?.metrics.at(-1));
	let map = $derived(mode === 'effect' ? snapshot?.effect : snapshot?.activation);
	let probe = $derived(
		historical ? snapshot?.probe : (lesionProbe ?? currentProbe ?? snapshot?.probe)
	);
	let points = $derived(
		map?.positions
			.map((position, id) => ({
				id,
				position:
					layout === 'model'
						? ([
								Math.floor(id / MODEL_CONFIG.hidden) * 1.3 - 0.65 + ((id % 16) - 7.5) * 0.062,
								(Math.floor((id % MODEL_CONFIG.hidden) / 16) - 3.5) * 0.12,
								0
							] as [number, number, number])
						: position,
				layer: Math.floor(id / MODEL_CONFIG.hidden),
				activation: probe?.activations[token]?.[id] ?? 0,
				effect: mode === 'effect' ? map.magnitudes[id] : undefined
			}))
			.filter((point) => layout === 'model' || map?.valid[point.id]) ?? []
	);
	let neighbors = $derived(selected === null ? [] : (map?.neighbors[selected] ?? []));
	let inspectedCheckpoint = $derived(
		run?.checkpoint?.step === probe?.step ? run?.checkpoint : undefined
	);
	let neighborRows = $derived.by(() => {
		if (selected === null) return [];
		const geometry = !historical
			? mode === 'effect'
				? effectGeometry
				: activationGeometry
			: undefined;
		return geometry
			? getNeighbors(geometry, selected, 5).map((item) => ({
					id: item.index,
					distance: item.distance
				}))
			: neighbors.map((id) => ({ id, distance: null }));
	});
	function setTheme(next: 'dark' | 'light') {
		theme = next;
		document.documentElement.dataset.theme = next;
		try {
			localStorage.setItem('tissue-theme', next);
		} catch {
			/* Theme still applies for this session. */
		}
	}
	function selectUnit(id: number) {
		selected = Math.max(0, Math.min(NEURON_COUNT - 1, Math.round(id)));
		if (layerFilter !== null && Math.floor(selected / MODEL_CONFIG.hidden) !== layerFilter)
			layerFilter = null;
	}
	function stepPrompt(delta: number) {
		if (!ready || historical) return;
		example = (example + delta + 16) % 16;
		void changeExample();
	}
	let observedEntries = $derived(runs.reduce((sum, record) => sum + record.observations.length, 0));

	function compactGeometry(geometry: GeometryResult): MapFrame {
		return {
			positions: geometry.positions,
			edges: geometry.edges,
			magnitudes: geometry.magnitudes,
			valid: geometry.valid,
			neighbors: geometry.positions.map((_, i) => getNeighbors(geometry, i, 5).map((n) => n.index)),
			neighborRetention: geometry.neighborRetention,
			explainedVariance: geometry.explainedVariance,
			rank: geometry.rank,
			boundaryDegenerate: geometry.boundaryDegenerate,
			alignment: geometry.alignment
		};
	}
	function recordMetric(metric: Metrics) {
		if (!run) return;
		const history = run.metrics.filter((m) => m.step !== metric.step);
		run = {
			...run,
			metrics: [...history, metric].sort((a, b) => a.step - b.step),
			updatedAt: new Date().toISOString()
		};
	}
	function handleEvent(event: EngineEvent) {
		if (phase === 'querying') {
			if (event.type === 'status') queryStatus = event.message;
			if (event.type === 'measurement')
				queryProgress = { completed: event.completed, total: event.total };
		}
		if (event.type === 'metrics') recordMetric(event.metrics);
		if (event.type === 'status') status = event.message;
		if (event.type === 'measurement') progress = { completed: event.completed, total: event.total };
	}
	function observe(
		kind: RunRecord['observations'][number]['kind'],
		title: string,
		detail: string,
		measuredProbe?: Probe
	) {
		if (!run) return;
		run = {
			...run,
			observations: [
				...run.observations,
				{
					id: crypto.randomUUID(),
					time: new Date().toISOString(),
					step: run.metrics.at(-1)?.step ?? 0,
					kind,
					title,
					detail,
					probe: measuredProbe
				}
			]
		};
	}
	async function persist() {
		if (!run) return;
		const record = run;
		try {
			localStorage.setItem('tissue-active-run', record.id);
		} catch {
			/* IndexedDB remains the primary record. */
		}
		runs = [record, ...runs.filter((r) => r.id !== record.id)];
		try {
			await saveRun(record);
			savedAt = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
			storageError = '';
		} catch (reason) {
			storageError = `This run could not be saved: ${reason instanceof Error ? reason.message : String(reason)}`;
		}
	}
	function fail(reason: unknown) {
		const message = reason instanceof Error ? reason.message : String(reason);
		if (/cancelled/i.test(message)) {
			phase = 'ready';
			status = message;
			error = '';
			observe('note', 'Measurement cancelled', message);
			void persist();
			return;
		}
		error = reason instanceof Error ? reason.message : String(reason);
		phase = 'error';
		status = 'Experiment interrupted';
		observe(
			'failure',
			'An interrupted experiment',
			`${error} Last observed step: ${run?.metrics.at(-1)?.step ?? 0}; last complete checkpoint: ${run?.checkpoint?.step ?? 0}.`
		);
		void persist();
	}
	async function capture(includeEffects = false) {
		if (!engine || !run) return;
		const atlas = await engine.captureAtlas(includeEffects);
		activationGeometry = await geometryEngine!.build(atlas.activationFingerprints, 'activation');
		if (atlas.effectFingerprints)
			effectGeometry = await geometryEngine!.build(atlas.effectFingerprints, 'effect');
		else effectGeometry = undefined;
		currentProbe = await engine.probe(example);
		if (selected === null) selected = 0;
		lesionProbe = null;
		const checkpoint = await engine.exportCheckpoint();
		const latestMetrics = run.metrics.at(-1)!;
		const captured: Snapshot = {
			step: atlas.step,
			metrics: latestMetrics,
			activation: compactGeometry(activationGeometry),
			effect: effectGeometry ? compactGeometry(effectGeometry) : undefined,
			probe: currentProbe
		};
		run = {
			...run,
			atlas,
			checkpoint,
			snapshots: [...run.snapshots.filter((s) => s.step !== atlas.step), captured].sort(
				(a, b) => a.step - b.step
			)
		};
		if (!captured.effect) mode = 'activation';
		cursor = -1;
		await persist();
	}
	async function startNew(backend: 'auto' | 'wasm' = 'auto') {
		if (busy && phase !== 'booting') return;
		const currentGeneration = ++generation;
		phase = 'booting';
		error = '';
		status = 'Opening the model worker…';
		await engine?.dispose();
		if (!mounted || currentGeneration !== generation) return;
		engine = new Engine((event) => {
			if (currentGeneration === generation) handleEvent(event);
		});
		await geometryEngine?.reset();
		activationGeometry = undefined;
		effectGeometry = undefined;
		currentProbe = null;
		lesionProbe = null;
		selected = null;
		mode = 'activation';
		example = 0;
		token = 13;
		cursor = -1;
		try {
			const initialization = await engine.initialize(Number(seed), backend);
			if (!mounted || currentGeneration !== generation) return;
			const now = new Date().toISOString();
			run = {
				version: 1,
				id: crypto.randomUUID(),
				createdAt: now,
				updatedAt: now,
				title: `Binding study · seed ${seed}`,
				seed: Number(seed),
				initialization,
				metrics: [initialization.metrics],
				snapshots: [],
				observations: [],
				provenance: { source: 'browser', appVersion: '0.5.0', userAgent: navigator.userAgent }
			};
			observe(
				'measurement',
				'An untrained reference',
				`${initialization.parameterCount.toLocaleString()} parameters. ${initialization.backend.toUpperCase()} backend. Seed ${seed}. Calibration and evaluation assignment mappings are excluded from training.`
			);
			status = 'Measuring the untrained reference…';
			await capture();
			phase = 'ready';
			status = 'Ready to grow';
		} catch (reason) {
			if (mounted && currentGeneration === generation) fail(reason);
		}
	}
	async function train() {
		if (!engine || !run || !ready) return;
		phase = 'training';
		stopRequested = false;
		error = '';
		cursor = -1;
		mode = 'activation';
		lesionProbe = null;
		const initialStep = run.metrics.at(-1)?.step ?? 0;
		const target = initialStep + Number(budget);
		observe(
			'training',
			'A new growth interval',
			`Requested ${budget} updates. Fixed calibration probes every 50 updates; held-out answer evaluation every 25.`
		);
		try {
			while (!stopRequested && (run?.metrics.at(-1)?.step ?? 0) < target) {
				status = `Learning · target ${target} steps`;
				const result = await engine.train(Math.min(50, target - (run.metrics.at(-1)?.step ?? 0)));
				recordMetric(result);
				status = 'Recording this checkpoint…';
				await capture();
			}
			observe(
				'training',
				stopRequested ? 'Paused and preserved' : 'Growth interval complete',
				`Step ${run.metrics.at(-1)?.step}. Held-out accuracy ${percent(run.metrics.at(-1)?.accuracy)}. Answer loss ${decimal(run.metrics.at(-1)?.validationLoss)} nats.`
			);
			await persist();
			phase = 'ready';
			status = 'Checkpoint preserved';
		} catch (reason) {
			if (mounted) fail(reason);
		}
	}
	function pause() {
		stopRequested = true;
		status = 'Pausing after the current update…';
		void engine?.pause().catch(fail);
	}
	async function measureEffects() {
		if (!engine || !run || !ready) return;
		phase = 'measuring';
		cursor = -1;
		error = '';
		progress = { completed: 0, total: NEURON_COUNT };
		status = 'Measuring each neuron’s intervention fingerprint…';
		try {
			await capture(true);
			mode = 'effect';
			interventionSetup = false;
			observe(
				'measurement',
				'An intervention atlas',
				`Silenced each of ${NEURON_COUNT} neurons across all token positions on ${run.atlas?.examples.length} fixed calibration prompts. Coordinates summarize changes in answer probabilities. 3D neighbor retention: ${percent(effectGeometry?.neighborRetention)}.`
			);
			await persist();
			phase = 'ready';
			status = 'Intervention atlas ready';
		} catch (reason) {
			if (mounted) fail(reason);
		}
	}
	async function changeExample() {
		if (!engine || !ready || historical) return;
		phase = 'probing';
		lesionProbe = null;
		try {
			currentProbe = await engine.probe(example);
			phase = 'ready';
		} catch (reason) {
			fail(reason);
		}
	}
	async function silence() {
		if (!engine || selected === null || !ready || historical) return;
		phase = 'probing';
		error = '';
		try {
			const requestedNeuron = selected;
			const requestedExample = example;
			lesionProbe = await engine.probe(requestedExample, requestedNeuron);
			const answer = Number(lesionProbe.example.answer);
			const before = currentProbe?.probabilities[answer] ?? 0;
			const after = lesionProbe.probabilities[answer];
			observe(
				'intervention',
				`Silenced L${Math.floor(requestedNeuron / 128) + 1} · neuron ${requestedNeuron % 128}`,
				`Calibration prompt ${requestedExample + 1}. Correct answer ${answer}: ${(before * 100).toFixed(2)}% → ${(after * 100).toFixed(2)}%. Zeroed this unit across all positions for this forward pass. Weights unchanged.`,
				lesionProbe
			);
			await persist();
			phase = 'ready';
			status = 'Intervention recorded';
		} catch (reason) {
			fail(reason);
		}
	}
	async function openRun(record: RunRecord, navigate = true) {
		if (busy || !engine || !record.checkpoint) return;
		if (navigate) tab = 'observatory';
		phase = 'booting';
		status = 'Restoring model and optimizer…';
		error = '';
		try {
			const recovering = record.metrics.some((metric) => metric.step > record.checkpoint!.step);
			if (recovering) await saveRun(record);
			const restoredMetrics = await engine.loadCheckpoint(record.checkpoint);
			await geometryEngine?.reset();
			const now = new Date().toISOString();
			run = {
				...record,
				...(recovering
					? {
							id: crypto.randomUUID(),
							title: `${record.title} · recovered`,
							createdAt: now,
							updatedAt: now
						}
					: {}),
				metrics: record.metrics.filter((metric) => metric.step <= restoredMetrics.step)
			};
			recordMetric(restoredMetrics);
			if (recovering)
				observe(
					'note',
					'Recovered from durable weights',
					`Resumed checkpoint ${restoredMetrics.step} in a new run. The interrupted source ${record.id} retains all later measured metrics and failure evidence.`
				);
			seed = record.seed;
			cursor = -1;
			example = 0;
			token = 13;
			selected = record.repairs?.at(-1)?.lesionNeuron ?? 0;
			lesionProbe = null;
			activationGeometry = record.atlas
				? await geometryEngine!.build(
						record.atlas.activationFingerprints,
						'activation',
						record.snapshots.at(-1)?.activation
					)
				: undefined;
			effectGeometry = record.atlas?.effectFingerprints
				? await geometryEngine!.build(
						record.atlas.effectFingerprints,
						'effect',
						record.snapshots.at(-1)?.effect
					)
				: undefined;
			currentProbe = await engine.probe(0);
			mode = 'activation';
			if (recovering) await persist();
			phase = 'ready';
			status = 'Saved specimen restored';
			try {
				localStorage.setItem('tissue-active-run', run.id);
			} catch {
				/* Optional selection memory. */
			}
		} catch (reason) {
			fail(reason);
		}
	}
	async function importFile(event: Event) {
		const file = (event.currentTarget as HTMLInputElement).files?.[0];
		if (!file) return;
		try {
			const imported = parseRun(await file.text());
			const now = new Date().toISOString();
			const record = { ...imported, id: crypto.randomUUID(), createdAt: now, updatedAt: now };
			await saveRun(record);
			runs = [record, ...runs.filter((r) => r.id !== record.id)];
			await openRun(record);
		} catch (reason) {
			error = reason instanceof Error ? reason.message : String(reason);
		}
		(event.target as HTMLInputElement).value = '';
	}
	function addNote() {
		if (!note.trim() || !run) return;
		observe('note', 'A field note', note.trim());
		note = '';
		void persist();
	}
	function percent(value: number | null | undefined) {
		return value === null || value === undefined ? '—' : `${(value * 100).toFixed(1)}%`;
	}
	function decimal(value: number | null | undefined) {
		return value === null || value === undefined ? '—' : value.toFixed(3);
	}
	function selectCheckpoint(index: number) {
		cursor = index;
		lesionProbe = null;
		if (mode === 'effect' && !run?.snapshots[index]?.effect) mode = 'activation';
	}

	async function boot() {
		try {
			runs = await readRuns();
			storageError = getJournalWarnings().join(' ');
		} catch {
			storageError = 'Your browser could not open the local journal. You can still export runs.';
		}
		if (!mounted) return;
		let activeId: string | null = null;
		try {
			activeId = localStorage.getItem('tissue-active-run');
		} catch {
			/* Optional selection memory. */
		}
		const saved =
			runs.find((record) => record.id === activeId && record.checkpoint) ??
			runs.find((record) => record.checkpoint);
		if (saved?.checkpoint) {
			try {
				const currentGeneration = ++generation;
				engine = new Engine((event) => {
					if (currentGeneration === generation) handleEvent(event);
				});
				await engine.initialize(saved.seed);
				if (!mounted) return;
				phase = 'ready';
				await openRun(saved, false);
			} catch (reason) {
				if (mounted) fail(reason);
			}
		} else await startNew();
	}

	async function compareRepair() {
		if (!engine || !run?.atlas?.effectFingerprints || selected === null || !ready || historical)
			return;
		phase = 'repairing';
		bottomTab = 'repairs';
		error = '';
		status = 'Comparing four repair neighborhoods…';
		try {
			const result = await engine.controlledRepair({
				atlas: run.atlas,
				lesionNeuron: selected,
				steps: 50,
				neighborCount: 8
			});
			run = { ...run, repairs: [...(run.repairs ?? []), result] };
			observe(
				'measurement',
				'A controlled repair pilot',
				`Neuron ${result.lesionNeuron}; four neighborhoods of ${result.neighborCount} same-layer units; ${result.steps} updates per arm. Intact loss ${result.intact.loss.toFixed(4)}, injured loss ${result.lesioned.loss.toFixed(4)}. ${result.caveat}`
			);
			await persist();
			phase = 'ready';
			status = 'Repair comparison recorded; source model preserved';
		} catch (reason) {
			if (mounted) fail(reason);
		}
	}
	async function openReference(path = '/experiments/binding-seed-42.json') {
		if (busy) return;
		try {
			const response = await fetch(publicAsset(path));
			if (!response.ok) throw new Error('The reference experiment could not be loaded.');
			const source = parseRun(await response.text());
			const now = new Date().toISOString();
			const record: RunRecord = {
				...source,
				provenance: { ...source.provenance, source: 'browser', userAgent: navigator.userAgent },
				id: crypto.randomUUID(),
				createdAt: now,
				updatedAt: now,
				observations: [
					...source.observations,
					{
						id: crypto.randomUUID(),
						time: now,
						step: source.checkpoint?.step ?? 0,
						kind: 'note',
						title: 'A reference specimen, opened for exploration',
						detail: `Copied measured reference ${source.id}. Subsequent observations belong to this local copy.`
					}
				]
			};
			await saveRun(record);
			runs = [record, ...runs.filter((r) => r.id !== record.id)];
			await openRun(record);
		} catch (reason) {
			error = reason instanceof Error ? reason.message : String(reason);
		}
	}

	async function showQueryStudy(record: QueryStudyRecord, ticket: number) {
		if (!mounted || ticket !== queryGeneration) return;
		queryRecord = record;
		queryAnalysis = null;
		queryActivity = 'analysis';
		queryStatus = 'Comparing calibration neighborhoods on held-out assignments…';
		const result = await queryAnalysisEngine!.analyze(record.measurement);
		if (!mounted || ticket !== queryGeneration) return;
		queryAnalysis = result;
		queryStatus = `Study ready · seed ${record.measurement.seed} · step ${record.measurement.step}`;
		try {
			localStorage.setItem('tissue-active-query-study', record.id);
		} catch {
			/* Optional selection preference. */
		}
	}
	async function keepQueryStudy(record: QueryStudyRecord) {
		queryRecords = [record, ...queryRecords.filter((item) => item.id !== record.id)];
		try {
			await saveQueryStudy(record);
		} catch (reason) {
			storageError = `Query evidence is available to export, but could not be saved: ${String(reason)}`;
		}
	}
	async function openQueryStudy(record: QueryStudyRecord) {
		if (queryBusy) return;
		const ticket = ++queryGeneration;
		queryBusy = true;
		queryError = '';
		tab = 'queries';
		try {
			await showQueryStudy(record, ticket);
		} catch (reason) {
			if (ticket === queryGeneration) queryError = String(reason);
		} finally {
			if (mounted && ticket === queryGeneration) queryBusy = false;
		}
	}
	async function restoreQueryStudies() {
		try {
			const archive = await readQueryStudies();
			if (!mounted) return;
			queryRecords = archive.records;
			if (archive.warnings.length) queryError = archive.warnings.join(' ');
			let id: string | null = null;
			try {
				id = localStorage.getItem('tissue-active-query-study');
			} catch {
				/* Use latest. */
			}
			const record = archive.records.find((item) => item.id === id) ?? archive.records[0];
			if (record && !queryRecord && !queryBusy) {
				const ticket = ++queryGeneration;
				queryBusy = true;
				try {
					await showQueryStudy(record, ticket);
				} finally {
					if (mounted && ticket === queryGeneration) queryBusy = false;
				}
			}
		} catch (reason) {
			if (mounted) queryError = String(reason);
		}
	}
	async function openQueryReference(referenceSeed: number) {
		if (queryBusy) return;
		const ticket = ++queryGeneration;
		queryBusy = true;
		queryError = '';
		queryActivity = 'loading';
		queryStatus = `Loading recorded query study · seed ${referenceSeed}…`;
		try {
			const response = await fetch(
				publicAsset(`/experiments/query-shifts-seed-${referenceSeed}.json`)
			);
			if (!response.ok) throw new Error('The recorded query study could not be loaded.');
			const record = parseQueryStudy(await response.text());
			if (!mounted || ticket !== queryGeneration) return;
			await keepQueryStudy(record);
			await showQueryStudy(record, ticket);
		} catch (reason) {
			if (mounted && ticket === queryGeneration) queryError = String(reason);
		} finally {
			if (mounted && ticket === queryGeneration) queryBusy = false;
		}
	}
	async function measureQueryStudy() {
		if (!ready || historical || queryBusy || !engine || !run) return;
		const ticket = ++queryGeneration;
		queryBusy = true;
		queryError = '';
		queryProgress = { completed: 0, total: 512 };
		queryActivity = 'measurement';
		phase = 'querying';
		let captured = false;
		try {
			const measurement = await engine.measureQueryShifts();
			captured = true;
			if (!mounted || ticket !== queryGeneration) return;
			const record: QueryStudyRecord = {
				version: 1,
				kind: 'tissue-query-study',
				id: crypto.randomUUID(),
				createdAt: measurement.capturedAt,
				source: 'browser',
				measurement
			};
			queryRecord = record;
			queryAnalysis = null;
			await keepQueryStudy(record);
			observe(
				'measurement',
				'Paired-query intervention study recorded',
				`Study ${record.id}; 16 calibration and 16 held-out assignment groups, each queried a/b/c. All 256 units silenced across all positions. Seed ${measurement.seed}, step ${measurement.step}, checkpoint SHA-256 ${measurement.checkpointHash}. Full raw evidence is in the Query shifts archive.`
			);
			await persist();
			await showQueryStudy(record, ticket);
			status = 'Paired-query study recorded; source checkpoint preserved';
		} catch (reason) {
			if (!mounted || ticket !== queryGeneration) return;
			const message = String(reason);
			if (message.toLowerCase().includes('cancelled')) {
				queryStatus = 'Measurement cancelled; no partial study was recorded.';
				status = 'Query measurement cancelled';
				observe(
					'note',
					'Paired-query measurement cancelled',
					'No partial result was accepted. The resident checkpoint was preserved.'
				);
				await persist();
			} else {
				queryError = message;
				if (!captured) fail(reason);
			}
		} finally {
			if (mounted && ticket === queryGeneration) {
				queryBusy = false;
				if (phase === 'querying') phase = 'ready';
			}
		}
	}
	function cancelQueryStudy() {
		if (phase === 'querying' && queryActivity === 'measurement') {
			queryStatus = 'Cancelling paired-query measurement…';
			void engine?.pause().catch((reason) => {
				queryError = String(reason);
			});
		} else {
			queryGeneration++;
			queryAnalysisEngine?.dispose();
			queryAnalysisEngine = new QueryAnalysisEngine();
			queryBusy = false;
			if (phase === 'querying') phase = 'ready';
			queryStatus = 'Analysis cancelled; any completed raw measurement remains in the archive.';
		}
	}
	async function importQueryFile(event: Event) {
		const input = event.currentTarget as HTMLInputElement;
		const file = input.files?.[0];
		input.value = '';
		if (!file || queryBusy) return;
		queryActivity = 'loading';
		const ticket = ++queryGeneration;
		queryBusy = true;
		queryError = '';
		try {
			if (file.size > 25_000_000) throw new Error('Query study exceeds the 25 MB import limit.');
			const imported = parseQueryStudy(await file.text());
			// Imports receive a local archive ID so an external ID cannot overwrite earlier evidence.
			const record = { ...imported, id: crypto.randomUUID() };
			if (!mounted || ticket !== queryGeneration) return;
			await keepQueryStudy(record);
			await showQueryStudy(record, ticket);
		} catch (reason) {
			if (mounted && ticket === queryGeneration) queryError = String(reason);
		} finally {
			if (mounted && ticket === queryGeneration) queryBusy = false;
		}
	}
	const repairNames: Record<string, string> = {
		activation: 'Activation neighbors',
		intervention: 'Intervention neighbors',
		'outgoing-weight': 'Outgoing-weight neighbors',
		random: 'Random same-layer units'
	};
	let latestRepair = $derived(run?.repairs?.at(-1));
	onMount(() => {
		mounted = true;
		if (new URLSearchParams(location.search).get('view') === 'tinystories') {
			storyVisited = true;
			tab = 'stories';
		}
		theme = document.documentElement.dataset.theme === 'light' ? 'light' : 'dark';
		geometryEngine = new GeometryEngine();
		queryAnalysisEngine = new QueryAnalysisEngine();
		void restoreQueryStudies();
		void boot();
		return () => {
			mounted = false;
			generation++;
			queryGeneration++;
			queryAnalysisEngine?.dispose();
			stopRequested = true;
			geometryEngine?.dispose();
			void engine?.dispose();
		};
	});
</script>

<svelte:head
	><title>Tissue — interpretability workbench</title><meta
		name="description"
		content="A browser research lab linking model anatomy, activation geometry, interventions, and reproducible experiments."
	/></svelte:head
>

<div class="lab-shell">
	<header class="app-header">
		<a class="brand" href={resolve('/')} aria-label="Tissue home"
			><Icon name="atom" size={24} /><span>tissue<span class="brand-dot">.</span></span></a
		>

		<nav aria-label="Lab views">
			<button class:active={tab === 'observatory'} onclick={() => (tab = 'observatory')}
				><Icon name="cube" size={14} />Workbench</button
			>
			<button
				class:active={tab === 'stories'}
				onclick={() => {
					storyVisited = true;
					tab = 'stories';
				}}><Icon name="network" size={14} />TinyStories</button
			>
			<button class:active={tab === 'queries'} onclick={() => (tab = 'queries')}
				><Icon name="target" size={14} />Query shifts</button
			>
			<button class:active={tab === 'journal'} onclick={() => (tab = 'journal')}
				><Icon name="book" size={14} />Field journal<span class="count"
					>{notebook.length + observedEntries}</span
				></button
			>
			<button class:active={tab === 'methods'} onclick={() => (tab = 'methods')}
				><Icon name="info" size={14} />Methods</button
			>
		</nav>
		<div class="header-actions">
			<button
				class="icon-button"
				onclick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
				aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
				title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
				><Icon name={theme === 'dark' ? 'sun' : 'moon'} /></button
			><a
				class="icon-button"
				href="https://github.com/NeoVand/tissue"
				target="_blank"
				rel="noreferrer"
				aria-label="Open GitHub repository"><Icon name="github" /></a
			>
		</div>
	</header>
	{#if error}<div class="error-banner" role="alert">
			<Icon name="info" /><span>{error}</span>{#if phase === 'error'}{#if run?.checkpoint}<button
						class="secondary"
						onclick={() => run && openRun(run)}>Restore saved checkpoint</button
					>{/if}<button class="secondary" onclick={() => startNew('wasm')}>New run with WASM</button
				>{:else}<button class="icon-button" aria-label="Dismiss error" onclick={() => (error = '')}
					><Icon name="close" /></button
				>{/if}
		</div>{/if}
	{#if storageError}<div class="storage-banner" role="status">{storageError}</div>{/if}
	{#if storyVisited}<div hidden={tab !== 'stories'}>
			<TinyStoriesWorkbench
				{theme}
				active={tab === 'stories'}
				disabled={(phase !== 'ready' && phase !== 'error') || queryBusy}
				onbusy={(value) => (storyBusy = value)}
			/>
		</div>{/if}
	{#if tab === 'observatory'}
		<div class="command-bar" bind:clientHeight={commandHeight}>
			<div class="study-title">
				<span class="study-icon"><Icon name="network" size={19} /></span>
				<div>
					<h1>Binding transformer</h1>
					<span
						>Study 001 <span class="slash">/</span> seed {run?.seed ?? seed}
						<span class="slash">/</span>
						{historical ? 'recorded checkpoint' : 'live model'}</span
					>
				</div>
			</div>
			<div class="lab-actions" role="toolbar" aria-label="Binding actions">
				<button
					class="primary"
					onclick={phase === 'training' ? pause : train}
					disabled={phase !== 'training' && (!ready || !run)}
					><Icon name={phase === 'training' ? 'pause' : 'activity'} size={17} />{phase ===
					'training'
						? 'Pause training'
						: metrics?.step
							? 'Continue training'
							: 'Start training'}</button
				>
				<button
					onclick={() => openBindingTools('inspector')}
					aria-pressed={!toolsCollapsed && toolPanel === 'inspector'}
					><Icon name="search" size={17} />Probe</button
				>
				<button
					onclick={() => {
						if (snapshot?.effect) mode = 'effect';
						else interventionSetup = true;
					}}><Icon name="target" size={17} />Intervene</button
				>
				<button
					onclick={() => {
						repairSetup = !repairSetup;
						openBindingTools('inspector');
					}}><Icon name="flask" size={17} />Repair</button
				>
				<button
					onclick={() => openBindingTools('model')}
					aria-pressed={!toolsCollapsed && toolPanel === 'model'}
					><Icon name="settings" size={17} />Model</button
				>
				<button
					onclick={() => {
						storyVisited = true;
						tab = 'stories';
					}}><Icon name="play" size={17} />Generate stories</button
				>
			</div>
		</div>
		<ResearchWorkspace
			name="binding"
			topOffset={commandHeight + 16}
			externalControls
			toolTitle={toolPanel === 'model' ? 'Model & runs' : 'Prompt & unit inspection'}
			bind:panel={toolPanel}
			bind:collapsed={toolsCollapsed}
		>
			{#snippet model()}
				<aside class="model-sidebar">
					<div class="file-actions">
						<button
							class="icon-button"
							onclick={() => run && exportRun(run)}
							disabled={!run?.checkpoint || busy}
							aria-label="Export experiment"
							title="Export experiment"><Icon name="download" /></button
						><button
							class="icon-button"
							onclick={() => importInput.click()}
							disabled={busy}
							aria-label="Import experiment"
							title="Import experiment"><Icon name="upload" /></button
						>
					</div>
					<div class="panel-title">
						<Icon name="layers" size={14} />
						<h2>Model anatomy</h2>
						<span class="small-tag">25.9K</span>
					</div>
					<div class="architecture">
						<ModelAnatomy
							compact
							{selected}
							{layerFilter}
							onlayer={(value) => (layerFilter = value)}
						/>
					</div>
					<div class="model-meta">
						<span><b>2</b> blocks</span><span><b>4</b> heads</span><span><b>32</b> width</span>
					</div>
					<section class="reference-list">
						<div class="sidebar-label">Recorded experiments</div>
						{#each references as reference (reference.seed)}<button
								class="reference-button"
								onclick={() => openReference(reference.path)}
								disabled={busy}
								aria-label={reference.seed === 42
									? 'Explore that specimen'
									: `Explore seed ${reference.seed}`}
								><span class="reference-symbol"><Icon name="flask" size={14} /></span><span
									><b>Binding · seed {reference.seed}</b><small
										>{(reference.accuracy * 100).toFixed(1)}% held-out · {reference.step.toLocaleString()}
										steps</small
									></span
								><Icon name="right" size={12} /></button
							>{/each}
					</section>
					<div class="new-run">
						<label for="new-seed">New seed</label><input
							id="new-seed"
							aria-label="New run seed"
							type="number"
							min="0"
							max="999999"
							step="1"
							bind:value={seed}
							disabled={busy}
						/><button
							class="icon-button"
							onclick={() => startNew()}
							disabled={busy}
							aria-label="New run"
							title="Initialize new run"><Icon name="reset" size={15} /></button
						>
					</div>
					<div class="sidebar-note">
						<Icon name="info" size={13} /><span
							>The atlas represents MLP channels. Select a block to isolate its units.</span
						>
					</div>
					<details class="lab-disclosure" bind:open={bindingTrainingOpen}>
						<summary>Training settings</summary>
						<div class="training-controls">
							<span class="control-label">Updates</span>
							<div class="segmented compact" aria-label="Growth interval">
								{#each [100, 600, 2000] as steps (steps)}<button
										class:chosen={budget === steps}
										disabled={busy}
										onclick={() => (budget = steps)}>{steps.toLocaleString()}</button
									>{/each}
							</div>
						</div>
					</details>
				</aside>
			{/snippet}
			<section class="spatial-panel">
				{#if repairSetup}<div
						class="action-setup"
						role="region"
						aria-label="Repair experiment setup"
					>
						<div>
							<strong>Compare repair neighborhoods</strong>
							<p>
								{!run?.atlas?.effectFingerprints
									? 'Measure the intervention atlas first to compare functional, causal, geometric and random neighborhoods.'
									: selected === null
										? 'Select a unit in the map or inspector to use as the lesion target.'
										: `Lesion unit ${selected}, then compare four repair neighborhoods with the same training budget.`}
							</p>
						</div>
						{#if !run?.atlas?.effectFingerprints}<button
								class="secondary"
								onclick={measureEffects}
								disabled={!ready || !run || historical}>Measure intervention atlas</button
							>{:else}<button
								class="primary"
								onclick={compareRepair}
								disabled={!ready || selected === null || historical}
								>Compare repair neighborhoods</button
							>{/if}
						<button
							class="icon-button"
							onclick={() => (repairSetup = false)}
							aria-label="Close repair setup"><Icon name="close" /></button
						>
					</div>{/if}
				<div class="map-toolbar">
					<div class="segmented" aria-label="Coordinate system">
						<button class:chosen={layout === 'functional'} onclick={() => (layout = 'functional')}
							><Icon name="cube" size={13} />Functional</button
						><button class:chosen={layout === 'model'} onclick={() => (layout = 'model')}
							><Icon name="network" size={13} />Model layout</button
						>
					</div>
					<div class="map-tools">
						<button
							class="icon-button"
							class:enabled={showEdges}
							onclick={() => (showEdges = !showEdges)}
							disabled={layout === 'model'}
							aria-label="Toggle similarity edges"
							aria-pressed={showEdges}
							title="Similarity edges"><Icon name="network" size={14} /></button
						>
						<div class="segmented" aria-label="Fingerprint">
							<button class:chosen={mode === 'activation'} onclick={() => (mode = 'activation')}
								>Activation</button
							><button
								class:chosen={mode === 'effect'}
								onclick={() => {
									if (snapshot?.effect) mode = 'effect';
									else interventionSetup = !interventionSetup;
								}}
								aria-pressed={mode === 'effect'}>Intervention</button
							>
						</div>
					</div>
				</div>
				{#if interventionSetup && !snapshot?.effect}<div
						class="intervention-setup"
						role="region"
						aria-label="Intervention map setup"
					>
						<div>
							<strong>Measure an intervention map</strong>
							<p>
								Activation similarity does not measure causal effects. This map requires silencing
								each of the 256 units and measuring changes in predictions.
							</p>
							{#if historical}<p>
									Select the latest checkpoint to measure new effects.
								</p>{:else if !ready}<p role="status">{status}</p>{/if}
						</div>
						{#if historical}<button class="secondary" onclick={() => (cursor = -1)}
								>Return to latest checkpoint</button
							>{:else}<button class="primary" onclick={measureEffects} disabled={!ready || !run}
								>{phase === 'measuring'
									? `Measuring ${progress.completed}/${progress.total}…`
									: 'Measure 256 interventions'}</button
							>{/if}
					</div>{/if}
				<div class="metric-strip">
					<div>
						<span>Held-out accuracy</span><strong data-testid="accuracy"
							>{percent(metrics?.accuracy)}</strong
						>
					</div>
					<div>
						<span>Answer loss <small>nats</small></span><strong data-testid="loss"
							>{decimal(metrics?.validationLoss)}</strong
						>
					</div>
					<div>
						<span>Training step</span><strong data-testid="step">{metrics?.step ?? 0}</strong>
					</div>
					<div>
						<span>Observed units</span><strong
							>{map
								? points.filter((p) => layerFilter === null || p.layer === layerFilter).length
								: '—'}<small> / {layerFilter === null ? 256 : 128}</small></strong
						>
					</div>
				</div>
				<div class="field-wrap">
					<NeuralField
						{points}
						edges={layout === 'functional' && showEdges ? (map?.edges ?? []) : []}
						{selected}
						onselect={(id) => {
							selectUnit(id);
							toolPanel = 'inspector';
							toolsCollapsed = false;
						}}
						{mode}
						{theme}
						{layerFilter}
						geometryLabel={`${layout === 'model' ? 'MODEL LAYOUT / LAYER × CHANNEL' : 'FUNCTIONAL SPACE / 3D PCA'} · ATLAS STEP ${snapshot?.step ?? '—'}`}
						loading={!map}
						onstats={(value) => (renderStats = value)}
					/>
				</div>
				<div class="projection-strip">
					<span><i class="layer-dot first"></i>Layer 1 <i class="layer-dot second"></i>Layer 2</span
					><span
						>{layout === 'functional'
							? map
								? `${NEURON_COUNT - points.length} undefined omitted`
								: 'Preparing atlas'
							: 'Channel order · not functional distance'}</span
					>{#if layout === 'functional'}<span class="quality"
							>Neighbors retained <b>{percent(map?.neighborRetention)}</b><span class="divider"
							></span>Variance <b>{percent(map?.explainedVariance)}</b></span
						>{/if}
				</div>
				<div class="prompt-strip">
					<div class="prompt-stepper">
						<button
							class="icon-button"
							onclick={() => stepPrompt(-1)}
							disabled={!ready || historical}
							aria-label="Previous calibration prompt"><Icon name="left" size={13} /></button
						><span
							>PROBE {String((probe?.exampleIndex ?? example) + 1).padStart(2, '0')}<small
								>/16</small
							></span
						><button
							class="icon-button"
							onclick={() => stepPrompt(1)}
							disabled={!ready || historical}
							aria-label="Next calibration prompt"><Icon name="right" size={13} /></button
						>
					</div>
					<div class="token-stream" aria-label="Input tokens">
						{#each probe?.example.tokens ?? [] as text, index (index)}<button
								class:chosen={token === index}
								onclick={() => (token = index)}
								aria-label={`Inspect token ${index + 1}: ${text}`}
								title={`Position ${index + 1}`}>{text}</button
							>{/each}<Icon name="right" size={12} /><span class="answer-token"
							>{probe?.example.answer ?? '?'}</span
						>
					</div>
					<span class="token-position">t = {token}</span>
				</div>
				<div class="timeline">
					<Icon name="activity" size={13} /><span class="timeline-label">CHECKPOINT</span>
					<div class="checkpoints">
						{#each run?.snapshots ?? [] as saved, index (saved.step)}<button
								class:current={snapshot?.step === saved.step}
								onclick={() => selectCheckpoint(index)}
								aria-label={`View checkpoint ${saved.step}`}><i></i>{saved.step}</button
							>{/each}
					</div>
					{#if historical}<button class="text-button" onclick={() => (cursor = -1)}
							>Return to live<Icon name="right" size={12} /></button
						>{:else}<span class="live-tag">LIVE</span>{/if}
				</div>
			</section>
			{#snippet inspector()}
				<aside class="inspector-sidebar">
					<div class="panel-title">
						<Icon name="target" size={14} />
						<h2>Unit inspector</h2>
						<span class="small-tag">MLP</span>
					</div>
					<div class="unit-picker">
						<label for="neuron-id">Neuron ID</label><button
							class="icon-button"
							onclick={() => selectUnit((selected ?? 1) - 1)}
							aria-label="Previous neuron"><Icon name="left" size={13} /></button
						><input
							id="neuron-id"
							aria-label="Neuron ID"
							type="number"
							min="0"
							max="255"
							placeholder="0–255"
							value={selected ?? ''}
							oninput={(event) => {
								const value = event.currentTarget.valueAsNumber;
								if (Number.isFinite(value)) selectUnit(value);
							}}
						/><button
							class="icon-button"
							onclick={() => selectUnit((selected ?? -1) + 1)}
							aria-label="Next neuron"><Icon name="right" size={13} /></button
						>
					</div>
					<div class="inspector-scroll">
						<ModelInspector
							compact
							{selected}
							checkpoint={inspectedCheckpoint}
							{probe}
							{token}
							{theme}
							onselect={selectUnit}
						/>
						{#if selected !== null}<div class="neighbors">
								<div class="sidebar-label">Nearest in fingerprint space</div>
								{#if neighborRows.length}<div class="neighbor-head">
										<span>CHANNEL</span><span>DISTANCE</span>
									</div>
									{#each neighborRows as neighbor (neighbor.id)}<button
											onclick={() => selectUnit(neighbor.id)}
											><span
												><i class:second={neighbor.id >= 128}></i>L{Math.floor(neighbor.id / 128) +
													1} /
												{String(neighbor.id % 128).padStart(3, '0')}</span
											><span
												>{neighbor.distance === null ? '—' : neighbor.distance.toFixed(3)}<Icon
													name="right"
													size={11}
												/></span
											></button
										>{/each}{:else}<p>No defined neighborhood for this fingerprint.</p>{/if}
							</div>{/if}
						<div class="prediction">
							<div class="sidebar-label">
								Next-token probabilities <span
									>{lesionProbe && !historical ? 'LESION' : 'INTACT'}</span
								>
							</div>
							<div class="probability-bars" aria-label="Answer probabilities">
								{#each probe?.probabilities ?? Array(8).fill(0) as probability, index (index)}<div
										class:correct={String(index) === probe?.example.answer}
									>
										<span>{(probability * 100).toFixed(0)}<small>%</small></span>
										<div class="prob-track">
											<i style:height={`${Math.max(1, probability * 100)}%`}></i>
										</div>
										<b
											>{index}{#if String(index) === probe?.example.answer}<i class="answer-dot"
												></i>{/if}</b
										>
									</div>{/each}
							</div>
							<p>Other tokens {percent(probe?.otherProbability)} · dot = target</p>
							{#if lesionProbe && !historical}<button
									class="text-button"
									onclick={() => (lesionProbe = null)}
									><Icon name="reset" size={12} />Restore intact view</button
								>{/if}
						</div>
					</div>
					<div class="intervention-actions">
						<button
							class="secondary"
							onclick={silence}
							disabled={selected === null || !ready || historical}
							><Icon name="target" size={13} />Silence selected unit</button
						><button class="secondary" onclick={measureEffects} disabled={!ready || !run}
							><Icon name="grid" size={13} />Measure all effects</button
						><button
							class="text-button repair-action"
							hidden={repairSetup}
							onclick={compareRepair}
							disabled={!ready ||
								selected === null ||
								historical ||
								!run?.atlas?.effectFingerprints}
							><Icon name="flask" size={13} />Compare repair neighborhoods<Icon
								name="right"
								size={12}
							/></button
						>
					</div>
				</aside>
			{/snippet}
			{#snippet evidence()}
				<details class="lab-disclosure evidence-drawer" open>
					<summary
						>Analysis & evidence <small>Activations, learning curves and repair experiments</small
						></summary
					>
					<section class="bottom-panel">
						<div class="bottom-tabs">
							<button
								class:active={bottomTab === 'activations'}
								onclick={() => (bottomTab = 'activations')}
								><Icon name="grid" size={13} />Activation matrix</button
							><button
								class:active={bottomTab === 'repairs'}
								onclick={() => (bottomTab = 'repairs')}
								><Icon name="flask" size={13} />Repair experiments{#if run?.repairs?.length}<span
										class="count">{run.repairs.length}</span
									>{/if}</button
							><button
								class:active={bottomTab === 'findings'}
								onclick={() => (bottomTab = 'findings')}
								><Icon name="chart" size={13} />Findings</button
							><span
								>{#if bottomTab === 'findings'}Recorded references · step 2,000{:else}Checkpoint {probe?.step ??
										'—'} · {historical ? 'recorded probe' : 'calibration only'}{/if}</span
							>
						</div>
						{#if bottomTab === 'activations'}<div class="activation-layout">
								<div class="heatmap-wrap">
									<ActivationHeatmap
										{probe}
										{selected}
										{token}
										onselect={selectUnit}
										ontoken={(value) => (token = value)}
										{theme}
										{layerFilter}
									/>
								</div>
								<div class="learning-panel">
									<div class="sidebar-label">
										<Icon name="chart" size={13} />Held-out learning curve
									</div>
									<LearningCurve metrics={run?.metrics ?? []} selectedStep={snapshot?.step ?? 0} />
									<div class="curve-key">
										<span><i></i>Answer loss</span><span
											><i class="dashed"></i>Uniform answer prior</span
										>
									</div>
									<p>96 fixed evaluation prompts<br />Chance 12.5% · input-copy baseline 33.3%</p>
								</div>
							</div>
						{:else if bottomTab === 'findings'}<div class="findings-wrap">
								<FingerprintComparison />
							</div>{:else}<div class="repair-results">
								{#if latestRepair}<div class="repair-baseline">
										<span
											>SEED {latestRepair.seed} / STEP {latestRepair.step} / UNIT {latestRepair.lesionNeuron}</span
										><span
											>Intact <b>{latestRepair.intact.loss.toFixed(4)}</b><Icon
												name="right"
												size={11}
											/>lesioned <b>{latestRepair.lesioned.loss.toFixed(4)}</b></span
										>
									</div>
									<div class="repair-body">
										<table>
											<thead
												><tr><th>Neighborhood</th><th>Loss</th><th>Accuracy</th><th>Weights</th></tr
												></thead
											><tbody
												>{#each latestRepair.arms as arm (arm.method)}<tr
														><td>{repairNames[arm.method]}</td><td
															>{arm.curve.at(-1)?.loss.toFixed(4)}</td
														><td>{percent(arm.curve.at(-1)?.accuracy)}</td><td
															>{arm.trainableParameters}</td
														></tr
													>{/each}</tbody
											>
										</table>
										<div class="repair-note">
											<span class="eyebrow">Pilot / not a validated finding</span>
											<p>
												{latestRepair.caveat} Improvement beyond intact loss also requires an unlesioned
												fine-tuning control.
											</p>
											<details>
												<summary>Neighborhood membership</summary
												>{#each latestRepair.arms as arm (arm.method)}<p>
														<b>{repairNames[arm.method]}:</b>
														{arm.neurons.join(', ')}
													</p>{/each}
											</details>
										</div>
									</div>{:else}<div class="empty-repair">
										<Icon name="flask" size={23} />
										<div>
											<h3>Compare recovery across neighborhoods.</h3>
											<p>
												Select a neuron, measure its intervention fingerprint, then compare four
												neighborhoods of eight same-layer units over 50 updates. Every arm starts
												from the same checkpoint.
											</p>
										</div>
									</div>{/if}
							</div>{/if}
					</section>
				</details>
			{/snippet}
		</ResearchWorkspace>
		<footer class="status-bar">
			<span class="operation-status" aria-live="polite"><i class:working={busy}></i>{status}</span
			>{#if phase === 'measuring'}<span class="measurement-progress"
					><progress max={progress.total} value={progress.completed}
					></progress>{progress.completed}/{progress.total}</span
				>{/if}{#if phase === 'measuring' || phase === 'repairing'}<button
					class="text-button"
					onclick={() => {
						void engine?.pause().catch(fail);
					}}>Cancel measurement</button
				>{/if}<span class="status-right"
				>{#if renderStats}<span>{renderStats.drawCalls} draw calls</span>{/if}<span
					>{savedAt ? `Saved ${savedAt}` : 'Local journal'}</span
				><Icon name="check" size={12} /></span
			>
		</footer>
	{:else if tab === 'queries'}
		{#if queryError}<div class="query-error" role="alert">
				<Icon name="info" />{queryError}<button
					class="icon-button"
					aria-label="Dismiss study error"
					onclick={() => (queryError = '')}><Icon name="close" /></button
				>
			</div>{/if}
		{#snippet savedStudies()}
			<div class="query-history" aria-label="Saved query studies">
				<span><Icon name="book" size={12} />Saved studies</span
				>{#each queryRecords as record (record.id)}<button
						class:current={queryRecord?.id === record.id}
						onclick={() => openQueryStudy(record)}
						disabled={queryBusy}
						title={record.createdAt}
						>Seed {record.measurement.seed} · {record.measurement.step}
						<small
							>{record.source === 'reference'
								? 'reference'
								: new Date(record.createdAt).toLocaleTimeString([], {
										hour: '2-digit',
										minute: '2-digit'
									})}</small
						></button
					>{/each}
			</div>
		{/snippet}
		<QueryShiftStudy
			savedStudies={queryRecords.length ? savedStudies : undefined}
			measurement={queryRecord?.measurement ?? null}
			analysis={queryAnalysis}
			busy={queryBusy}
			status={queryStatus}
			activity={queryActivity}
			progress={queryProgress}
			{theme}
			selected={querySelected}
			onselect={(id) => (querySelected = id)}
			onrun={measureQueryStudy}
			oncancel={cancelQueryStudy}
			onreference={openQueryReference}
			onexport={() => queryRecord && exportQueryStudy(queryRecord)}
			onimport={() => queryInput.click()}
			source={queryRecord?.source === 'reference' ? 'reference' : 'current'}
			canRun={ready && !historical}
			checkpoint={run?.checkpoint
				? { seed: run.seed, step: run.checkpoint.step, backend: run.initialization.backend }
				: null}
		/>
		{#if !queryBusy}<p class="query-operation-status" role="status">{queryStatus}</p>{/if}
	{:else if tab === 'journal'}<ResearchJournal
			{run}
			{runs}
			{busy}
			onopen={openRun}
			onreference={openReference}
			onexport={exportRun}
			onimport={() => importInput.click()}
			queryStudies={queryRecords}
			onquery={openQueryStudy}
			onnote={(value) => {
				note = value;
				addNote();
			}}
		/>
	{:else if tab === 'methods'}<ResearchMethods />{/if}
	<input
		class="file-input"
		type="file"
		accept=".json,application/json"
		{@attach (element) => {
			importInput = element;
		}}
		onchange={importFile}
		aria-label="Import experiment JSON"
	/>
	<input
		class="file-input"
		type="file"
		accept=".json,application/json"
		{@attach (element) => {
			queryInput = element;
		}}
		onchange={importQueryFile}
		aria-label="Import query study JSON"
	/>
</div>

<style>
	.intervention-setup {
		display: flex;
		align-items: center;
		flex-wrap: wrap;
		gap: 12px;
		padding: 16px;
		border-bottom: 1px solid var(--line);
	}
	.intervention-setup > div {
		flex: 1 1 300px;
	}
	.intervention-setup p {
		margin: 6px 0 0;
		color: var(--muted);
		font-size: 13px;
	}
	.query-operation-status {
		margin: 8px 18px 14px;
		font: 12px var(--mono);
		color: var(--muted);
	}
	.query-history {
		display: flex;
		align-items: center;
		gap: 7px;
		padding: 9px 18px;
		overflow-x: auto;
		border-bottom: 1px solid var(--line);
		background: var(--surface);
	}
	.query-history > span {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		flex-shrink: 0;
		font-size: 12px;
		color: var(--muted);
		margin-right: 5px;
	}
	.query-history button {
		white-space: nowrap;
		font: 12px var(--mono);
		min-height: 27px;
		padding: 4px 9px;
	}
	.query-history button.current {
		border-color: var(--accent);
	}
	.query-history small {
		color: var(--muted);
		margin-left: 6px;
	}
	.query-error {
		display: flex;
		align-items: center;
		gap: 8px;
		margin: 12px 18px;
		padding: 9px 12px;
		border: 1px solid var(--danger);
		border-radius: 5px;
		font-size: 12px;
	}
	.query-error button {
		margin-left: auto;
	}
	.lab-shell {
		min-height: 100dvh;
		background: var(--bg);
	}
	.app-header {
		height: 48px;
		display: flex;
		align-items: center;
		gap: 20px;
		padding: 0 18px;
		border-bottom: 1px solid var(--line);
		background: var(--surface);
	}
	.brand {
		display: flex;
		align-items: center;
		gap: 8px;
		text-decoration: none;
		color: var(--accent);
	}
	.brand > span {
		font-size: 23px;
		font-weight: 650;
		letter-spacing: -1px;
		color: var(--ink);
	}
	.brand-dot {
		color: var(--accent);
	}
	nav {
		display: flex;
		align-self: stretch;
		margin: auto;
		gap: 8px;
	}
	nav button {
		display: flex;
		align-items: center;
		gap: 7px;
		border: 0;
		background: none;
		padding: 0 12px;
		position: relative;
		color: var(--muted);
		font-size: 13px;
	}
	nav button.active {
		color: var(--ink);
	}
	nav button.active:after {
		content: '';
		position: absolute;
		bottom: 0;
		left: 12px;
		right: 12px;
		height: 2px;
		background: var(--accent);
	}
	.count {
		font: 12px var(--mono);
		padding: 2px 4px;
		border: 1px solid var(--line);
		border-radius: 3px;
		color: var(--muted);
	}
	.header-actions {
		display: flex;
		align-items: center;
		gap: 7px;
	}
	.operation-status i {
		width: 5px;
		height: 5px;
		border-radius: 50%;
		display: inline-block;
		background: var(--accent);
	}
	.working {
		animation: pulse 1.3s ease-in-out infinite;
	}
	@keyframes pulse {
		50% {
			opacity: 0.3;
		}
	}
	.command-bar {
		height: 50px;
		border-bottom: 1px solid var(--line);
		display: flex;
		align-items: center;
		padding: 0 17px;
		gap: 14px;
		background: var(--surface);
	}
	.study-title {
		display: flex;
		align-items: center;
		gap: 10px;
		min-width: 225px;
	}
	.study-title h1 {
		font-size: 13px;
		letter-spacing: -0.15px;
		line-height: 1.1;
		font-weight: 550;
		margin: 0 0 4px;
	}
	.study-title > div > span {
		font: 12px var(--mono);
		color: var(--muted);
	}
	.study-icon {
		width: 31px;
		height: 31px;
		display: grid;
		place-items: center;
		background: var(--surface-raised);
		border: 1px solid var(--line);
		border-radius: 6px;
		color: var(--accent);
	}
	.slash {
		color: var(--faint);
		margin: 0 5px;
	}
	.training-controls {
		display: flex;
		align-items: center;
		gap: 8px;
		margin-left: auto;
	}
	.control-label {
		font-size: 12px;
		color: var(--muted);
		margin-right: 2px;
	}
	.segmented {
		display: flex;
		align-items: center;
		gap: 2px;
		border: 1px solid var(--line);
		border-radius: 5px;
		padding: 2px;
		background: var(--bg);
	}
	.segmented button {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		gap: 6px;
		min-height: 26px;
		padding: 4px 9px;
		border: 1px solid transparent;
		background: transparent;
		color: var(--muted);
		font-size: 12px;
		border-radius: 3px;
		white-space: nowrap;
	}
	.segmented button.chosen {
		background: var(--surface-raised);
		color: var(--ink);
		border-color: var(--line);
	}
	.segmented button:hover:not(:disabled) {
		color: var(--ink);
	}
	.segmented.compact button {
		font: 12px var(--mono);
		min-height: 22px;
		padding: 3px 9px;
	}
	.file-actions {
		display: flex;
		border-left: 1px solid var(--line);
		padding-left: 12px;
		gap: 5px;
	}
	.model-sidebar,
	.inspector-sidebar,
	.spatial-panel,
	.bottom-panel {
		background: var(--surface);
		min-width: 0;
		min-height: 0;
	}
	.model-sidebar {
		grid-row: 1/3;
		display: flex;
		flex-direction: column;
		overflow-y: auto;
	}
	.panel-title {
		height: 39px;
		flex-shrink: 0;
		display: flex;
		align-items: center;
		gap: 7px;
		padding: 0 13px;
		border-bottom: 1px solid var(--line);
		color: var(--muted);
	}
	.panel-title h2 {
		font-size: 13px;
		font-weight: 550;
		color: var(--ink);
		margin: 0;
	}
	.small-tag {
		margin-left: auto;
		font: 12px var(--mono);
		color: var(--muted);
		border: 1px solid var(--line);
		border-radius: 3px;
		padding: 2px 4px;
	}
	.architecture {
		padding: 12px 10px 2px;
	}
	.model-meta {
		display: flex;
		justify-content: space-around;
		border-top: 1px solid var(--line);
		border-bottom: 1px solid var(--line);
		padding: 10px 7px;
		margin-top: 8px;
		color: var(--muted);
		font: 12px var(--mono);
	}
	.model-meta b {
		color: var(--ink);
		font-weight: 500;
	}
	.reference-list {
		padding: 14px 10px 10px;
		margin-top: auto;
	}
	.sidebar-label {
		font: 12px var(--mono);
		color: var(--muted);
		display: flex;
		align-items: center;
		gap: 7px;
		margin-bottom: 10px;
		letter-spacing: 0.025em;
	}
	.reference-button {
		display: flex;
		align-items: center;
		gap: 7px;
		text-align: left;
		width: 100%;
		padding: 10px 6px;
		background: var(--surface-raised);
		border: 1px solid var(--line);
		border-radius: 5px;
		margin-top: 6px;
	}
	.reference-button:hover:not(:disabled) {
		border-color: var(--faint);
	}
	.reference-button > span:nth-child(2) {
		min-width: 0;
		flex: 1;
	}
	.reference-button b {
		display: block;
		font-weight: 500;
		font-size: 12px;
		margin-bottom: 4px;
	}
	.reference-button small {
		display: block;
		font: 12px var(--mono);
		color: var(--muted);
		white-space: nowrap;
	}
	.reference-symbol {
		color: var(--muted);
	}
	.new-run {
		display: flex;
		align-items: center;
		gap: 7px;
		padding: 8px 12px;
	}
	.new-run label {
		font-size: 12px;
		color: var(--muted);
		flex: 1;
	}
	.new-run input {
		width: 61px;
		padding: 5px 6px;
		font-size: 12px;
	}
	.sidebar-note {
		padding: 9px 12px 15px;
		color: var(--faint);
		font-size: 12px;
		line-height: 1.6;
		display: flex;
		gap: 7px;
	}
	.spatial-panel {
		grid-column: 2;
		grid-row: 1;
		display: flex;
		flex-direction: column;
		overflow: hidden;
		background: var(--bg);
	}
	.map-toolbar {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 7px 10px;
		border-bottom: 1px solid var(--line);
		height: 43px;
		gap: 6px;
		background: var(--surface);
	}
	.map-tools {
		display: flex;
		align-items: center;
		gap: 6px;
	}
	.map-tools .enabled {
		color: var(--accent);
	}
	.metric-strip {
		display: grid;
		grid-template-columns: 1.35fr 1fr 0.85fr 1fr;
		flex-shrink: 0;
		padding: 11px 17px 8px;
		gap: 18px;
		border-bottom: 1px solid color-mix(in srgb, var(--line) 60%, transparent);
	}
	.metric-strip > div > span {
		display: block;
		color: var(--muted);
		font: 12px var(--mono);
		margin-bottom: 4px;
	}
	.metric-strip strong {
		font: 19px var(--mono);
		letter-spacing: -0.6px;
		color: var(--ink);
	}
	.metric-strip strong small {
		font-size: 12px;
		color: var(--faint);
		letter-spacing: 0;
	}
	.metric-strip > div > span > small {
		font-size: 12px;
		color: var(--faint);
	}
	.field-wrap {
		flex: 1;
		min-height: 220px;
		position: relative;
		overflow: hidden;
	}
	.projection-strip {
		min-height: 27px;
		display: flex;
		align-items: center;
		gap: 14px;
		padding: 5px 12px;
		font: 12px var(--mono);
		color: var(--muted);
		border-top: 1px solid var(--line);
		background: var(--surface);
		flex-wrap: wrap;
	}
	.projection-strip > span:first-child {
		display: flex;
		align-items: center;
		gap: 5px;
	}
	.layer-dot {
		width: 5px;
		height: 5px;
		display: inline-block;
		border-radius: 50%;
		background: var(--layer-1);
	}
	.layer-dot.second {
		background: var(--layer-2);
		margin-left: 5px;
	}
	.quality {
		margin-left: auto;
		display: flex;
		gap: 5px;
		align-items: center;
	}
	.quality b {
		font-weight: 500;
		color: var(--ink);
	}
	.divider {
		height: 9px;
		border-left: 1px solid var(--line);
		margin: 0 5px;
	}
	.prompt-strip {
		display: flex;
		align-items: center;
		gap: 10px;
		min-height: 43px;
		padding: 5px 10px;
		border-top: 1px solid var(--line);
		background: var(--surface);
	}
	.prompt-stepper {
		display: flex;
		align-items: center;
		gap: 0;
		border-right: 1px solid var(--line);
		padding-right: 8px;
		flex-shrink: 0;
	}
	.prompt-stepper > span {
		font: 12px var(--mono);
		white-space: nowrap;
		color: var(--muted);
	}
	.prompt-stepper small {
		font-size: 12px;
		color: var(--faint);
		margin-left: 3px;
	}
	.prompt-stepper .icon-button {
		width: 21px;
		height: 25px;
	}
	.token-stream {
		display: flex;
		align-items: center;
		gap: 3px;
		flex: 1;
	}
	.token-stream button {
		font: 13px var(--mono);
		padding: 4px 5px;
		min-width: 19px;
		height: 25px;
		background: var(--surface-raised);
		border: 1px solid transparent;
		border-radius: 3px;
	}
	.token-stream button.chosen {
		border-color: var(--accent);
		background: var(--acid);
		color: var(--ink);
	}
	.token-stream > :global(.icon) {
		margin: 0 5px;
	}
	.answer-token {
		font: 12px var(--mono);
		color: var(--accent);
		margin-left: 3px;
	}
	.token-position {
		font: 12px var(--mono);
		color: var(--faint);
		white-space: nowrap;
	}
	.timeline {
		height: 36px;
		display: flex;
		align-items: center;
		gap: 9px;
		padding: 0 13px;
		border-top: 1px solid var(--line);
		color: var(--muted);
		background: var(--surface);
		flex-shrink: 0;
	}
	.timeline-label {
		font: 12px var(--mono);
		letter-spacing: 0.04em;
	}
	.checkpoints {
		display: flex;
		align-items: center;
		gap: 3px;
		flex: 1;
		overflow: auto;
	}
	.checkpoints button {
		display: flex;
		align-items: center;
		gap: 5px;
		border: 0;
		background: transparent;
		color: var(--muted);
		font: 12px var(--mono);
		padding: 6px;
		white-space: nowrap;
	}
	.checkpoints i {
		height: 4px;
		width: 4px;
		border-radius: 50%;
		background: var(--faint);
	}
	.checkpoints button.current {
		color: var(--accent);
	}
	.checkpoints button.current i {
		background: var(--accent);
		box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 10%, transparent);
	}
	.live-tag {
		font: 12px var(--mono);
		letter-spacing: 0.08em;
		color: var(--accent);
	}
	.timeline > .text-button {
		font-size: 12px;
	}
	.inspector-sidebar {
		grid-column: 3;
		display: flex;
		flex-direction: column;
		grid-row: 1;
	}
	.unit-picker {
		display: flex;
		gap: 4px;
		align-items: center;
		padding: 8px 12px;
		border-bottom: 1px solid var(--line);
	}
	.unit-picker label {
		font-size: 12px;
		color: var(--muted);
		margin-right: auto;
	}
	.unit-picker input {
		width: 62px;
		font: 12px var(--mono);
		padding: 5px 7px;
		text-align: center;
	}
	.unit-picker .icon-button {
		width: 24px;
		height: 25px;
	}
	.inspector-scroll {
		overflow-y: auto;
		flex: 1;
		min-height: 0;
		padding: 0 12px;
	}
	.neighbors {
		border-top: 1px solid var(--line);
		padding: 12px 0 10px;
	}
	.neighbor-head {
		display: flex;
		justify-content: space-between;
		font: 12px var(--mono);
		color: var(--faint);
		margin-bottom: 6px;
	}
	.neighbors button {
		display: flex;
		align-items: center;
		justify-content: space-between;
		width: 100%;
		padding: 5px 3px;
		background: none;
		border: 0;
		border-bottom: 1px solid color-mix(in srgb, var(--line) 60%, transparent);
		font: 12px var(--mono);
	}
	.neighbors button:hover {
		background: var(--surface-hover);
	}
	.neighbors button > span {
		display: flex;
		align-items: center;
		gap: 6px;
	}
	.neighbors button > span:last-child {
		color: var(--muted);
	}
	.neighbors button i {
		width: 5px;
		height: 5px;
		border-radius: 50%;
		background: var(--layer-1);
	}
	.neighbors button i.second {
		background: var(--layer-2);
	}
	.neighbors p {
		font-size: 12px;
		color: var(--muted);
	}
	.prediction {
		padding: 12px 0;
		border-top: 1px solid var(--line);
	}
	.prediction .sidebar-label {
		justify-content: space-between;
		font-size: 12px;
	}
	.prediction .sidebar-label span {
		font-size: 12px;
		color: var(--faint);
	}
	.probability-bars {
		display: flex;
		gap: 5px;
		height: 68px;
	}
	.probability-bars > div {
		display: flex;
		flex-direction: column;
		align-items: center;
		flex: 1;
	}
	.probability-bars > div > span {
		font: 12px var(--mono);
		color: var(--muted);
		height: 14px;
	}
	.probability-bars small {
		font-size: 12px;
	}
	.prob-track {
		height: 38px;
		width: 100%;
		background: var(--surface-raised);
		position: relative;
		border-radius: 2px;
		overflow: hidden;
	}
	.prob-track > i {
		position: absolute;
		bottom: 0;
		width: 100%;
		background: var(--faint);
	}
	.correct .prob-track > i {
		background: var(--accent);
	}
	.probability-bars b {
		display: flex;
		align-items: center;
		gap: 3px;
		font: 12px var(--mono);
		height: 16px;
	}
	.answer-dot {
		height: 3px;
		width: 3px;
		border-radius: 50%;
		background: var(--accent);
	}
	.prediction p {
		font: 12px var(--mono);
		color: var(--faint);
		margin: 7px 0;
	}
	.intervention-actions {
		padding: 9px 12px 10px;
		border-top: 1px solid var(--line);
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		background: var(--surface);
	}
	.intervention-actions .secondary {
		flex: 1;
		white-space: nowrap;
		font-size: 12px;
		padding: 5px 7px;
		min-height: 28px;
	}
	.repair-action {
		padding: 6px 0 0;
		width: 100%;
		justify-content: space-between;
		font-size: 12px;
	}
	.bottom-panel {
		grid-column: 2/4;
		grid-row: 2;
		display: flex;
		flex-direction: column;
		overflow: hidden;
	}
	.bottom-tabs {
		height: 34px;
		display: flex;
		align-items: stretch;
		padding: 0 12px;
		border-bottom: 1px solid var(--line);
		gap: 17px;
		flex-shrink: 0;
	}
	.bottom-tabs button {
		display: flex;
		align-items: center;
		gap: 6px;
		border: 0;
		border-bottom: 2px solid transparent;
		background: none;
		font-size: 12px;
		color: var(--muted);
		padding: 0 2px;
	}
	.bottom-tabs button.active {
		border-bottom-color: var(--accent);
		color: var(--ink);
	}
	.bottom-tabs > span {
		margin-left: auto;
		align-self: center;
		font: 12px var(--mono);
		color: var(--faint);
	}
	.activation-layout {
		display: grid;
		grid-template-columns: minmax(300px, 1fr) 270px;
		min-height: 0;
		flex: 1;
	}
	.findings-wrap {
		min-height: 0;
		overflow: auto;
	}
	.heatmap-wrap {
		min-width: 0;
		overflow: auto;
		padding: 4px 8px;
	}
	.learning-panel {
		border-left: 1px solid var(--line);
		padding: 13px 13px 5px;
		min-width: 0;
	}
	.learning-panel .sidebar-label {
		font-size: 12px;
		margin-bottom: 12px;
	}
	.learning-panel :global(.curve) {
		margin: 10px 0;
	}
	.curve-key {
		display: flex;
		gap: 10px;
		color: var(--muted);
		font: 12px var(--mono);
		margin-top: 7px;
	}
	.curve-key > span {
		display: flex;
		gap: 4px;
		align-items: center;
	}
	.curve-key i {
		width: 12px;
		border-top: 2px solid var(--accent);
	}
	.curve-key i.dashed {
		border-top: 1px dashed var(--warning);
	}
	.learning-panel > p {
		font: 12px/1.7 var(--mono);
		color: var(--faint);
		margin-top: 14px;
	}
	.repair-results {
		padding: 10px 14px;
		overflow: auto;
		flex: 1;
	}
	.repair-baseline {
		display: flex;
		align-items: center;
		justify-content: space-between;
		font: 12px var(--mono);
		color: var(--muted);
		margin-bottom: 8px;
	}
	.repair-baseline > span:last-child {
		display: flex;
		align-items: center;
		gap: 6px;
	}
	.repair-baseline b {
		color: var(--ink);
		font-weight: 500;
	}
	.repair-body {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 24px;
	}
	table {
		width: 100%;
		border-collapse: collapse;
		font-size: 12px;
	}
	th {
		text-align: left;
		color: var(--faint);
		font: 12px var(--mono);
		padding: 4px 5px 8px;
		border-bottom: 1px solid var(--line);
	}
	td {
		padding: 8px 5px;
		border-bottom: 1px solid var(--line);
	}
	td:not(:first-child) {
		font: 12px var(--mono);
	}
	.repair-note .eyebrow {
		color: var(--warning);
		font-size: 12px;
	}
	.repair-note p {
		font-size: 12px;
		line-height: 1.65;
		color: var(--muted);
		margin: 7px 0;
	}
	.repair-note details {
		font-size: 12px;
		color: var(--muted);
	}
	summary {
		cursor: pointer;
	}
	.empty-repair {
		display: flex;
		align-items: center;
		gap: 18px;
		max-width: 690px;
		margin: 35px auto;
		color: var(--muted);
	}
	.empty-repair h3 {
		font-size: 13px;
		font-weight: 500;
		color: var(--ink);
		margin: 0 0 8px;
	}
	.empty-repair p {
		font-size: 13px;
		line-height: 1.7;
		margin: 0;
	}
	.status-bar {
		height: 26px;
		display: flex;
		align-items: center;
		gap: 14px;
		padding: 0 13px;
		border-top: 1px solid var(--line);
		background: var(--surface);
		font: 12px var(--mono);
		color: var(--muted);
	}
	.operation-status {
		display: flex;
		align-items: center;
		gap: 7px;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.status-right {
		margin-left: auto;
		display: flex;
		align-items: center;
		gap: 14px;
		white-space: nowrap;
	}
	.measurement-progress {
		display: flex;
		align-items: center;
		gap: 7px;
	}
	.measurement-progress progress {
		width: 70px;
		height: 3px;
		accent-color: var(--accent);
	}
	.status-bar .text-button {
		font-size: 12px;
	}
	.file-input {
		display: none;
	}
	.error-banner,
	.storage-banner {
		padding: 9px 15px;
		border-bottom: 1px solid var(--line);
		background: var(--surface-raised);
		color: var(--danger);
		display: flex;
		align-items: center;
		gap: 12px;
		font-size: 13px;
	}
	.error-banner > span {
		flex: 1;
	}
	.storage-banner {
		color: var(--warning);
	}
	.app-header {
		min-height: 68px;
		height: auto;
		padding: 12px 24px;
		gap: 28px;
	}
	nav {
		gap: 4px;
		margin: 0 auto 0 20px;
		align-self: center;
		flex-wrap: wrap;
	}
	nav button {
		min-height: 38px;
		padding: 8px 13px;
		border: 1px solid transparent;
		border-radius: 8px;
		font-size: 14px;
	}
	nav button.active {
		border-color: var(--line);
		background: var(--surface-raised);
	}
	nav button.active::after {
		display: none;
	}
	.command-bar {
		min-height: 92px;
		height: auto;
		background: var(--bg);
		border: 0;
		padding: 20px 24px;
		gap: 16px;
	}
	.study-title h1 {
		font-size: 22px;
	}
	.study-title > div > span {
		font:
			13px/1.5 'DM Sans',
			sans-serif;
	}
	.file-actions {
		margin-left: auto;
	}
	.training-controls {
		padding: 16px;
		flex-wrap: wrap;
	}
	.training-controls .control-label {
		width: 100%;
	}
	.model-sidebar,
	.inspector-sidebar {
		display: block;
		max-height: none;
		overflow: visible;
	}
	.inspector-scroll {
		max-height: none;
		overflow: visible;
	}
	.panel-title {
		height: auto;
		padding: 18px;
	}
	.architecture {
		padding: 16px;
	}
	.reference-list,
	.model-meta,
	.sidebar-note {
		padding: 16px;
	}
	.spatial-panel {
		display: flex;
		min-height: 0;
	}
	.field-wrap {
		flex: none;
		height: clamp(420px, 52vh, 680px);
	}
	.map-toolbar {
		height: auto;
		min-height: 60px;
		padding: 12px 16px;
		flex-wrap: wrap;
	}
	.segmented button {
		padding: 6px 10px;
		min-height: 32px;
	}
	.metric-strip {
		padding: 16px 18px;
		background: var(--surface);
	}
	.metric-strip > div > span {
		font:
			12px/1.5 'DM Sans',
			sans-serif;
	}
	.projection-strip {
		padding: 12px 16px;
		font:
			12px/1.5 'DM Sans',
			sans-serif;
	}
	.prompt-strip {
		flex-wrap: wrap;
		padding: 12px 16px;
		gap: 12px;
	}
	.token-stream {
		flex-wrap: wrap;
	}
	.token-stream button {
		min-width: 28px;
		min-height: 30px;
	}
	.timeline {
		padding: 12px 16px;
	}
	.bottom-panel {
		height: auto;
		overflow: visible;
	}
	.bottom-tabs {
		height: auto;
		min-height: 52px;
		flex-wrap: wrap;
		gap: 8px;
		padding: 8px 16px;
	}
	.bottom-tabs button {
		min-height: 34px;
	}
	.activation-layout {
		min-height: 270px;
		grid-template-columns: minmax(0, 1fr) 340px;
	}
	.status-bar {
		height: auto;
		padding: 12px 24px 20px;
		flex-wrap: wrap;
		gap: 10px;
		border: 0;
	}
	@media (max-width: 1100px) {
		.app-header {
			flex-wrap: wrap;
			gap: 12px;
		}
		nav {
			order: 3;
			width: 100%;
			margin: 0;
		}
		.header-actions {
			margin-left: auto;
		}
	}
	@media (max-width: 700px) {
		.app-header {
			padding: 14px 12px;
		}
		nav {
			overflow: visible;
			gap: 4px;
		}
		nav button {
			padding: 8px 9px;
			font-size: 13px;
		}
		.command-bar {
			padding: 18px 12px;
			min-height: 80px;
		}
		.activation-layout {
			grid-template-columns: minmax(0, 1fr);
		}
		.metric-strip {
			grid-template-columns: 1fr 1fr;
			gap: 12px;
		}
		.study-title h1 {
			font-size: 20px;
		}
	}

	@media (max-width: 700px) {
		.repair-body {
			grid-template-columns: minmax(0, 1fr);
		}
		.repair-results {
			overflow-x: auto;
		}
		.prompt-stepper {
			flex-shrink: 0;
		}
		.inspector-scroll {
			border: 0;
		}
		.status-right {
			flex-wrap: wrap;
		}
		.metric-strip strong {
			font-size: 18px;
		}
		.bottom-tabs > span {
			width: 100%;
			margin-left: 0;
		}
		.activation-layout {
			display: block;
		}
		.learning-panel {
			padding: 16px;
		}
	}
	.command-bar {
		flex-wrap: wrap;
		min-width: 0;
		position: sticky;
		top: 0;
		z-index: 20;
		background: var(--bg);
		padding: 16px 24px;
		gap: 16px;
	}
	.model-sidebar > .file-actions {
		padding: 12px 16px;
		justify-content: flex-end;
		border-bottom: 1px solid var(--line);
	}
	.action-setup {
		padding: 18px;
		border-bottom: 1px solid var(--line);
		display: flex;
		gap: 16px;
		align-items: center;
		flex-wrap: wrap;
	}
	.action-setup > div {
		flex: 1;
		min-width: 240px;
	}
	.action-setup p {
		color: var(--muted);
		line-height: 1.6;
		margin: 8px 0 0;
		font-size: 14px;
	}
</style>
