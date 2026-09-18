<script lang="ts">
	import type { Snippet } from 'svelte';
	import ResearchWorkspace from './ResearchWorkspace.svelte';
	import { onMount, tick } from 'svelte';
	import { publicAsset } from '$lib/deployment/public-assets';
	import { LiveTokenStoryEngine } from '$lib/token-stories/live-engine';
	import type { LiveTokenStoryFrame } from '$lib/token-stories/live-protocol';
	import { LayerPlaybackClock } from './layer-playback-clock';
	import TokenStoryLivePanel from './TokenStoryLivePanel.svelte';
	import { TokenStoryGeometryEngine } from '$lib/token-stories/geometry-engine';
	import type { TokenStoryNeighbor } from '$lib/token-stories/geometry';
	import {
		TOKEN_STORY_PRESETS,
		tokenStoryParameterCount,
		tokenStoryUnitCount,
		type TokenStoryBackend,
		type TokenStoryPreset,
		type TokenStoryProbe,
		type TokenStoryGeneration,
		type TokenStoryMetrics,
		type TokenStoryEvent
	} from '$lib/token-stories/protocol';
	import {
		listTokenStoryRuns,
		loadTokenStoryRun,
		saveTokenStoryRun,
		importTokenStoryRun,
		exportTokenStoryRun,
		loadTokenStoryReference,
		getTokenStoryArchiveWarnings,
		type TokenStoryRunRecord,
		type TokenStoryRunSummary,
		type TokenStoryReference
	} from '$lib/token-stories/live-archive';
	import Icon from './Icon.svelte';
	import NeuralField from './NeuralField.svelte';
	import TokenStoryMetricsPanel from './StoryMetrics.svelte';
	import TokenStoryProbePanel from './TokenStoryProbe.svelte';
	import TokenStoryTokenization from './TokenStoryTokenization.svelte';
	import {
		createTokenStoryTokenizer,
		validateTokenStoryTokenizerIdentity,
		type TokenStoryTokenizer,
		type TokenStoryTokenizerData,
		type TokenStoryCorpusMetadata
	} from '$lib/token-stories/tokenizer';

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
	let toolsCollapsed = $state(true);
	let interventionOpen = $state(false);
	let trainingOpen = $state(false);
	let commandHeight = $state(88);
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
		| 'replaying'
		| 'loading'
		| 'saving'
		| 'error';
	let phase = $state<Phase>('idle');
	let status = $state('Choose a model size, then initialize a new run or open a saved specimen.');
	let error = $state('');
	let storageWarning = $state('');
	let preset = $state<TokenStoryPreset>('small');
	let seed = $state(42);
	let tokenizer = $state.raw<TokenStoryTokenizer | null>(null);
	let corpusMetadata = $state.raw<TokenStoryCorpusMetadata | null>(null);
	let backend = $state<TokenStoryBackend | 'auto'>('auto');
	let budget = $state<number | null>(null);
	const mapCadence = 100;
	let record = $state.raw<TokenStoryRunRecord | null>(null);
	let runs = $state.raw<TokenStoryRunSummary[]>([]);
	let references = $state.raw<TokenStoryReference[]>([]);
	let runtimeReady = $state(false);
	let residentModelId = $state<string | null>(null);
	let savedCheckpointStep = $state<number | null>(null);
	let snapshotIndex = $state<number | null>(null);
	let selected = $state<number | null>(null);
	let layerFilter = $state<number | null>(null);
	let viewMode = $state<'functional' | 'architecture'>('functional');
	let activationEncoding = $state<'size' | 'brightness'>('brightness');
	let token = $state(0);
	let probePrompt = $state('Once upon a time, there was a little');
	let probeOpen = $state(false);
	let probe = $state.raw<TokenStoryProbe | null>(null);
	let lesioned = $state.raw<TokenStoryProbe | null>(null);
	let nearest = $state.raw<TokenStoryNeighbor[]>([]);
	let samplePrompt = $state('Once upon a time, there was a little girl named Lily.');
	let samplingSeed = $state(7);
	let temperature = $state(0.8);
	let sampleLength = $state(128);
	let topK = $state(40);
	let sample = $state.raw<TokenStoryGeneration | null>(null);
	let liveGeneration = $state(false);
	let playbackFrame = $state.raw<LiveTokenStoryFrame | null>(null);
	let playbackFrames = $state.raw<LiveTokenStoryFrame[]>([]);
	let playbackLayer = $state<number | null>(null);
	let playbackPaused = $state(false);
	let playbackPace = $state(160);
	let playbackTokens = $state.raw<number[]>([]);
	let playbackPrompt = $state('');
	let playbackComplete = $state(false);
	const playbackClock = new LayerPlaybackClock();
	let trainingTarget = $state<number | null>(null);
	let stopRequested = $state(false);
	let measurementProgress = $state({ completed: 0, total: 0 });
	let footerTab = $state<'samples' | 'history'>('samples');
	let importInput: HTMLInputElement | undefined;
	let engine: LiveTokenStoryEngine | null = null;
	let geometryEngine: TokenStoryGeometryEngine | null = null;
	let mounted = false;
	let lifecycle = 0;
	let neighborRevision = 0;
	let trainCallActive = false;

	const presetIds: TokenStoryPreset[] = ['small', 'medium', 'large'];
	const promptSeeds = [
		{ name: 'Story opening', text: 'Once upon a time, there was a little' },
		{
			name: 'Object in a box',
			text: 'Lily put the red ball in the box. Then she opened the box and saw'
		},
		{ name: 'Who is speaking?', text: 'Tom gave Anna a toy. She smiled and said' },
		{ name: 'A negative statement', text: 'The box was not empty. Inside the box there was' }
	];
	let playing = $derived((phase === 'generating' && liveGeneration) || phase === 'replaying');
	let busy = $derived(!['idle', 'ready', 'archived', 'error'].includes(phase));
	let blocked = $derived(busy || disabled);
	let config = $derived(TOKEN_STORY_PRESETS[record?.presetId ?? preset]);
	let unitCount = $derived(tokenStoryUnitCount(config));
	let latestMetric = $derived(record?.metrics.at(-1));
	let snapshot = $derived(
		record ? record.snapshots[snapshotIndex ?? record.snapshots.length - 1] : undefined
	);
	let activityFrame = $derived(
		playbackFrame &&
			(viewMode === 'architecture' ||
				(snapshot?.atlas.modelId === playbackFrame.modelId &&
					snapshot.atlas.step === playbackFrame.step))
			? playbackFrame
			: null
	);
	let focusedLayerHidden = $derived(
		!!activityFrame &&
			layerFilter !== null &&
			playbackLayer !== null &&
			layerFilter !== playbackLayer
	);
	let historical = $derived(snapshotIndex !== null && snapshot?.atlas.step !== latestMetric?.step);
	let matchingProbe = $derived(
		probe &&
			snapshot &&
			probe.prompt.original === probePrompt &&
			probe.step === snapshot.atlas.step &&
			probe.modelId === snapshot.atlas.modelId
			? probe
			: null
	);
	let liveProbe = $derived(
		!historical &&
			runtimeReady &&
			probe?.prompt.original === probePrompt &&
			probe?.modelId === residentModelId &&
			probe?.step === latestMetric?.step
			? probe
			: null
	);
	let ready = $derived(runtimeReady && !blocked && !historical);
	let trainedReference = $derived(
		references.find((reference) => reference.title === 'TinyStories BPE small · seed 42')
	);
	let canLoadModel = $derived(!runtimeReady && !!(record?.checkpoint || trainedReference));
	let generationAvailability = $derived(
		blocked
			? status
			: historical
				? 'You are inspecting an earlier checkpoint. Select Latest in Measurement details to generate.'
				: runtimeReady
					? 'Enter a prefix, then generate. Activations follow each token through the layers; the same button pauses playback.'
					: record?.checkpoint
						? `Load the saved weights at step ${record.checkpoint.step}, then generate with live activations.`
						: 'Load the trained 1.85M-parameter example (27.5 MB), then generate with live activations. Recorded replays contain measurements only.'
	);
	let interventionHint = $derived(
		playing
			? 'Stop generation or replay before running a separate intervention.'
			: blocked
				? status
				: !runtimeReady
					? 'Saved activations cannot run interventions. Load model weights first.'
					: historical
						? 'Return to the latest checkpoint before measuring an intervention.'
						: activityFrame
							? 'This is a recorded token frame. Measure a prompt to compare intact and silenced predictions.'
							: !liveProbe
								? 'Run a valid prompt at this checkpoint before silencing a unit.'
								: selected === null
									? 'Select a point in the network or enter a unit ID first.'
									: ''
	);
	let points = $derived.by(() => {
		if (!snapshot && !(activityFrame && viewMode === 'architecture')) return [];
		const atlas = snapshot?.atlas;
		const geometry = snapshot?.geometry;
		const columns = 32;
		const rows = Math.ceil(config.hidden / columns);
		return Array.from({ length: activityFrame?.unitCount ?? atlas!.unitCount }, (_, id) => {
			if (viewMode === 'functional' && !geometry?.valid[id]) return null;
			const layer = Math.floor(id / config.hidden);
			const channel = id % config.hidden;
			const position: [number, number, number] =
				viewMode === 'functional'
					? geometry!.positions[id]
					: [
							(layer / Math.max(1, config.layers - 1) - 0.5) * 2.8,
							(0.5 - Math.floor(channel / columns) / Math.max(1, rows - 1)) * 1.35,
							((channel % columns) / (columns - 1) - 0.5) * 1.1
						];
			const activation = activityFrame
				? activityFrame.activations[id]
				: matchingProbe
					? matchingProbe.activations[
							Math.min(token, matchingProbe.prompt.tokenIds.length - 1) * unitCount + id
						]
					: atlas!.fingerprints[id * atlas!.dimensions];
			return { id, position, layer, channel, activation };
		}).filter((point) => point !== null);
	});
	let edges = $derived<[number, number][]>(
		viewMode === 'functional' && selected !== null
			? nearest.map((neighbor) => [selected!, neighbor.index])
			: []
	);
	let corpus = $derived(
		corpusMetadata && (!record || record.corpusId === corpusMetadata.id) ? corpusMetadata : null
	);
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
	function appendMetric(metric: TokenStoryMetrics): void {
		if (!record) return;
		if (probe && probe.step !== metric.step) invalidateProbe();
		record = {
			...record,
			metrics: [...record.metrics.filter((entry) => entry.step !== metric.step), metric].sort(
				(a, b) => a.step - b.step
			)
		};
	}
	function handleEvent(event: TokenStoryEvent, ticket: number): void {
		if (!current(ticket)) return;
		if (event.type === 'status') status = event.message;
		else if (event.type === 'metrics') appendMetric(event.metrics);
		else measurementProgress = { completed: event.completed, total: event.total };
	}
	async function refreshRuns(): Promise<void> {
		const next = await listTokenStoryRuns();
		if (mounted) {
			runs = next;
			const warnings = getTokenStoryArchiveWarnings();
			if (warnings.length) storageWarning = warnings.join(' ');
		}
	}
	async function persist(): Promise<boolean> {
		if (!record) return false;
		const next = { ...record, updatedAt: new Date().toISOString() };
		record = next;
		try {
			await saveTokenStoryRun(next);
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
		playbackClock.stop();
		clearPlaybackView();
		const previous = engine;
		engine = null;
		runtimeReady = false;
		residentModelId = null;
		invalidateProbe();
		if (previous) await previous.dispose();
	}
	function getGeometry(): TokenStoryGeometryEngine {
		return (geometryEngine ??= new TokenStoryGeometryEngine());
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
		if (
			next === 'functional' &&
			playbackFrame &&
			(snapshot?.atlas.modelId !== playbackFrame.modelId ||
				snapshot.atlas.step !== playbackFrame.step)
		) {
			error = `No functional map is open for trace step ${playbackFrame.step}. Return to the probe or open a matching recorded map before changing layout.`;
			return;
		}
		viewMode = next;
		void updateNeighbors();
	}
	function invalidateProbe(): void {
		probe = null;
		lesioned = null;
	}
	async function probeCurrent(ticket: number): Promise<void> {
		invalidateProbe();
		if (!engine) return;
		const next = await engine.probe(probePrompt);
		if (!current(ticket)) return;
		if (
			next.modelId !== residentModelId ||
			next.step !== latestMetric?.step ||
			next.tokenizerId !== record?.tokenizer.id ||
			next.corpusId !== record?.corpusId
		)
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
			const owner = new LiveTokenStoryEngine((event) => handleEvent(event, ticket));
			engine = owner;
			status = `Initializing ${(tokenStoryParameterCount(TOKEN_STORY_PRESETS[preset]) / 1e6).toFixed(3)}M parameters…`;
			const init = await owner.initialize(preset, seed, backend);
			if (!current(ticket)) return;
			tokenizer = createTokenStoryTokenizer(init.tokenizer);
			corpusMetadata = init.corpus;
			const now = new Date().toISOString();
			record = {
				version: 1,
				kind: 'tissue-token-story-run',
				tokenizer: init.tokenizer,
				corpusId: init.corpus.id,
				id: crypto.randomUUID(),
				createdAt: now,
				updatedAt: now,
				title: `TinyStories subword ${(init.parameterCount / 1e6).toFixed(3)}M · seed ${seed}`,
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
			chooseSample(null);
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
		clearPlaybackView();
		trainingTarget = budget === null ? null : (latestMetric?.step ?? 0) + budget;
		setPhase('training');
		try {
			while (
				!stopRequested &&
				(trainingTarget === null || (latestMetric?.step ?? 0) < trainingTarget)
			) {
				status =
					trainingTarget === null
						? `Continuous training · step ${latestMetric?.step ?? 0}; map and checkpoint every ${mapCadence} updates…`
						: `Training toward step ${trainingTarget}…`;
				measurementProgress = { completed: 0, total: 0 };
				trainCallActive = true;
				const next = await engine
					.train(
						trainingTarget === null ? 25 : Math.min(25, trainingTarget - (latestMetric?.step ?? 0))
					)
					.finally(() => {
						trainCallActive = false;
					});
				if (!current(ticket)) return;
				appendMetric(next);
				const lastMapStep = record.snapshots.at(-1)?.atlas.step ?? -mapCadence;
				if (
					next.step - lastMapStep >= mapCadence ||
					stopRequested ||
					(trainingTarget !== null && next.step >= trainingTarget)
				)
					await capture(ticket, true);
				else await persist();
				if (!current(ticket)) return;
			}

			if (record.snapshots.at(-1)?.atlas.step !== latestMetric?.step) await capture(ticket, true);
			if (!current(ticket)) return;
			observe(
				stopRequested ? 'Training paused' : 'Training interval completed',
				`At step ${latestMetric?.step}. Worker bursts are at most 25 updates. Calibration maps and checkpoints are saved every 100 updates and at the final paused step.`
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
		if (liveGeneration || phase === 'replaying') {
			await stopPlayback();
			return;
		}
		stopRequested = true;
		status =
			phase === 'generating'
				? 'Stopping after the current sampled token…'
				: 'Pausing after the current worker operation, then recording evidence…';
		try {
			if (trainCallActive || phase === 'generating') await engine?.pause();
		} catch (reason) {
			if (mounted) error = String(reason);
		}
	}
	async function inspectPrompt(): Promise<void> {
		if (!ready) return;
		clearPlaybackView();
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
				result.tokenizerId !== intact.tokenizerId ||
				result.corpusId !== intact.corpusId ||
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
							prompt: intact.prompt.original,
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
	function clearPlaybackView(): void {
		playbackClock.stop();
		playbackFrame = null;
		playbackLayer = null;
		playbackPaused = false;
	}
	function returnToProbe(): void {
		clearPlaybackView();
		probeOpen = true;
		interventionOpen = false;
		toolPanel = 'inspector';
		toolsCollapsed = false;
	}
	function chooseSample(next: TokenStoryGeneration | null): void {
		clearPlaybackView();
		sample = next;
		const index = next ? record?.samples?.indexOf(next) : -1;
		playbackFrames =
			record?.liveTraces?.find((trace) => trace.generationIndex === index)?.frames ?? [];
		playbackTokens = next?.tokenIds ?? [];
		playbackPrompt = next?.prompt.text ?? '';
		playbackComplete = !!next;
	}
	function togglePlaybackPause(): void {
		playbackPaused = !playbackPaused;
		if (playbackPaused) playbackClock.pause();
		else playbackClock.play();
	}
	function nextPlaybackLayer(): void {
		if (playing) {
			playbackPaused = true;
			playbackClock.nextLayer();
		} else if (playbackFrame)
			playbackLayer = Math.min(config.layers - 1, (playbackLayer ?? -1) + 1);
	}
	async function nextPlaybackToken(): Promise<void> {
		if (playing) {
			playbackPaused = true;
			playbackClock.nextToken();
		} else
			await inspectPlaybackFrame(
				Math.min(playbackFrames.length - 1, (playbackFrame?.index ?? -1) + 1)
			);
	}
	function setPlaybackPace(value: number): void {
		playbackPace = value;
		playbackClock.setPace(value);
	}
	async function stopPlayback(): Promise<void> {
		stopRequested = true;
		status = liveGeneration
			? 'Stopping live generation; preserving acknowledged tokens…'
			: 'Stopping recorded playback…';
		try {
			// The worker must discard its provisional sample before a UI gate is released.
			if (liveGeneration) await engine?.pause();
		} catch (reason) {
			if (mounted) error = reason instanceof Error ? reason.message : String(reason);
		} finally {
			playbackClock.stop();
		}
	}
	async function presentFrame(frame: LiveTokenStoryFrame, ticket: number): Promise<boolean> {
		if (!current(ticket) || stopRequested || !(await playbackClock.waitForFrame())) return false;
		if (!current(ticket) || stopRequested) return false;
		playbackFrame = frame;
		for (let layer = 0; layer < frame.config.layers; layer++) {
			if (!current(ticket) || stopRequested) return false;
			playbackLayer = layer;
			await tick();
			if (!(await playbackClock.waitLayer()) || !current(ticket) || stopRequested) return false;
		}
		playbackClock.finishToken();
		return true;
	}
	async function prepareTraceView(frame: LiveTokenStoryFrame): Promise<boolean> {
		probeOpen = false;
		const index =
			record?.snapshots.findIndex(
				(entry) => entry.atlas.step === frame.step && entry.atlas.modelId === frame.modelId
			) ?? -1;
		if (viewMode === 'functional' && index < 0) {
			error = `No functional map was recorded at step ${frame.step}. Choose Model layout to inspect this trace without using coordinates from another checkpoint.`;
			return false;
		}
		if (index >= 0 && snapshot !== record?.snapshots[index]) await selectSnapshot(index);
		return true;
	}
	async function inspectPlaybackFrame(index: number): Promise<void> {
		if (blocked || !playbackFrames[index]) return;
		const frame = playbackFrames[index];
		if (!(await prepareTraceView(frame))) return;
		if (!mounted) return;
		playbackFrame = frame;
		playbackLayer = 0;
		playbackTokens = sample?.tokenIds.slice(0, frame.index + 1) ?? [];
		status = `Recorded next-token frame ${frame.index + 1} · checkpoint ${frame.step}`;
	}
	async function replayTrace(): Promise<void> {
		if (blocked || !playbackFrames.length) return;
		const frames = playbackFrames;
		if (!(await prepareTraceView(frames[0]))) return;
		if (!mounted) return;
		const ticket = lifecycle;
		setPhase('replaying');
		stopRequested = false;
		playbackPaused = false;
		playbackTokens = [];
		playbackComplete = false;
		playbackClock.start();
		playbackClock.setPace(playbackPace);
		try {
			for (const frame of frames) {
				if (!(await presentFrame(frame, ticket))) break;
				playbackTokens = [...playbackTokens, frame.sampledToken];
			}
		} finally {
			playbackClock.stop();
			if (current(ticket)) {
				playbackComplete = true;
				playbackPaused = false;
				status = 'Recorded trace playback · no model inference performed';
				setPhase(runtimeReady ? 'ready' : 'archived');
			}
		}
	}
	async function generateLive(): Promise<void> {
		if (!ready || !engine || !record) return;
		if (
			viewMode === 'functional' &&
			(snapshot?.atlas.modelId !== residentModelId || snapshot.atlas.step !== latestMetric?.step)
		) {
			error =
				'The functional map does not match the current checkpoint. Choose Model layout to generate without using coordinates from another checkpoint.';
			return;
		}
		try {
			playbackPrompt =
				tokenizer?.decode(tokenizer.encode(samplePrompt, { bos: true }).slice(-config.context)) ??
				samplePrompt;
		} catch (reason) {
			error = reason instanceof Error ? reason.message : String(reason);
			return;
		}
		const ticket = lifecycle;
		error = '';
		probeOpen = false;
		liveGeneration = true;
		setPhase('generating');
		stopRequested = false;
		playbackPaused = false;
		playbackComplete = false;
		playbackFrame = null;
		playbackFrames = [];
		playbackTokens = [];
		playbackClock.start();
		playbackClock.setPace(playbackPace);
		status = 'Live generation · measured layer playback';
		const frames: LiveTokenStoryFrame[] = [];
		try {
			const result = await engine.generateLive(
				samplePrompt,
				{ maxTokens: sampleLength, temperature, seed: samplingSeed, topK },
				async (frame) => {
					if (!current(ticket) || stopRequested) return;
					if (
						frame.modelId !== residentModelId ||
						frame.step !== latestMetric?.step ||
						frame.tokenizerId !== record?.tokenizer.id ||
						frame.corpusId !== record?.corpusId
					)
						throw new Error('Live frame does not match the resident checkpoint and tokenizer.');
					if (!(await presentFrame(frame, ticket))) return;
					frames.push(frame);
					playbackFrames = [...frames];
					playbackTokens = [...playbackTokens, frame.sampledToken];
				}
			);
			if (!current(ticket) || !record) return;
			const committed = frames.slice(0, result.tokenIds.length);
			if (
				committed.length !== result.tokenIds.length ||
				committed.some(
					(frame, index) => frame.index !== index || frame.sampledToken !== result.tokenIds[index]
				)
			)
				throw new Error('Live trace is incomplete; refusing to record mismatched token evidence.');
			const generationIndex = record.samples?.length ?? 0;
			record = {
				...record,
				samples: [...(record.samples ?? []), result],
				liveTraces: [...(record.liveTraces ?? []), { generationIndex, frames: committed }]
			};
			sample = result;
			playbackFrames = committed;
			playbackTokens = result.tokenIds;
			playbackFrame = committed.at(-1) ?? null;
			playbackLayer = committed.length ? config.layers - 1 : null;
			playbackComplete = true;
			observe(
				'Live token trace recorded',
				`${result.tokenIds.length} acknowledged tokens at step ${result.step}; all measured final-input MLP activations and raw next-token probabilities retained${result.cancelled ? '; stopped early' : ''}. Layer pacing is presentation timing.`
			);
			await persist();
			if (current(ticket))
				status = `${result.cancelled ? 'Stopped' : 'Completed'} live generation · ${result.tokenIds.length} tokens and frames recorded`;
		} catch (reason) {
			if (current(ticket)) error = reason instanceof Error ? reason.message : String(reason);
		} finally {
			playbackClock.stop();
			if (current(ticket)) {
				liveGeneration = false;
				playbackPaused = false;
				setPhase('ready');
			}
		}
	}
	async function generate(): Promise<void> {
		if (!ready || !engine || !record) return;
		clearPlaybackView();
		const ticket = lifecycle;
		error = '';
		setPhase('generating');
		stopRequested = false;
		status = 'Sampling tokens from the current model…';
		try {
			const result = await engine.generate(samplePrompt, {
				maxTokens: sampleLength,
				temperature,
				seed: samplingSeed,
				topK
			});
			if (!current(ticket) || !record) return;
			record = { ...record, samples: [...(record.samples ?? []), result] };
			chooseSample(result);
			observe(
				'Text sampled',
				`Step ${result.step}; sampling seed ${result.samplingSeed}; temperature ${result.temperature}; top-k ${result.topK}; ${result.tokenIds.length} generated tokens${result.cancelled ? '; cancelled' : ''}.`
			);
			await persist();
			if (current(ticket)) status = `Sample recorded at step ${result.step}`;
		} catch (reason) {
			if (current(ticket)) error = reason instanceof Error ? reason.message : String(reason);
		} finally {
			if (current(ticket)) setPhase('ready');
		}
	}
	async function openRecord(next: TokenStoryRunRecord): Promise<void> {
		const ticket = ++lifecycle;
		setPhase('loading');
		error = '';
		status = 'Opening recorded measurements; model remains unloaded…';
		try {
			await disposeRuntime();
			if (!current(ticket)) return;
			record = next;
			tokenizer = createTokenStoryTokenizer(next.tokenizer);
			preset = next.presetId;
			if (preset !== 'small' && backend === 'wasm') backend = 'auto';
			seed = next.seed;
			snapshotIndex = null;
			selected = null;
			layerFilter = null;
			nearest = [];
			probe = null;
			lesioned = null;
			chooseSample(next.samples?.at(-1) ?? null);
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
			const next = await loadTokenStoryRun(id);
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
	async function openReference(reference: TokenStoryReference): Promise<void> {
		if (blocked) return;
		setPhase('loading');
		status = 'Loading and verifying recorded specimen…';
		try {
			const original = await loadTokenStoryReference(reference);
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
			await saveTokenStoryRun(next);
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
			const original = await importTokenStoryRun(file);
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
			await saveTokenStoryRun(next);
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
					liveTraces: source.liveTraces?.flatMap((trace) => {
						const original = source.samples?.[trace.generationIndex];
						if (!original || original.step > checkpoint.step) return [];
						const generationIndex = source
							.samples!.slice(0, trace.generationIndex)
							.filter((entry) => entry.step <= checkpoint.step).length;
						return [{ generationIndex, frames: trace.frames }];
					}),
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
			const owner = new LiveTokenStoryEngine((event) => handleEvent(event, ticket));
			engine = owner;
			const restored = await owner.loadCheckpoint(checkpoint, backend);
			if (!current(ticket)) return;
			tokenizer = createTokenStoryTokenizer(restored.tokenizer);
			corpusMetadata = restored.corpus;
			const metric = restored.metrics;
			appendMetric(metric);
			residentModelId = restored.modelId;
			runtimeReady = true;
			snapshotIndex = null;
			chooseSample(record?.samples?.at(-1) ?? null);
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
	async function loadForInference(): Promise<void> {
		if (blocked || runtimeReady) return;
		if (!record?.checkpoint) {
			if (!trainedReference) return;
			await openReference(trainedReference);
		}
		if (record?.checkpoint && (phase === 'archived' || phase === 'error')) await resume();
	}
	function openTools(kind: 'model' | 'probe' | 'intervention' | 'train') {
		toolPanel = kind === 'model' || kind === 'train' ? 'model' : 'inspector';
		interventionOpen = kind === 'intervention';
		probeOpen = kind === 'probe';
		if (kind === 'train') {
			trainingOpen = true;
			setupOpen = !runtimeReady && !record?.checkpoint;
		}
		toolsCollapsed = false;
	}
	async function trainFromBar() {
		if (phase === 'training') {
			pause();
			return;
		}
		if (ready) {
			await train();
			return;
		}
		openTools('train');
	}
	async function startReplay() {
		if (blocked) return;
		if (!playbackFrames.length) {
			if (record) return;
			const reference = references.find((item) => item.title.startsWith('Live replay'));
			if (!reference) return;
			await openReference(reference);
		}
		if (!blocked && playbackFrames.length) await replayTrace();
	}
	let canGenerate = $derived(
		(ready || (canLoadModel && !blocked)) &&
			Number.isInteger(samplingSeed) &&
			samplingSeed >= 0 &&
			samplingSeed <= 0xffffffff &&
			Number.isFinite(temperature) &&
			temperature >= 0.1 &&
			temperature <= 2 &&
			Number.isInteger(topK) &&
			topK >= 1 &&
			topK <= 256
	);
	let generationLabel = $derived(
		playing
			? `${playbackPaused ? 'Resume' : 'Pause'} ${phase === 'replaying' ? 'replay' : 'generation'}`
			: runtimeReady
				? 'Generate live'
				: blocked
					? 'Loading model…'
					: 'Load & generate live'
	);
	async function startLive(): Promise<void> {
		if (blocked) return;
		if (!runtimeReady) await loadForInference();
		if (ready) await generateLive();
	}
	async function prepareIntervention(): Promise<void> {
		if (blocked) return;
		if (!runtimeReady) {
			await loadForInference();
			return;
		}
		if (historical) await selectSnapshot(null);
		const keepIntervention = interventionOpen;
		returnToProbe();
		interventionOpen = keepIntervention;
		if (keepIntervention) probeOpen = false;
		await inspectPrompt();
	}
	async function selectSnapshot(index: number | null): Promise<void> {
		if (blocked || !record) return;
		clearPlaybackView();
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
			exportTokenStoryRun(record);
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
	function archivedEffects(entry: NonNullable<TokenStoryRunRecord['interventions']>[number]) {
		return Array.from(entry.probabilities, (probability, id) => ({
			id,
			probability,
			lesioned: entry.lesionedProbabilities[id],
			delta: entry.lesionedProbabilities[id] - probability
		}))
			.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))
			.slice(0, 8);
	}
	function tokenPiece(id: number): string {
		return tokenizer?.tokenPiece(id).replaceAll(' ', '·').replaceAll('\n', '↵') ?? String(id);
	}
	async function loadTokenizerPreview(): Promise<void> {
		try {
			const responses = await Promise.all([
				fetch(publicAsset('/data/tinystories-bpe/tokenizer.json')),
				fetch(publicAsset('/data/tinystories-bpe/corpus.json'))
			]);
			if (!responses.every((response) => response.ok))
				throw new Error('Tokenizer assets could not be loaded.');
			const data = (await responses[0].json()) as TokenStoryTokenizerData;
			const metadata = (await responses[1].json()) as TokenStoryCorpusMetadata;
			await validateTokenStoryTokenizerIdentity(data);
			if (!mounted || record) return;
			tokenizer = createTokenStoryTokenizer(data);
			corpusMetadata = metadata;
		} catch (reason) {
			if (mounted)
				error = `Tokenization preview unavailable: ${reason instanceof Error ? reason.message : String(reason)}`;
		}
	}
	onMount(() => {
		mounted = true;
		void loadTokenizerPreview();
		void refreshRuns().catch((reason) => {
			if (mounted) storageWarning = String(reason);
		});
		void fetch(publicAsset('/experiments/token-stories-index.json'))
			.then(async (response) => {
				if (!response.ok) return;
				const data = (await response.json()) as {
					version?: number;
					references?: TokenStoryReference[];
				};
				if (mounted && data.version === 1 && Array.isArray(data.references))
					references = data.references;
			})
			.catch(() => {});
		return () => {
			mounted = false;
			playbackClock.stop();
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
	aria-label="Import subword specimen"
/>
<div class="token-story-lab" data-active={active}>
	<header class="token-story-toolbar" bind:clientHeight={commandHeight}>
		<div class="lab-title">
			<Icon name="book" size={21} />
			<div>
				<h1>TinyStories</h1>
				<span
					>{record
						? `${(tokenStoryParameterCount(config) / 1e6).toFixed(2)}M parameters · ${config.layers} layers · measured step ${snapshot?.atlas.step ?? '—'}`
						: 'Subword transformer · measured MLP activations'}</span
				>
			</div>
			<span class="study-label">Study 004</span>
		</div>

		<nav class="lab-actions" aria-label="TinyStories actions">
			<button
				class="primary generate-action"
				onclick={playing ? togglePlaybackPause : startLive}
				disabled={playing ? stopRequested : !canGenerate}
				title={generationAvailability}
			>
				<Icon name={playing && !playbackPaused ? 'pause' : 'play'} size={17} />{generationLabel}
			</button>
			<button
				onclick={startReplay}
				disabled={blocked ||
					(!playbackFrames.length &&
						(!!record || !references.some((item) => item.title.startsWith('Live replay'))))}
				title={record && !playbackFrames.length
					? 'This specimen has no recorded trace. Generate live to record one.'
					: 'Play measured activations without loading model weights'}
				><Icon name="reset" size={17} />{record ? 'Replay trace' : 'Replay example'}</button
			>
			<button
				onclick={trainFromBar}
				disabled={phase === 'training' ? stopRequested : blocked}
				title={ready
					? 'Start training with the configured update budget'
					: 'Set up or restore a model to train'}
				><Icon name={phase === 'training' ? 'pause' : 'activity'} size={17} />{phase === 'training'
					? 'Pause training'
					: 'Train'}</button
			>
			<button
				aria-pressed={!toolsCollapsed && toolPanel === 'inspector' && !interventionOpen}
				onclick={() => openTools('probe')}><Icon name="search" size={17} />Probe</button
			>
			<button
				aria-pressed={!toolsCollapsed && toolPanel === 'inspector' && interventionOpen}
				onclick={() => openTools('intervention')}><Icon name="target" size={17} />Intervene</button
			>
			<button
				aria-pressed={!toolsCollapsed && toolPanel === 'model'}
				onclick={() => openTools('model')}><Icon name="settings" size={17} />Model</button
			>
		</nav>
	</header>
	{#if error || storageWarning}<div class="notice" role="alert">
			<Icon name="info" size={14} /><span
				>{error}{error && storageWarning ? ' · ' : ''}{storageWarning}</span
			><button
				class="icon-button"
				aria-label="Dismiss subword message"
				onclick={() => {
					error = '';
					storageWarning = '';
				}}><Icon name="close" size={13} /></button
			>
		</div>{/if}
	{#if disabled}<div class="external-busy">
			Another lab worker is active. Model operations here are paused until it finishes.
		</div>{/if}
	<ResearchWorkspace
		name="token-story"
		topOffset={commandHeight + 16}
		externalControls
		toolTitle={toolPanel === 'model'
			? 'Model & runs'
			: interventionOpen
				? 'Intervention'
				: 'Prompt & unit inspection'}
		bind:panel={toolPanel}
		bind:collapsed={toolsCollapsed}
	>
		{#snippet model()}
			<aside class="setup-panel">
				<div class="model-utilities">
					{@render representation?.()}
					<div class="file-controls">
						<button
							class="icon-button"
							onclick={saveNow}
							disabled={!record || blocked}
							aria-label="Save subword checkpoint"
							title="Save latest evidence and checkpoint"><Icon name="check" size={15} /></button
						><button
							class="icon-button"
							onclick={exportNow}
							disabled={!record || busy}
							aria-label="Export subword run"
							title="Export binary .tissue specimen"><Icon name="download" size={15} /></button
						><button
							class="icon-button"
							onclick={() => importInput?.click()}
							disabled={blocked}
							aria-label="Import subword run"
							title="Import binary .tissue specimen"><Icon name="upload" size={15} /></button
						>
					</div>
				</div>
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
						{#each presetIds as id (id)}{@const option = TOKEN_STORY_PRESETS[id]}<button
								class:chosen={preset === id}
								onclick={() => {
									preset = id;
									if (id !== 'small' && backend === 'wasm') backend = 'auto';
								}}
								disabled={blocked}
								aria-pressed={preset === id}
								><span
									><strong>{(tokenStoryParameterCount(option) / 1e6).toFixed(3)}M</strong><small
										>{id === 'small' ? 'Compact' : id === 'medium' ? 'Expanded' : 'Large'}</small
									></span
								><span>{option.layers} layers · {option.width} width · {option.heads} heads</span
								><span
									>{tokenStoryUnitCount(option).toLocaleString()} MLP channels · {option.context} context</span
								></button
							>{/each}
					</div>
					<div class="initialize-options">
						<label for="token-story-seed">Initialization seed</label><input
							id="token-story-seed"
							type="number"
							min="0"
							max="4294967295"
							bind:value={seed}
							disabled={blocked}
						/><span class="control-caption">Backend</span>
						<div class="backend-options">
							{#each ['auto', 'webgpu', 'wasm'] as device (device)}<button
									class:chosen={backend === device}
									onclick={() => (backend = device as TokenStoryBackend | 'auto')}
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
				<details class="lab-disclosure" bind:open={trainingOpen}>
					<summary>Training <small>Updates & checkpoints</small></summary>
					<div class="transport">
						<span>Training</span>
						<div class="segmented">
							{#each [null, 25, 100, 500] as steps (steps)}<button
									class:chosen={budget === steps}
									aria-pressed={budget === steps}
									onclick={() => (budget = steps)}
									disabled={blocked}>{steps ?? 'Continuous'}</button
								>{/each}
						</div>
						<small class="training-cadence"
							>Map + checkpoint every {mapCadence} updates · pause saves current state</small
						>
						{#if historical}<button
								class="secondary"
								onclick={() => selectSnapshot(null)}
								disabled={blocked}>Return to latest checkpoint</button
							>{/if}
						<p class="training-help">
							{historical
								? 'Training continues from the latest checkpoint. Return to it before starting.'
								: runtimeReady
									? 'Use Train in the action bar. Pause saves the current weights and measurements.'
									: 'Initialize a model or restore a saved checkpoint, then use Train in the action bar.'}
						</p>
					</div>
				</details>
				<details class="lab-disclosure">
					<summary>Generation settings <small>Sampling & batch comparison</small></summary>
					<div class="sample-controls">
						<span class="batch-caption"
							>Settings apply to the next generation. Sample runs a batch without a live trace.</span
						>
						<label for="token-story-sample-prompt">Sampling prompt</label><input
							id="token-story-sample-prompt"
							bind:value={samplePrompt}
							maxlength="100000"
							disabled={blocked}
						/><label for="token-story-sampling-seed">Seed</label><input
							id="token-story-sampling-seed"
							type="number"
							min="0"
							max="4294967295"
							bind:value={samplingSeed}
							disabled={blocked}
						/><label for="token-story-temperature">Temperature</label><input
							id="token-story-temperature"
							type="number"
							min="0.1"
							max="2"
							step="0.1"
							bind:value={temperature}
							disabled={blocked}
						/>
						<label for="token-story-top-k">Top-k</label><input
							id="token-story-top-k"
							type="number"
							min="1"
							max="256"
							bind:value={topK}
							disabled={blocked}
						/>
						<div class="segmented">
							{#each [32, 64, 128, 256] as length (length)}<button
									class:chosen={sampleLength === length}
									onclick={() => (sampleLength = length)}
									disabled={blocked}>{length}</button
								>{/each}
						</div>
						<button
							class="secondary"
							onclick={generate}
							disabled={!ready ||
								!Number.isInteger(topK) ||
								topK < 1 ||
								topK > 256 ||
								!Number.isInteger(samplingSeed) ||
								samplingSeed < 0 ||
								samplingSeed > 0xffffffff ||
								!Number.isFinite(temperature) ||
								temperature < 0.1 ||
								temperature > 2}><Icon name="play" size={12} />Sample</button
						>
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
						<strong>TinyStories · subword BPE</strong>
						<p>
							{corpus
								? `${corpus.splits.train.stories.toLocaleString()} training stories; ${corpus.splits.calibration.stories} calibration and ${corpus.splits.evaluation.stories} evaluation stories.`
								: '2,097 training stories and 184 held-out stories. Calibration and evaluation are separated by story.'}
						</p>
						<span>4,096 tokens · train-only BPE vocabulary</span><a
							href="https://huggingface.co/datasets/roneneldan/TinyStories"
							target="_blank"
							rel="noreferrer">Dataset & provenance <Icon name="right" size={10} /></a
						>
					</div>
				</details>
			</aside>
		{/snippet}
		<main class="token-story-main">
			<TokenStoryLivePanel
				prompt={samplePrompt}
				sample={liveGeneration ? null : sample}
				{tokenizer}
				frame={activityFrame}
				frames={playbackFrames}
				layer={playbackLayer}
				layers={config.layers}
				{selected}
				tokens={playbackTokens}
				prefix={playbackPrompt}
				running={liveGeneration && phase === 'generating'}
				replaying={phase === 'replaying'}
				paused={playbackPaused}
				stopping={stopRequested && playing}
				complete={playbackComplete}
				{blocked}
				availability={generationAvailability}
				pace={playbackPace}
				limit={sampleLength}
				{temperature}
				{samplingSeed}
				{topK}
				onprompt={(text) => (samplePrompt = text)}
				onnextlayer={nextPlaybackLayer}
				onnexttoken={nextPlaybackToken}
				onstop={stopPlayback}
				onpace={setPlaybackPace}
				onlayer={(layer) => (playbackLayer = layer)}
				onframe={inspectPlaybackFrame}
				onclear={returnToProbe}
			/>
			<div class="view-toolbar">
				<div class="segmented">
					<button
						class:chosen={viewMode === 'functional'}
						aria-pressed={viewMode === 'functional'}
						onclick={() => selectView('functional')}
						><Icon name="cube" size={12} />Functional</button
					><button
						class:chosen={viewMode === 'architecture'}
						aria-pressed={viewMode === 'architecture'}
						onclick={() => selectView('architecture')}
						><Icon name="network" size={12} />Model layout</button
					>
				</div>
				<div class="segmented encoding-options" role="group" aria-label="Activation encoding">
					<button
						class:chosen={activationEncoding === 'size'}
						aria-pressed={activationEncoding === 'size'}
						onclick={() => (activationEncoding = 'size')}
						title="Measured activity changes node size">Size</button
					>
					<button
						class:chosen={activationEncoding === 'brightness'}
						aria-pressed={activationEncoding === 'brightness'}
						onclick={() => (activationEncoding = 'brightness')}
						title="Fixed-size nodes; measured activity changes brightness and glow"
						>Brightness</button
					>
				</div>
				<div class="layer-options">
					<span>Layers</span><button
						class:chosen={layerFilter === null}
						aria-pressed={layerFilter === null}
						onclick={() => selectLayer(null)}>All</button
					>{#each Array.from({ length: config.layers }, (_, i) => i) as layer (layer)}<button
							class:chosen={layerFilter === layer}
							aria-pressed={layerFilter === layer}
							onclick={() => selectLayer(layer)}>L{layer + 1}</button
						>{/each}
				</div>
				<span class="architecture-label"
					>{config.layers} × {config.width} · {unitCount.toLocaleString()} units</span
				>
			</div>
			{#if focusedLayerHidden}<div class="layer-filter-note">
					Playback L{(playbackLayer ?? 0) + 1} is outside the L{(layerFilter ?? 0) + 1} filter. Showing
					the selected layer’s measured activity.
					<button onclick={() => selectLayer(null)}>Show all layers</button>
				</div>{/if}
			<div class="token-story-field" class:live-view={!!activityFrame}>
				{#if snapshot || activityFrame}<NeuralField
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
						{activationEncoding}
						layoutKey={viewMode}
						mode="activation"
						activityMode={!!activityFrame}
						activeLayer={activityFrame && !focusedLayerHidden ? playbackLayer : null}
						geometryLabel={viewMode === 'functional'
							? `Calibration activation geometry · step ${snapshot?.atlas.step}`
							: activityFrame
								? `Model layout · next-token frame ${activityFrame.index + 1} · step ${activityFrame.step}`
								: 'Layer / channel coordinates'}
						edgeLabel={viewMode === 'functional'
							? 'Selected neighbors · full fingerprint space'
							: 'Architectural coordinates · no graph edges'}
					/>{:else}<div class="field-empty">
						<Icon name="cube" size={38} />
						<h2>
							{busy ? 'Preparing the first measurement' : 'Explore a language model'}
						</h2>
						<p>
							{busy
								? status
								: 'Choose Generate live above to load the trained model, or Replay trace to explore recorded activations immediately. Open Model to build and train your own.'}
						</p>
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
				<TokenStoryMetricsPanel unit="token" metrics={record?.metrics ?? []} compact />
				<div class="geometry-footer">
					<span
						>{snapshot
							? `${snapshot.atlas.unitCount.toLocaleString()} / ${snapshot.atlas.unitCount.toLocaleString()} channels captured`
							: 'All channels will be captured'}</span
					><span
						>{snapshot
							? `${snapshot.atlas.dimensions} fixed calibration coordinates`
							: '8 windows × 16 token positions'}</span
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
							>{activityFrame
								? `Activity: measured final input token ${activityFrame.position + 1}, predicting output token ${activityFrame.index + 1}. Layer playback is paced, not GPU timing.`
								: matchingProbe
									? `Activity: prompt token ${token + 1}, measured step ${matchingProbe.step}.`
									: `Activity: calibration window 1, token ${(snapshot.atlas.tokenPositions[0]?.[0] ?? 0) + 1}.`}</span
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
			<details class="prompt-probe" bind:open={probeOpen} hidden={interventionOpen}>
				<summary>Prompt probe <span>Inspect every input position</span></summary>
				<div class="probe-input">
					<div class="prompt-seeds">
						<span>Try a context</span>{#each promptSeeds as choice (choice.name)}<button
								class:chosen={probePrompt === choice.text}
								onclick={() => (probePrompt = choice.text)}
								disabled={blocked}>{choice.name}</button
							>{/each}
					</div>
					<label for="token-story-probe-text">Probe context</label><textarea
						id="token-story-probe-text"
						bind:value={probePrompt}
						rows="2"
						maxlength="100000"
						spellcheck="false"
						disabled={blocked}
						placeholder="Enter a prompt using the subword vocabulary"></textarea><button
						class="secondary"
						onclick={inspectPrompt}
						disabled={!ready}><Icon name="activity" size={13} />Run prompt</button
					><span
						>{liveProbe
							? `Measured ${liveProbe.prompt.tokenIds.length} tokens at step ${liveProbe.step}`
							: 'This input is unmeasured. Run the prompt on a resident checkpoint to inspect its activations.'}</span
					>
				</div>
				<TokenStoryTokenization text={probePrompt} {tokenizer} context={config.context} />
			</details>
			<div class="token-story-inspector">
				<TokenStoryProbePanel
					interventionFirst={interventionOpen}
					liveFrame={activityFrame}
					{tokenizer}
					atlas={snapshot?.atlas ?? null}
					oncontext={(text) => {
						probePrompt = text;
						probeOpen = true;
						toolPanel = 'inspector';
						toolsCollapsed = false;
					}}
					probe={activityFrame ? null : liveProbe}
					{lesioned}
					{selected}
					{token}
					{config}
					busy={blocked}
					{interventionHint}
					prepareLabel={!runtimeReady
						? record?.checkpoint
							? 'Load model weights'
							: 'Load trained example'
						: historical
							? 'Return to latest checkpoint'
							: 'Measure prompt for intervention'}
					onprepare={!blocked &&
					((!runtimeReady && canLoadModel) ||
						(runtimeReady && (historical || !!activityFrame || !liveProbe)))
						? prepareIntervention
						: undefined}
					onselect={selectUnit}
					ontoken={(position) => (token = position)}
					onlesion={silenceUnit}
				/>
			</div>
		{/snippet}
		{#snippet evidence()}
			<details class="lab-disclosure evidence-drawer">
				<summary
					>Samples & evidence <small>Batch generation, learning curves and experiment history</small
					></summary
				>
				<div class="token-story-lower">
					<section class="sample-panel">
						<div class="lower-tabs">
							<button class:chosen={footerTab === 'samples'} onclick={() => (footerTab = 'samples')}
								><Icon name="book" size={13} />Generate & compare</button
							><button
								class:chosen={footerTab === 'history'}
								onclick={() => (footerTab = 'history')}
								><Icon name="activity" size={13} />Experiment history
								<span>{record?.observations.length ?? 0}</span></button
							>
						</div>
						{#if footerTab === 'samples'}
							<TokenStoryTokenization
								text={samplePrompt}
								{tokenizer}
								context={config.context}
								compact
							/>
							<div class="sample-output">
								{#if sample}<div class="sample-meta">
										<span
											>Step {sample.step} · sampling seed {sample.samplingSeed} · temperature {sample.temperature}
											· top-k {sample.topK}</span
										><span
											>{sample.tokenIds.length} / {sample.requestedTokens} tokens{sample.cancelled
												? ' · cancelled'
												: sample.stoppedOnEos
													? ' · end-of-story token'
													: ' · length limit'}</span
										>
									</div>
									<p>
										<span class="sample-prefix">{sample.prompt.text}</span>{sample.completion}
									</p>
									<div class="generation-coverage">
										Prefix: {sample.prompt.tokenIds.length} context tokens, {sample.prompt.text
											.length} retained characters{sample.prompt.truncatedTokens
											? `; ${sample.prompt.truncatedTokens} earlier tokens omitted`
											: ''}. Completion: {sample.completion.length} characters. The model uses the latest
										{config.context}
										tokens as generation advances.
									</div>
									<details class="generated-tokens">
										<summary
											>Inspect generated token boundaries · {sample.tokenIds.length} tokens</summary
										>
										<div>
											{#each sample.tokenIds as id, index (index)}<span
													title={`Vocabulary ID ${id}`}>{tokenPiece(id)}</span
												>{/each}
										</div>
									</details>
								{:else}<p class="empty-sample">
										Sampled text will appear here. Samples are observations of the current
										checkpoint, not evidence of story understanding.
									</p>{/if}
							</div>
							{#if (record?.samples?.length ?? 0) > 1}<div class="sample-history">
									<span>Recorded samples</span
									>{#each record?.samples ?? [] as past, i (`${past.step}-${i}`)}<button
											onclick={() => chooseSample(past)}
											disabled={blocked}
											aria-label={`Inspect sample ${i + 1}: step ${past.step}, seed ${past.samplingSeed}`}
											title={`Original prefix: ${past.prompt.original}`}
											class:chosen={sample === past}
											>Step {past.step} · seed {past.samplingSeed}</button
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
											probability changes; the archive retains all {config.vocabularySize.toLocaleString()}
											values.
										</p>
										<p>
											Original prompt; the saved tokenizer and context length reproduce the exact
											retained tokens.
										</p>
										<code class="intervention-prompt">{intervention.prompt}</code>
										<table>
											<thead
												><tr
													><th>Token</th><th>Intact</th><th>Lesioned</th><th>Δ percentage points</th
													></tr
												></thead
											><tbody
												>{#each archivedEffects(intervention) as effect (effect.id)}<tr
														><td>{tokenPiece(effect.id)}</td><td
															>{(effect.probability * 100).toFixed(3)}%</td
														><td>{(effect.lesioned * 100).toFixed(3)}%</td><td
															>{effect.delta > 0 ? '+' : ''}{number(effect.delta * 100, 3)}</td
														></tr
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
						<TokenStoryMetricsPanel unit="token" metrics={record?.metrics ?? []} />
					</section>
				</div>
			</details>
		{/snippet}
	</ResearchWorkspace>
	<footer class="token-story-status" role="status">
		<span class:working={busy}></span>
		<p>{status}</p>
		{#if phase === 'training'}<span
				>{latestMetric?.step ?? 0}{trainingTarget === null
					? ' · continuous'
					: ` / ${trainingTarget}`} updates</span
			>{/if}{#if measurementProgress.total > 0 && busy && phase !== 'generating'}<span
				>Calibration {measurementProgress.completed} / {measurementProgress.total}</span
			>{/if}<span>{latestMetric?.backend.toUpperCase() ?? 'No backend allocated'}</span><span
			>Resident step {residentStep ?? '—'}</span
		>{#if latestMetric?.stepMs}<span>{number(latestMetric.stepMs)} ms / update</span>{/if}
	</footer>
</div>

<style>
	.training-cadence {
		font: 12px/1.5 var(--mono);
		color: var(--faint);
		max-width: 165px;
	}
	.batch-caption {
		width: 100%;
		font: 12px var(--mono);
		color: var(--muted);
	}
	.prompt-seeds {
		grid-column: 1/-1;
		display: flex;
		align-items: center;
		gap: 5px;
		flex-wrap: wrap;
		margin-bottom: 1px;
	}
	.prompt-seeds > span {
		font: 12px var(--mono);
		color: var(--muted);
		margin-right: 4px;
	}
	.prompt-seeds button {
		background: transparent;
		border: 1px solid var(--line);
		border-radius: 4px;
		padding: 4px 6px;
		font: 12px var(--mono);
		color: var(--muted);
		margin: 0;
	}
	.prompt-seeds button.chosen {
		color: var(--accent);
		border-color: color-mix(in srgb, var(--accent) 50%, var(--line));
	}
	.generation-coverage {
		color: var(--muted);
		font: 12px/1.7 var(--mono);
		margin-top: 9px;
	}
	.generated-tokens {
		margin-top: 10px;
		color: var(--muted);
		font: 12px/1.6 var(--mono);
	}
	.generated-tokens summary {
		cursor: pointer;
	}
	.generated-tokens > div {
		display: flex;
		flex-wrap: wrap;
		gap: 3px;
		margin-top: 8px;
		max-height: 130px;
		overflow: auto;
	}
	.generated-tokens span {
		border: 1px solid var(--line);
		border-radius: 3px;
		padding: 3px 5px;
		color: var(--ink);
		overflow-wrap: anywhere;
	}
	.token-story-lab {
		color: var(--ink);
	}
	.file-input {
		display: none;
	}
	.token-story-toolbar {
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
	.token-story-main {
		min-width: 0;
	}
	.prompt-probe {
		background: var(--surface);
		border-bottom: 1px solid var(--line);
	}
	.prompt-probe > summary {
		padding: 9px 12px;
		color: var(--muted);
		font: 12px var(--mono);
		cursor: pointer;
	}
	.prompt-probe > summary span {
		color: var(--faint);
		font-size: 12px;
		margin-left: 9px;
	}
	.view-toolbar {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 9px 12px;
		border-bottom: 1px solid var(--line);
		background: var(--surface);
	}
	.encoding-options button {
		font: 12px var(--mono);
		padding: 5px 7px;
	}
	.layer-filter-note {
		display: flex;
		align-items: center;
		gap: 8px;
		flex-wrap: wrap;
		padding: 7px 12px;
		border-bottom: 1px solid var(--line);
		color: var(--muted);
		font: 12px/1.5 var(--mono);
	}
	.layer-filter-note button {
		background: transparent;
		border: 1px solid var(--line);
		border-radius: 3px;
		color: var(--accent);
		font: 12px var(--mono);
		padding: 3px 6px;
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
	.token-story-field {
		height: 480px;
		min-width: 0;
	}
	.token-story-field.live-view {
		height: clamp(340px, calc(100dvh - 610px), 480px);
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
	.token-story-inspector {
		min-width: 0;
		border-left: 1px solid var(--line);
		background: var(--surface);
	}
	.token-story-lower {
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
	.token-story-status {
		display: flex;
		align-items: center;
		gap: 15px;
		padding: 9px 17px;
		font: 12px var(--mono);
		color: var(--muted);
		background: var(--surface);
		flex-wrap: wrap;
	}
	.token-story-status > span:first-child {
		width: 4px;
		height: 4px;
		border-radius: 50%;
		background: var(--faint);
	}
	.token-story-status > span.working {
		background: var(--warning);
	}
	.token-story-status p {
		flex: 1;
	}
	.token-story-toolbar {
		padding: 16px 24px;
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
	.token-story-inspector {
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
	.token-story-field,
	.token-story-field.live-view {
		height: clamp(320px, calc(100dvh - 490px), 720px);
	}
	.token-story-lower {
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
	.token-story-status {
		padding: 12px 24px 20px;
		flex-wrap: wrap;
		gap: 10px;
	}
	@media (max-width: 1000px) {
		.token-story-inspector {
			grid-column: auto;
		}
		.token-story-lower {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	@media (max-width: 600px) {
		.token-story-toolbar {
			padding: 18px 12px;
		}
		.token-story-field,
		.token-story-field.live-view {
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
	.token-story-toolbar {
		position: sticky;
		top: 0;
		z-index: 20;
		background: var(--bg);
		min-height: 88px;
		gap: 16px;
	}
	.model-utilities {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 10px;
		padding: 14px;
		flex-wrap: wrap;
		border-bottom: 1px solid var(--line);
	}
	.training-help {
		color: var(--muted);
		line-height: 1.6;
		margin: 0;
		font-size: 13px;
	}
	.setup-panel .sample-controls {
		display: flex;
		flex-wrap: wrap;
		padding: 16px;
		gap: 10px;
	}
	.setup-panel .sample-controls label {
		width: 100%;
	}
	.setup-panel .sample-controls input {
		width: 100%;
		min-width: 0;
	}
</style>
