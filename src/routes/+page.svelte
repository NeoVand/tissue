<script lang="ts">
	import { onMount } from 'svelte';
	import { resolve } from '$app/paths';
	import NeuralField from '$lib/components/NeuralField.svelte';
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

	let tab = $state<'observatory' | 'journal' | 'methods'>('observatory');
	let phase = $state<
		'booting' | 'ready' | 'training' | 'measuring' | 'probing' | 'repairing' | 'error'
	>('booting');
	let status = $state('Preparing a small mind…');
	let error = $state('');
	let storageError = $state('');
	let seed = $state(42);
	let budget = $state(600);
	let mode = $state<'activation' | 'effect'>('activation');
	let selected = $state<number | null>(null);
	let token = $state(13);
	let example = $state(0);
	let cursor = $state(-1);
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
	let activationGeometry: GeometryResult | undefined;
	let effectGeometry: GeometryResult | undefined;
	let stopRequested = false;
	let generation = 0;
	let mounted = false;

	let busy = $derived(phase !== 'ready' && phase !== 'error');
	let ready = $derived(phase === 'ready');
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
				position,
				layer: Math.floor(id / MODEL_CONFIG.hidden),
				activation: probe?.activations[token]?.[id] ?? 0,
				effect: mode === 'effect' ? map.magnitudes[id] : undefined
			}))
			.filter((point) => map?.valid[point.id]) ?? []
	);
	let neighbors = $derived(selected === null ? [] : (map?.neighbors[selected] ?? []));
	let selectedActivation = $derived(
		selected === null ? null : probe?.activations[token]?.[selected]
	);
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
				provenance: { source: 'browser', appVersion: '0.1.0', userAgent: navigator.userAgent }
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
	async function openRun(record: RunRecord) {
		if (busy || !engine || !record.checkpoint) return;
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
			selected = null;
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
			tab = 'observatory';
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
				await openRun(saved);
			} catch (reason) {
				if (mounted) fail(reason);
			}
		} else await startNew();
	}

	async function compareRepair() {
		if (!engine || !run?.atlas?.effectFingerprints || selected === null || !ready || historical)
			return;
		phase = 'repairing';
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
			const response = await fetch(path);
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
	const repairNames: Record<string, string> = {
		activation: 'Activation neighbors',
		intervention: 'Intervention neighbors',
		'outgoing-weight': 'Outgoing-weight neighbors',
		random: 'Random same-layer units'
	};
	let latestRepair = $derived(run?.repairs?.at(-1));
	onMount(() => {
		mounted = true;
		geometryEngine = new GeometryEngine();
		void boot();
		return () => {
			mounted = false;
			generation++;
			stopRequested = true;
			geometryEngine?.dispose();
			void engine?.dispose();
		};
	});
</script>

<svelte:head>
	<title>Tissue — a living atlas of a small mind</title>
	<meta
		name="description"
		content="An experimental observatory for learning models. Train a small transformer, inspect its functional anatomy, and keep the evidence."
	/>
</svelte:head>

<div class="lab-shell">
	<header class="masthead">
		<a class="brand" href={resolve('/')} aria-label="Tissue home"
			><svg viewBox="0 0 40 40" aria-hidden="true"
				><ellipse cx="20" cy="20" rx="16" ry="7" transform="rotate(-35 20 20)" /><ellipse
					cx="20"
					cy="20"
					rx="16"
					ry="7"
					transform="rotate(35 20 20)"
				/><ellipse cx="20" cy="20" rx="7" ry="16" /></svg
			><span>tissue<span class="brand-dot">.</span></span></a
		>
		<span class="masthead-note eyebrow">An open notebook<br />on the shape of learning</span>
		<nav aria-label="Lab views">
			{#each ['observatory', 'journal', 'methods'] as item (item)}<button
					type="button"
					class:active={tab === item}
					onclick={() => (tab = item as typeof tab)}
					>{item === 'observatory'
						? 'Observatory'
						: item === 'journal'
							? 'Field journal'
							: 'Methods'}{#if item === 'journal'}<span class="nav-count"
							>{notebook.length + observedEntries}</span
						>{/if}</button
				>{/each}
		</nav>
		<a
			class="github-link eyebrow"
			href="https://github.com/NeoVand/tissue"
			target="_blank"
			rel="noreferrer">Open research ↗</a
		>
	</header>

	{#if error}<div class="error-banner" role="alert">
			<span>{error}</span>
			{#if phase === 'error'}
				{#if run?.checkpoint}<button type="button" onclick={() => run && openRun(run)}
						>Restore saved checkpoint</button
					>{/if}
				<button type="button" onclick={() => startNew('wasm')}>New run with WASM</button>
			{:else}<button type="button" onclick={() => (error = '')}>Dismiss</button>{/if}
		</div>{/if}
	{#if storageError}<div class="storage-banner" role="status">{storageError}</div>{/if}
	{#if tab === 'observatory'}
		<section class="intro">
			<div>
				<div class="eyebrow intro-label">
					<span class="tiny-cross">+</span> Study 001 / functional anatomy
				</div>
				<h1>A small mind,<br /><em>taking shape.</em></h1>
			</div>
			<div class="intro-copy">
				<p>
					Train it. Observe it. Change it.<br />A living experiment in what a model’s<br
						class="desktop-break"
					/> geometry can tell us.
				</p>
				<div class="live-status">
					<span class:working={busy}></span><span
						>{metrics?.backend?.toUpperCase() ?? 'LOCAL'} / {phase === 'training'
							? 'LEARNING'
							: phase === 'ready'
								? 'READY'
								: phase === 'error'
									? 'INTERRUPTED'
									: 'PREPARING'}</span
					>
				</div>
			</div>
		</section>
		<div class="workbench">
			<section class="specimen">
				<div class="specimen-toolbar">
					<div class="eyebrow">
						<span class="live-dot"></span>
						{historical ? 'Recorded specimen' : 'Live specimen'}
						<span class="subtle">/ seed {run?.seed ?? seed}</span>
					</div>
					<div class="map-toggle" aria-label="Geometry">
						<button
							type="button"
							class:chosen={mode === 'activation'}
							onclick={() => (mode = 'activation')}>Activation</button
						><button
							type="button"
							class:chosen={mode === 'effect'}
							onclick={() => (mode = 'effect')}
							disabled={!snapshot?.effect}>Intervention</button
						>
					</div>
				</div>
				<div class="field-wrap">
					<NeuralField
						{points}
						edges={map?.edges ?? []}
						{selected}
						onselect={(id) => (selected = id)}
						{mode}
						loading={!map}
					/>
				</div>
				<div class="map-footnote">
					<span class="field-scale"
						><span class="layer-dot first"></span> Layer 1 <span class="layer-dot second"></span> Layer
						2</span
					><span>{map ? NEURON_COUNT - points.length : 0} unresolved units omitted</span>
				</div>
				<div class="projection-strip">
					<div>
						<span class="eyebrow">Neighbors retained in 3D</span><strong
							>{percent(map?.neighborRetention)}</strong
						>
					</div>
					<div>
						<span class="eyebrow">Variance shown</span><strong
							>{percent(map?.explainedVariance)}</strong
						>
					</div>
					<p>
						Position = functional similarity.<br />Size = compressed {mode === 'effect'
							? 'effect magnitude'
							: 'activation'}.
					</p>
				</div>
				<div class="checkpoint-strip">
					<div class="checkpoint-heading">
						<span class="eyebrow">Development / checkpoint {snapshot?.step ?? '—'}</span
						>{#if historical}<button type="button" class="text-button" onclick={() => (cursor = -1)}
								>Return to live</button
							>{:else}<span class="eyebrow subtle">Measured checkpoints</span>{/if}
					</div>
					<div class="checkpoints">
						{#each run?.snapshots ?? [] as saved, i (saved.step)}<button
								type="button"
								aria-label={`View checkpoint ${saved.step}`}
								class:current={snapshot?.step === saved.step}
								onclick={() => selectCheckpoint(i)}><span></span>{saved.step}</button
							>{/each}
					</div>
				</div>
			</section>

			<aside class="instruments">
				<div class="instrument-heading">
					<span class="eyebrow">01 / grow</span><span class="specimen-number">T—001</span>
				</div>
				<h2>Binding, from scratch.</h2>
				<p class="instrument-description">
					A tiny transformer learns which value belongs to a queried key.
				</p>
				<div class="model-specs">
					<span>2 layers</span><span>4 heads</span><span
						>{run ? (run.initialization.parameterCount / 1000).toFixed(1) : '—'}k parameters</span
					>
				</div>
				<div class="primary-metric">
					<div>
						<span class="eyebrow">Held-out answer accuracy</span><strong data-testid="accuracy"
							>{percent(metrics?.accuracy)}</strong
						>
					</div>
					<span class="metric-subtitle"
						>{metrics?.validationExamples ?? 96} fixed<br />unseen prompts</span
					>
				</div>
				<div class="metric-row">
					<span
						>Answer loss <strong data-testid="loss">{decimal(metrics?.validationLoss)}</strong>
						<small>nats</small></span
					><span>Step <strong data-testid="step">{metrics?.step ?? 0}</strong></span>
				</div>
				<div class="baseline-label">Chance: 12.5% · random input-copy: 33.3%</div>
				<div class="training-controls">
					<label
						>Growth interval<select bind:value={budget} disabled={busy}
							><option value={100}>100 steps</option><option value={300}>300 steps</option><option
								value={600}>600 steps</option
							><option value={2000}>2,000 steps</option></select
						></label
					>{#if phase === 'training'}<button type="button" class="primary" onclick={pause}
							>Ⅱ Pause training</button
						>{:else}<button type="button" class="primary" onclick={train} disabled={!ready || !run}
							>↗ {metrics?.step ? 'Continue training' : 'Start training'}</button
						>{/if}
				</div>
				<p class="operation-status" aria-live="polite">{status}</p>
				{#if phase === 'measuring' || phase === 'repairing'}<button
						type="button"
						class="text-button cancel-analysis"
						onclick={() => {
							void engine?.pause().catch(fail);
						}}>Cancel this measurement</button
					>{/if}
				<p class="reference-hint">
					First recorded run: a long plateau, then retrieval at 2,000 steps. <button
						type="button"
						class="text-button"
						onclick={() => openReference()}
						disabled={busy}>Explore that specimen ↗</button
					>
				</p>
				<div class="instrument-rule"></div>
				<div class="instrument-heading">
					<span class="eyebrow">02 / inspect</span><span class="eyebrow subtle"
						>{historical ? 'Saved prompt' : 'Calibration prompt'}</span
					>
				</div>
				<div class="prompt-heading">
					<span>Follow a token.</span><select
						aria-label="Calibration prompt"
						bind:value={example}
						onchange={changeExample}
						disabled={!ready || historical}
						>{#each Array.from({ length: run?.atlas?.examples.length ?? 16 }, (_, i) => i) as i (i)}<option
								value={i}>Prompt {i + 1}</option
							>{/each}</select
					>
				</div>
				<div class="tokens" aria-label="Input tokens">
					{#each probe?.example.tokens ?? [] as text, i (`${i}-${text}`)}<button
							type="button"
							class:token-active={token === i}
							onclick={() => (token = i)}
							aria-label={`Inspect token ${i + 1}: ${text}`}>{text}</button
						>{/each}<span class="answer-arrow">→</span><span class="expected-answer"
						>{probe?.example.answer ?? '?'}</span
					>
				</div>
				<div class="probability-heading">
					<span class="eyebrow">Next-token prediction</span><span class="eyebrow"
						>{lesionProbe && !historical ? 'Unit silenced' : 'Intact model'}</span
					>
				</div>
				<div class="probabilities" aria-label="Answer probabilities">
					{#each probe?.probabilities ?? [] as probability, i (i)}<div
							class:correct={String(i) === probe?.example.answer}
						>
							<span class="prob-value">{(probability * 100).toFixed(0)}<small>%</small></span>
							<div class="prob-track">
								<span style:height={`${Math.max(0, probability * 100)}%`}></span>
							</div>
							<span class="prob-digit"
								>{i}{#if String(i) === probe?.example.answer}<span class="correct-dot"
									></span>{/if}</span
							>
						</div>{/each}
				</div>
				<p class="probability-note">
					Dot marks the correct answer. Other tokens: {percent(probe?.otherProbability)}.
				</p>
				{#if lesionProbe && !historical}<button
						type="button"
						class="text-button restore"
						onclick={() => (lesionProbe = null)}>Restore intact view</button
					>{/if}
			</aside>
		</div>

		<div class="evidence-row">
			<section class="learning-panel">
				<div class="section-heading">
					<span class="eyebrow">03 / evidence of learning</span><span class="curve-legend"
						><i></i> Held-out loss <i class="dashed"></i> Uniform answer prior</span
					>
				</div>
				<LearningCurve metrics={run?.metrics ?? []} selectedStep={snapshot?.step ?? 0} />
			</section>
			<section class="neuron-panel">
				<div class="section-heading">
					<span class="eyebrow">04 / intervention bench</span><select
						aria-label="Select neuron"
						value={selected ?? ''}
						onchange={(event) =>
							(selected =
								event.currentTarget.value === '' ? null : Number(event.currentTarget.value))}
						><option value="">Pick a neuron</option
						>{#each Array.from({ length: NEURON_COUNT }, (_, id) => id) as id (id)}<option
								value={id}>L{Math.floor(id / 128) + 1} · {String(id % 128).padStart(3, '0')}</option
							>{/each}</select
					>
				</div>
				{#if selected !== null}<div class="neuron-detail">
						<strong
							>L{Math.floor(selected / 128) + 1}<span> / </span>{String(selected % 128).padStart(
								3,
								'0'
							)}</strong
						>
						<div>
							Activation <b>{decimal(selectedActivation)}</b><br /><span class="subtle"
								>at token {token + 1}</span
							>
						</div>
					</div>
					<div class="neighbor-list">
						<span class="eyebrow subtle">Nearest in fingerprint space</span
						>{#each neighbors as id (id)}<button type="button" onclick={() => (selected = id)}
								>L{Math.floor(id / 128) + 1}·{id % 128}</button
							>{/each}
					</div>{:else}<p class="select-invitation">
						Every point is a place to ask a question.<br /><span
							>Select a neuron in the specimen or the menu.</span
						>
					</p>{/if}
				<div class="intervention-actions">
					<button
						type="button"
						class="secondary"
						onclick={silence}
						disabled={selected === null || !ready || historical}>Silence selected unit</button
					><button
						type="button"
						class="text-button"
						onclick={measureEffects}
						disabled={!ready || !run}>Measure all effects ↗</button
					>
				</div>
				{#if phase === 'measuring'}<div class="measurement-progress">
						<progress max={progress.total} value={progress.completed}></progress><span
							>{progress.completed} / {progress.total} neurons</span
						>
					</div>{/if}
			</section>
		</div>
		<section class="repair-panel">
			<div class="repair-intro">
				<div>
					<span class="eyebrow">05 / ask the map to predict</span>
					<h2>Where does repair happen?</h2>
					<p>
						Silence one neuron. Let eight surviving units learn for 50 updates.<br />Compare four
						ways of choosing those units, from the same starting model.
					</p>
				</div>
				<button
					type="button"
					class="primary"
					onclick={compareRepair}
					disabled={!ready || selected === null || historical || !run?.atlas?.effectFingerprints}
					>Compare repair neighborhoods ↗</button
				>
			</div>
			{#if !run?.atlas?.effectFingerprints}<p class="repair-hint">
					Select a neuron and measure all effects to prepare this experiment.
				</p>{/if}
			{#if latestRepair}
				<div class="repair-baseline">
					<span class="eyebrow"
						>Pilot / seed {latestRepair.seed} / checkpoint {latestRepair.step} / neuron {latestRepair.lesionNeuron}</span
					><span
						>Intact loss <b>{latestRepair.intact.loss.toFixed(4)}</b> → injured
						<b>{latestRepair.lesioned.loss.toFixed(4)}</b></span
					>
				</div>
				<div class="repair-table-wrap">
					<table>
						<thead
							><tr
								><th>Repair neighborhood</th><th>Answer loss</th><th>Accuracy</th><th
									>Weights allowed to learn</th
								></tr
							></thead
						><tbody
							>{#each latestRepair.arms as arm (arm.method)}<tr
									><td>{repairNames[arm.method]}</td><td>{arm.curve.at(-1)?.loss.toFixed(4)}</td><td
										>{percent(arm.curve.at(-1)?.accuracy)}</td
									><td>{arm.trainableParameters}</td></tr
								>{/each}</tbody
						>
					</table>
				</div>
				<p class="repair-caveat">
					{latestRepair.caveat} Improvements beyond the intact model also need an unlesioned fine-tuning
					control. One pilot does not establish a superior geometry.
				</p>
				<details>
					<summary>Inspect neighborhood membership and protocol</summary>
					<p>
						All four arms use the same training batches, optimizer initialization, update count, and
						number of trainable weights. The lesion remains clamped. The original model is restored
						unchanged.
					</p>
					{#each latestRepair.arms as arm (arm.method)}<p>
							<b>{repairNames[arm.method]}:</b>
							{arm.neurons.map((id) => `L${Math.floor(id / 128) + 1}·${id % 128}`).join(', ')}
						</p>{/each}
				</details>
			{/if}
		</section>

		<div class="session-bar">
			<div>
				<span class="eyebrow">A reproducible specimen</span><label
					>Seed <input
						aria-label="New run seed"
						type="number"
						min="0"
						max="999999"
						step="1"
						bind:value={seed}
						disabled={busy}
					/></label
				><button type="button" class="text-button" onclick={() => startNew()} disabled={busy}
					>New run</button
				>
			</div>
			<div>
				<span class="saved-indicator"
					>{savedAt ? `Saved locally · ${savedAt}` : 'Saving in this browser'}</span
				><button
					type="button"
					class="secondary"
					onclick={() => run && exportRun(run)}
					disabled={!run?.checkpoint || busy}>Export experiment ↓</button
				><button
					type="button"
					class="text-button"
					onclick={() => importInput.click()}
					disabled={busy}>Import</button
				>
			</div>
		</div>
	{:else if tab === 'journal'}
		<section class="journal-intro">
			<div class="eyebrow">The experiment is the story</div>
			<h1>A record of<br /><em>looking closely.</em></h1>
			<p>
				Questions, observations, wrong turns, and the evidence we keep.<br />Your runs live in this
				browser. Export them to carry the full record with you.
			</p>
		</section>
		<div class="journal-columns">
			<section class="research-notes">
				<div class="section-heading">
					<span class="eyebrow">Research notebook</span><span class="eyebrow subtle"
						>Started September 2026</span
					>
				</div>
				{#each [...notebook].sort((a, b) => b.id.localeCompare(a.id)) as entry (entry.id)}<article
						class="notebook-entry"
					>
						<div class="entry-meta">
							<span class="entry-number">{entry.id}</span><span class="entry-kind"
								>{entry.kind}</span
							><time>{entry.date}</time>
						</div>
						<h2>{entry.title}</h2>
						<p>{entry.text}</p>
						<div class="entry-lesson">{entry.lesson}</div>
						{#if entry.links}<div class="entry-links">
								{#each entry.links as link (link.href)}
									<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -->
									<a href={link.href} target="_blank" rel="noreferrer">{link.title} ↗</a>{/each}
							</div>{/if}
					</article>{/each}
			</section>
			<section class="run-archive">
				<div class="section-heading">
					<span class="eyebrow">Specimen archive</span><button
						type="button"
						class="text-button"
						onclick={() => importInput.click()}
						disabled={busy}>Import run</button
					>
				</div>
				<div class="reference-catalog">
					<span class="eyebrow subtle">Recorded studies / available to explore</span
					>{#each references as reference (reference.seed)}<article>
							<div class="reference-number">{String(reference.seed).padStart(2, '0')}</div>
							<div>
								<h3>{reference.title}</h3>
								<p>{reference.description}</p>
								<button
									type="button"
									class="text-button"
									onclick={() => openReference(reference.path)}
									disabled={busy}>Open measured specimen ↗</button
								>
							</div>
						</article>{/each}
				</div>
				{#each runs as record (record.id)}<article class="run-card">
						<div class="entry-meta">
							<span class="entry-kind"
								>{record.provenance.source === 'reference'
									? 'Reference run'
									: record.id === run?.id
										? 'Current run'
										: 'Saved run'}</span
							><time>{new Date(record.createdAt).toLocaleDateString()}</time>
						</div>
						<h3>{record.title}</h3>
						<div class="run-stats">
							<span><b>{record.metrics.at(-1)?.step ?? 0}</b> steps</span><span
								><b>{percent(record.metrics.at(-1)?.accuracy)}</b> accuracy</span
							>
						</div>
						<LearningCurve
							metrics={record.metrics}
							selectedStep={record.metrics.at(-1)?.step ?? 0}
						/>
						<div class="run-actions">
							<button
								type="button"
								class="text-button"
								onclick={() => openRun(record)}
								disabled={busy || !record.checkpoint}>Open specimen ↗</button
							><button type="button" class="text-button" onclick={() => exportRun(record)}
								>Export ↓</button
							>
						</div>
					</article>{/each}
				{#if run}<div class="field-note">
						<label for="field-note">Leave a field note</label><textarea
							id="field-note"
							bind:value={note}
							placeholder="What surprised you? What should we test next?"
							rows="3"></textarea><button
							type="button"
							class="secondary"
							onclick={addNote}
							disabled={!note.trim()}>Keep this observation</button
						>
					</div>{/if}
			</section>
		</div>
		{#if run}<section class="activity-log">
				<div class="section-heading">
					<span class="eyebrow">Current run / observations</span><span class="eyebrow subtle"
						>Seed {run.seed}</span
					>
				</div>
				{#each [...run.observations].reverse() as observation (observation.id)}<article>
						<div class="observation-marker">
							<span></span><span class="eyebrow">{observation.step}</span>
						</div>
						<div>
							<div class="entry-meta">
								<span class="entry-kind">{observation.kind}</span><time
									>{new Date(observation.time).toLocaleTimeString()}</time
								>
							</div>
							<h3>{observation.title}</h3>
							<p>{observation.detail}</p>
						</div>
					</article>{/each}
			</section>{/if}
	{:else}
		<section class="journal-intro">
			<div class="eyebrow">The instrument, explained</div>
			<h1>Beautiful enough to explore.<br /><em>Precise enough to question.</em></h1>
		</section>
		<div class="methods-grid">
			<article>
				<span class="eyebrow">01 / the specimen</span>
				<h2>A deliberately small mind.</h2>
				<p>
					A causal transformer with two layers, width 32, four attention heads, and 128 ReLU MLP
					units per layer. It predicts one answer token after a 14-token binding prompt. We optimize
					answer cross-entropy, not the predictable punctuation.
				</p>
				<p>
					Eight possible values; three distinct values in each prompt. Uniform value guessing gets
					12.5% accuracy. Randomly copying one of the three supplied values gets 33.3%.
				</p>
			</article>
			<article>
				<span class="eyebrow">02 / the map</span>
				<h2>Every coordinate has a source.</h2>
				<p>
					Activation fingerprints concatenate each neuron’s responses across fixed calibration
					prompts and all token positions. We center and normalize each fingerprint, then use PCA to
					produce three coordinates.
				</p>
				<p>
					Intervention fingerprints record changes in answer probabilities after zeroing a unit
					across all token positions. Effects are normalized for direction; their magnitude is
					preserved separately. Links connect neighbors in the original fingerprint space.
				</p>
			</article>
			<article>
				<span class="eyebrow">03 / the controls</span>
				<h2>Keep the picture accountable.</h2>
				<p>
					Neuron positions are derived after training updates. No spatial loss encourages a
					particular shape. Successive maps are aligned by an orthogonal transformation; alignment
					does not change their pairwise distances.
				</p>
				<p>
					Neighbor retention measures the local relationships that survive projection. Variance
					shown measures the fraction retained by three principal components. Neither is an
					interpretability score. Silencing a unit is an intervention on this task, not proof of a
					universal concept.
				</p>
			</article>
			<article>
				<span class="eyebrow">04 / the record</span>
				<h2>Keep enough to do it again.</h2>
				<p>
					Runs include seed, architecture version, backend, held-out metrics, checkpoint maps,
					prompts, interventions, notes, raw latest fingerprints, model weights, optimizer moments,
					and training RNG state. Export them as JSON and import to continue.
				</p>
				<p>
					Assignment mappings are disjoint between training, calibration, and evaluation. Inspection
					uses a separate deterministic set and does not consume the training random stream. GPU
					arithmetic can still vary across hardware.
				</p>
			</article>
		</div>
		<div class="research-next">
			<span class="eyebrow">The next experiment</span>
			<h2>Does a neighborhood know how to heal?</h2>
			<p>
				Choose repair neighborhoods before damaging the model. Compare recovery against activation
				neighbors, outgoing weights, matched random units, and repair gradients. The current atlas
				establishes measurement and intervention; predictive repair remains an open experiment.
			</p>
		</div>
	{/if}
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
	<footer>
		<span class="brand-footer">tissue.</span><span
			>A laboratory for discovering the shape of computation.</span
		><span class="eyebrow">Observe / intervene / learn</span>
	</footer>
</div>

<style>
	.lab-shell {
		max-width: 1600px;
		margin: 0 auto;
		padding: 0 52px;
	}
	.masthead {
		height: 100px;
		display: flex;
		align-items: center;
		gap: 28px;
		border-bottom: 1px solid var(--line);
	}
	.brand {
		display: flex;
		align-items: center;
		gap: 11px;
		text-decoration: none;
	}
	.brand > span {
		font-size: 32px;
		font-weight: 600;
		letter-spacing: -1.6px;
	}
	.brand-dot {
		color: var(--clay);
	}
	.brand svg {
		width: 35px;
		height: 35px;
		fill: none;
		stroke: var(--forest);
		stroke-width: 1.1;
	}
	.masthead-note {
		line-height: 1.7;
		color: var(--muted);
		border-left: 1px solid var(--line);
		padding-left: 25px;
	}
	nav {
		display: flex;
		align-self: stretch;
		gap: 30px;
		margin-left: auto;
	}
	nav button {
		font-size: 13px;
		background: none;
		border: 0;
		position: relative;
		padding: 0;
		color: var(--muted);
	}
	nav button.active {
		color: var(--ink);
	}
	nav button.active::after {
		content: '';
		position: absolute;
		height: 2px;
		background: var(--forest);
		bottom: -1px;
		left: 0;
		right: 0;
	}
	.nav-count {
		font: 9px var(--mono);
		background: #e6e8dc;
		padding: 3px 5px;
		border-radius: 3px;
		margin-left: 7px;
	}
	.github-link {
		text-decoration: none;
		margin-left: 23px;
		white-space: nowrap;
	}
	.intro {
		display: flex;
		justify-content: space-between;
		align-items: flex-end;
		padding: 38px 0 33px;
	}
	.intro-label {
		color: var(--forest);
		display: flex;
		gap: 9px;
		align-items: center;
	}
	.tiny-cross {
		font-size: 20px;
		line-height: 10px;
	}
	h1 {
		font: 66px/0.98 var(--serif);
		letter-spacing: -1.5px;
		margin: 19px 0 0;
		font-weight: 400;
	}
	h1 em {
		font-weight: 400;
		color: var(--forest);
	}
	.intro-copy {
		width: 307px;
		padding-bottom: 2px;
	}
	.intro-copy p {
		font-size: 14px;
		line-height: 1.65;
		margin: 0 0 23px;
		color: var(--muted);
	}
	.live-status {
		display: flex;
		align-items: center;
		gap: 8px;
		font: 10px var(--mono);
		letter-spacing: 0.08em;
	}
	.live-status > span:first-child,
	.live-dot {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: var(--forest);
		display: inline-block;
	}
	.live-status > span.working {
		background: var(--clay);
		animation: breathe 1.5s infinite;
	}
	.workbench {
		display: grid;
		grid-template-columns: minmax(0, 1fr) 338px;
		border: 1px solid var(--line);
		border-radius: 7px;
		overflow: hidden;
		background: var(--surface);
	}
	.specimen {
		min-width: 0;
		background: var(--paper);
	}
	.specimen-toolbar {
		height: 57px;
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 0 22px;
		border-bottom: 1px solid var(--line);
	}
	.specimen-toolbar .eyebrow {
		letter-spacing: 0.07em;
	}
	.live-dot {
		margin-right: 7px;
	}
	.map-toggle {
		display: flex;
		gap: 2px;
		border: 1px solid var(--line);
		border-radius: 4px;
		padding: 3px;
	}
	.map-toggle button {
		background: transparent;
		border: 0;
		padding: 6px 10px;
		border-radius: 2px;
		color: var(--muted);
		font-size: 11px;
	}
	.map-toggle .chosen {
		color: var(--forest);
		background: #e4e8db;
	}
	.field-wrap {
		height: 435px;
		position: relative;
	}
	.field-scale {
		display: flex;
		align-items: center;
		gap: 7px;
		font: 10px var(--mono);
		color: var(--muted);
		pointer-events: none;
	}
	.map-footnote {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 12px;
		padding: 9px 24px;
		border-top: 1px solid var(--line);
		font: 9px var(--mono);
		color: var(--muted);
	}
	.layer-dot {
		display: inline-block;
		width: 6px;
		height: 6px;
		border-radius: 50%;
	}
	.layer-dot.first {
		background: var(--clay);
	}
	.layer-dot.second {
		background: var(--forest);
		margin-left: 12px;
	}
	.projection-strip {
		border-top: 1px solid var(--line);
		display: flex;
		align-items: center;
		gap: 32px;
		padding: 19px 24px;
	}
	.projection-strip > div {
		display: grid;
		gap: 7px;
	}
	.projection-strip .eyebrow {
		color: var(--muted);
		font-size: 9px;
		letter-spacing: 0.05em;
	}
	.projection-strip strong {
		font: 20px var(--mono);
		font-weight: 400;
	}
	.projection-strip p {
		margin: 0 0 0 auto;
		color: var(--muted);
		font-size: 10px;
		line-height: 1.7;
	}
	.checkpoint-strip {
		padding: 18px 24px 21px;
		border-top: 1px solid var(--line);
		min-height: 93px;
	}
	.checkpoint-heading {
		display: flex;
		align-items: center;
		justify-content: space-between;
	}
	.checkpoint-heading .eyebrow {
		font-size: 9px;
		letter-spacing: 0.07em;
	}
	.checkpoint-heading button {
		font-size: 11px;
	}
	.checkpoints {
		display: flex;
		gap: 0;
		margin-top: 14px;
		flex-wrap: wrap;
	}
	.checkpoints button {
		flex: 1;
		min-width: 32px;
		max-width: 65px;
		background: none;
		border: 0;
		font: 9px var(--mono);
		color: var(--muted);
		position: relative;
		padding: 0;
		text-align: left;
	}
	.checkpoints button::before {
		content: '';
		height: 1px;
		position: absolute;
		top: 3px;
		left: 5px;
		right: 0;
		background: var(--line);
	}
	.checkpoints button:last-child::before {
		display: none;
	}
	.checkpoints button > span {
		display: block;
		width: 7px;
		height: 7px;
		border: 1px solid #a5ad9a;
		border-radius: 50%;
		margin-bottom: 7px;
		background: var(--paper);
		position: relative;
		z-index: 1;
	}
	.checkpoints button.current {
		color: var(--forest);
	}
	.checkpoints button.current > span {
		background: var(--forest);
		border-color: var(--forest);
		box-shadow: 0 0 0 3px #dfe5d5;
	}
	.instruments {
		border-left: 1px solid var(--line);
		padding: 22px 24px 20px;
	}
	.instrument-heading {
		display: flex;
		justify-content: space-between;
		align-items: center;
	}
	.instrument-heading > .eyebrow:first-child {
		color: var(--forest);
	}
	.specimen-number {
		font: 10px var(--mono);
		color: var(--muted);
	}
	h2 {
		font: 29px/1.14 var(--serif);
		font-weight: 400;
		letter-spacing: -0.3px;
		margin: 18px 0 9px;
	}
	.instrument-description {
		color: var(--muted);
		font-size: 12px;
		line-height: 1.6;
		margin: 0;
	}
	.model-specs {
		display: flex;
		gap: 12px;
		font: 9px var(--mono);
		margin: 17px 0 24px;
		color: var(--muted);
	}
	.primary-metric {
		display: flex;
		align-items: flex-end;
		justify-content: space-between;
	}
	.primary-metric > div {
		display: grid;
		gap: 7px;
	}
	.primary-metric .eyebrow {
		font-size: 9px;
		letter-spacing: 0.06em;
	}
	.primary-metric strong {
		font: 43px/1.1 var(--serif);
		letter-spacing: -1px;
	}
	.metric-subtitle {
		color: var(--muted);
		font: 9px/1.7 var(--mono);
		text-align: right;
		padding-bottom: 4px;
	}
	.metric-row {
		display: flex;
		align-items: center;
		justify-content: space-between;
		font-size: 11px;
		margin-top: 17px;
	}
	.metric-row strong {
		font: 12px var(--mono);
		margin-left: 4px;
	}
	.metric-row small {
		color: var(--muted);
		font-size: 10px;
	}
	.baseline-label {
		font-size: 9px;
		color: var(--muted);
		margin-top: 10px;
	}
	.training-controls {
		display: flex;
		gap: 10px;
		align-items: flex-end;
		margin-top: 21px;
	}
	.training-controls label {
		display: grid;
		gap: 6px;
		font-size: 10px;
		color: var(--muted);
	}
	select {
		background: transparent;
		border: 1px solid var(--line);
		border-radius: 3px;
		padding: 8px 7px;
		font-size: 11px;
		color: var(--ink);
	}
	.training-controls select {
		height: 39px;
	}
	.training-controls .primary {
		font-size: 11px;
		padding: 11px 12px;
		white-space: nowrap;
		flex: 1;
		height: 39px;
	}
	.operation-status {
		font: 9px/1.5 var(--mono);
		color: var(--muted);
		margin: 11px 0 0;
		min-height: 14px;
	}
	.instrument-rule {
		height: 1px;
		background: var(--line);
		margin: 22px 0;
	}
	.prompt-heading {
		display: flex;
		align-items: center;
		justify-content: space-between;
		margin: 16px 0 12px;
		font-size: 12px;
	}
	.prompt-heading select {
		font-size: 10px;
		padding: 5px;
		max-width: 104px;
	}
	.tokens {
		display: flex;
		gap: 2px;
		align-items: center;
		flex-wrap: wrap;
	}
	.tokens button {
		font: 12px var(--mono);
		padding: 5px 3px;
		border: 0;
		border-radius: 2px;
		background: #edeee6;
		min-width: 14px;
	}
	.tokens .token-active {
		background: var(--acid);
		box-shadow: inset 0 -2px var(--forest);
	}
	.answer-arrow {
		color: var(--muted);
		margin: 0 2px;
		font-size: 13px;
	}
	.expected-answer {
		font: 14px var(--mono);
		color: var(--forest);
	}
	.probability-heading {
		display: flex;
		justify-content: space-between;
		margin-top: 20px;
	}
	.probability-heading .eyebrow {
		font-size: 8px;
		letter-spacing: 0.03em;
		color: var(--muted);
	}
	.probabilities {
		display: flex;
		gap: 7px;
		margin-top: 12px;
	}
	.probabilities > div {
		flex: 1;
		text-align: center;
	}
	.prob-value {
		display: block;
		font: 9px var(--mono);
		color: var(--muted);
		margin-bottom: 5px;
	}
	.prob-value small {
		font-size: 7px;
	}
	.prob-track {
		height: 46px;
		background: #ebece4;
		display: flex;
		align-items: flex-end;
		overflow: hidden;
	}
	.prob-track > span {
		background: #b7bfae;
		width: 100%;
		transition: height 0.25s;
	}
	.correct .prob-track > span {
		background: var(--forest);
	}
	.prob-digit {
		display: inline-flex;
		align-items: center;
		gap: 2px;
		font: 11px var(--mono);
		margin-top: 6px;
		position: relative;
	}
	.correct-dot {
		position: absolute;
		right: -6px;
		top: 4px;
		width: 3px;
		height: 3px;
		background: var(--forest);
		border-radius: 50%;
	}
	.probability-note {
		font-size: 9px;
		color: var(--muted);
		margin: 9px 0 0;
		line-height: 1.5;
	}
	.restore {
		font-size: 10px;
		margin-top: 9px;
	}
	.evidence-row {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 40px;
		margin-top: 30px;
	}
	.section-heading {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
		padding-bottom: 15px;
		border-bottom: 1px solid var(--line);
	}
	.learning-panel .section-heading {
		margin-bottom: 16px;
	}
	.curve-legend {
		font-size: 9px;
		color: var(--muted);
		display: flex;
		gap: 5px;
		align-items: center;
	}
	.curve-legend i {
		width: 12px;
		height: 2px;
		background: var(--forest);
		margin-left: 7px;
	}
	.curve-legend i.dashed {
		background: none;
		border-top: 1px dashed var(--clay);
	}
	.neuron-panel .section-heading {
		padding-bottom: 9px;
	}
	.neuron-panel .section-heading select {
		padding: 5px;
		font-size: 10px;
	}
	.select-invitation {
		font: 22px/1.5 var(--serif);
		margin: 20px 0;
	}
	.select-invitation span {
		font:
			11px 'DM Sans',
			sans-serif;
		color: var(--muted);
	}
	.neuron-detail {
		display: flex;
		gap: 25px;
		align-items: center;
		margin: 17px 0 12px;
	}
	.neuron-detail > strong {
		font: 30px var(--mono);
		letter-spacing: -2px;
	}
	.neuron-detail > strong > span {
		color: #a4ac99;
	}
	.neuron-detail > div {
		font-size: 10px;
		line-height: 1.6;
	}
	.neuron-detail b {
		font-family: var(--mono);
		font-weight: 400;
	}
	.neighbor-list {
		display: flex;
		gap: 6px;
		align-items: center;
		flex-wrap: wrap;
		margin-bottom: 15px;
	}
	.neighbor-list .eyebrow {
		font-size: 8px;
		margin-right: 4px;
	}
	.neighbor-list button {
		border: 0;
		background: #e7eadd;
		color: var(--forest);
		border-radius: 3px;
		padding: 4px 6px;
		font: 9px var(--mono);
	}
	.intervention-actions {
		display: flex;
		gap: 18px;
		align-items: center;
	}
	.intervention-actions button {
		font-size: 10px;
	}
	.intervention-actions .secondary {
		padding: 9px 12px;
	}
	.measurement-progress {
		display: flex;
		gap: 15px;
		align-items: center;
		margin-top: 13px;
		font: 10px var(--mono);
	}
	progress {
		height: 4px;
		flex: 1;
		accent-color: var(--forest);
	}
	.cancel-analysis {
		font-size: 10px;
		margin-top: 8px;
	}
	.reference-hint {
		font-size: 10px;
		line-height: 1.6;
		color: var(--muted);
		margin: 13px 0 0;
	}
	.reference-hint button {
		font-size: 10px;
	}
	.repair-panel {
		border-top: 1px solid var(--line);
		margin-top: 30px;
		padding: 25px 0 4px;
	}
	.repair-intro {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 25px;
	}
	.repair-intro h2 {
		margin-top: 12px;
		font-size: 32px;
	}
	.repair-intro p,
	.repair-hint {
		font-size: 12px;
		line-height: 1.7;
		color: var(--muted);
	}
	.repair-intro button {
		font-size: 11px;
	}
	.repair-baseline {
		display: flex;
		justify-content: space-between;
		gap: 15px;
		padding: 18px 0;
		margin-top: 15px;
		font-size: 11px;
	}
	.repair-baseline b {
		font: 12px var(--mono);
	}
	.repair-table-wrap {
		overflow-x: auto;
	}
	table {
		width: 100%;
		border-collapse: collapse;
		font-size: 12px;
	}
	th {
		font: 9px var(--mono);
		text-align: left;
		text-transform: uppercase;
		color: var(--muted);
	}
	td,
	th {
		padding: 12px 15px;
		border-bottom: 1px solid var(--line);
	}
	td:first-child,
	th:first-child {
		padding-left: 0;
	}
	td:not(:first-child) {
		font-family: var(--mono);
	}
	.repair-caveat {
		font-size: 11px;
		line-height: 1.75;
		color: var(--muted);
		max-width: 900px;
	}
	details {
		font-size: 11px;
		line-height: 1.8;
		margin: 15px 0;
	}
	summary {
		cursor: pointer;
		color: var(--forest);
	}
	@media (max-width: 650px) {
		.repair-intro {
			flex-direction: column;
			align-items: flex-start;
			gap: 10px;
		}
		.repair-baseline {
			flex-direction: column;
		}
		td,
		th {
			padding: 10px 8px;
		}
		th {
			font-size: 8px;
		}
	}
	.session-bar {
		display: flex;
		justify-content: space-between;
		align-items: center;
		border-top: 1px solid var(--line);
		padding: 20px 0;
		gap: 20px;
		margin-top: 28px;
	}
	.session-bar > div {
		display: flex;
		align-items: center;
		gap: 17px;
	}
	.session-bar label {
		font-size: 11px;
		color: var(--muted);
	}
	.session-bar input {
		width: 66px;
		background: transparent;
		border: 1px solid var(--line);
		border-radius: 3px;
		padding: 7px;
		font: 11px var(--mono);
		margin-left: 6px;
	}
	.session-bar button {
		font-size: 10px;
	}
	.saved-indicator {
		font: 9px var(--mono);
		color: var(--muted);
	}
	footer {
		border-top: 1px solid var(--line);
		padding: 27px 0 30px;
		display: flex;
		align-items: center;
		gap: 22px;
		color: var(--muted);
		font-size: 11px;
	}
	.brand-footer {
		font-size: 20px;
		color: var(--ink);
		font-weight: 600;
		letter-spacing: -0.8px;
	}
	footer .eyebrow {
		margin-left: auto;
		font-size: 9px;
	}
	.error-banner,
	.storage-banner {
		border: 1px solid #d7ad99;
		padding: 15px 20px;
		margin-bottom: 20px;
		font-size: 12px;
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 15px;
		background: #f1e3d8;
	}
	.error-banner button {
		border: 0;
		background: none;
		text-decoration: underline;
		white-space: nowrap;
	}
	.file-input {
		display: none;
	}
	.journal-intro {
		padding: 48px 0 42px;
	}
	.journal-intro .eyebrow {
		color: var(--forest);
	}
	.journal-intro p {
		font-size: 14px;
		line-height: 1.7;
		color: var(--muted);
		margin-top: 22px;
	}
	.journal-columns {
		display: grid;
		grid-template-columns: 1.35fr 1fr;
		gap: 75px;
	}
	.notebook-entry {
		padding: 25px 0 30px;
		border-bottom: 1px solid var(--line);
	}
	.entry-meta {
		display: flex;
		align-items: center;
		gap: 12px;
		font: 9px var(--mono);
		color: var(--muted);
	}
	.entry-number {
		color: var(--forest);
	}
	.entry-kind {
		text-transform: uppercase;
		letter-spacing: 0.07em;
	}
	.entry-meta time {
		margin-left: auto;
	}
	.notebook-entry h2 {
		font-size: 30px;
	}
	.notebook-entry p {
		font-size: 13px;
		line-height: 1.8;
		color: var(--muted);
	}
	.entry-lesson {
		padding: 13px 17px;
		background: #e9ecdf;
		border-left: 2px solid #9aa78b;
		font-size: 12px;
		line-height: 1.7;
	}
	.entry-links {
		display: flex;
		gap: 15px;
		margin-top: 15px;
		font-size: 11px;
		color: var(--forest);
	}
	.reference-catalog {
		padding: 20px 0 5px;
	}
	.reference-catalog article {
		display: flex;
		gap: 15px;
		padding: 18px 0;
		border-bottom: 1px solid var(--line);
	}
	.reference-number {
		font: 28px var(--serif);
		color: var(--forest);
		padding-top: 4px;
	}
	.reference-catalog h3 {
		font: 25px var(--serif);
		margin: 0 0 8px;
	}
	.reference-catalog p {
		font-size: 11px;
		line-height: 1.7;
		color: var(--muted);
		margin: 0 0 11px;
	}
	.reference-catalog button {
		font-size: 11px;
	}
	.run-card {
		padding: 23px 20px 20px;
		margin-top: 18px;
		background: var(--surface);
		border: 1px solid var(--line);
		border-radius: 5px;
	}
	.run-card h3 {
		font: 25px var(--serif);
		margin: 14px 0;
	}
	.run-stats {
		display: flex;
		gap: 20px;
		font-size: 11px;
		margin-bottom: 16px;
		color: var(--muted);
	}
	.run-stats b {
		font: 13px var(--mono);
		color: var(--ink);
		margin-right: 5px;
	}
	.run-actions {
		display: flex;
		justify-content: space-between;
		font-size: 11px;
		margin-top: 14px;
	}
	.field-note {
		margin: 25px 0;
		display: grid;
		gap: 12px;
	}
	.field-note label {
		font: 25px var(--serif);
	}
	textarea {
		width: 100%;
		padding: 13px;
		border: 1px solid var(--line);
		border-radius: 4px;
		background: var(--surface);
		font-size: 12px;
		line-height: 1.6;
		resize: vertical;
	}
	.field-note button {
		justify-self: start;
		font-size: 11px;
	}
	.activity-log {
		margin: 45px 0;
	}
	.activity-log article {
		display: grid;
		grid-template-columns: 60px 1fr;
		padding: 24px 0;
		border-bottom: 1px solid var(--line);
	}
	.observation-marker {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 8px;
		padding-top: 3px;
	}
	.observation-marker > span:first-child {
		width: 7px;
		height: 7px;
		border-radius: 50%;
		background: var(--forest);
	}
	.activity-log h3 {
		font: 24px var(--serif);
		margin: 9px 0;
	}
	.activity-log p {
		font-size: 12px;
		line-height: 1.7;
		color: var(--muted);
		margin: 0;
	}
	.methods-grid {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 38px 75px;
		margin: 0 0 45px;
	}
	.methods-grid article {
		border-top: 1px solid var(--line);
		padding-top: 24px;
	}
	.methods-grid .eyebrow {
		color: var(--forest);
	}
	.methods-grid h2 {
		font-size: 32px;
	}
	.methods-grid p {
		font-size: 13px;
		line-height: 1.85;
		color: var(--muted);
	}
	.research-next {
		padding: 30px;
		border: 1px solid var(--line);
		background: #e9ecdf;
		margin-bottom: 45px;
		max-width: 850px;
	}
	.research-next h2 {
		font-size: 32px;
	}
	.research-next p {
		font-size: 13px;
		line-height: 1.8;
		color: var(--muted);
	}
	@keyframes breathe {
		50% {
			opacity: 0.35;
		}
	}
	@media (min-width: 1400px) {
		.field-wrap {
			height: 467px;
		}
	}
	@media (max-width: 1150px) {
		.lab-shell {
			padding: 0 30px;
		}
		.masthead-note {
			display: none;
		}
		.github-link {
			margin-left: 0;
		}
		.workbench {
			grid-template-columns: minmax(0, 1fr) 315px;
		}
		.instruments {
			padding: 20px;
		}
		.projection-strip {
			gap: 20px;
		}
		.projection-strip p {
			display: none;
		}
		.intro-copy {
			width: 285px;
		}
		.specimen-toolbar {
			padding: 0 16px;
		}
		.specimen-toolbar .eyebrow {
			font-size: 9px;
		}
		.evidence-row {
			gap: 25px;
		}
		.curve-legend {
			font-size: 8px;
		}
		.session-bar > div {
			gap: 12px;
		}
		.session-bar > div > .eyebrow {
			display: none;
		}
		.journal-columns {
			gap: 40px;
		}
	}
	@media (max-width: 850px) {
		.lab-shell {
			padding: 0 22px;
		}
		.masthead {
			height: 83px;
			gap: 15px;
		}
		nav {
			gap: 20px;
		}
		.github-link {
			display: none;
		}
		.intro {
			padding-top: 31px;
		}
		h1 {
			font-size: 56px;
		}
		.intro-copy {
			width: auto;
			max-width: 250px;
		}
		.intro-copy p {
			font-size: 12px;
		}
		.workbench {
			grid-template-columns: 1fr;
		}
		.field-wrap {
			height: 430px;
		}
		.instruments {
			border-left: 0;
			border-top: 1px solid var(--line);
			padding: 25px;
			display: block;
		}
		.instrument-description {
			max-width: 360px;
		}
		.primary-metric {
			max-width: 370px;
		}
		.metric-row {
			max-width: 370px;
		}
		.training-controls {
			max-width: 370px;
		}
		.probabilities {
			max-width: 480px;
		}
		.prob-track {
			height: 60px;
		}
		.model-specs {
			margin-bottom: 16px;
		}
		.evidence-row {
			grid-template-columns: 1fr;
			gap: 30px;
		}
		.session-bar {
			flex-wrap: wrap;
		}
		.journal-columns {
			grid-template-columns: 1fr;
			gap: 40px;
		}
		.methods-grid {
			gap: 30px;
		}
		footer .eyebrow {
			display: none;
		}
		.projection-strip p {
			display: block;
		}
	}
	@media (max-width: 540px) {
		.lab-shell {
			padding: 0 17px;
		}
		.masthead {
			height: 74px;
			gap: 14px;
		}
		.brand > span {
			font-size: 26px;
		}
		.brand svg {
			width: 26px;
			height: 26px;
		}
		.brand {
			gap: 6px;
		}
		nav {
			gap: 14px;
		}
		nav button {
			font-size: 11px;
		}
		.nav-count {
			display: none;
		}
		.intro {
			align-items: flex-start;
			gap: 10px;
			padding: 27px 0;
		}
		.intro h1 {
			font-size: 48px;
		}
		.intro-label {
			font-size: 8px;
			letter-spacing: 0.05em;
		}
		.intro-copy {
			max-width: 110px;
			padding-top: 33px;
		}
		.intro-copy p {
			font-size: 10px;
			line-height: 1.6;
			margin-bottom: 14px;
		}
		.desktop-break {
			display: none;
		}
		.live-status {
			font-size: 8px;
			gap: 5px;
			letter-spacing: 0;
		}
		.specimen-toolbar {
			padding: 0 11px;
			gap: 8px;
		}
		.specimen-toolbar .eyebrow {
			font-size: 8px;
			letter-spacing: 0;
		}
		.specimen-toolbar .subtle {
			display: none;
		}
		.map-toggle button {
			padding: 6px;
			font-size: 10px;
		}
		.field-wrap {
			height: 400px;
		}
		.field-scale {
			left: 16px;
			bottom: 16px;
			font-size: 9px;
		}
		.projection-strip {
			padding: 17px;
			gap: 24px;
		}
		.projection-strip p {
			display: none;
		}
		.projection-strip strong {
			font-size: 19px;
		}
		.projection-strip .eyebrow {
			font-size: 8px;
		}
		.checkpoint-strip {
			padding: 17px;
		}
		.checkpoint-heading .subtle {
			display: none;
		}
		.instruments {
			padding: 22px;
		}
		.section-heading > .eyebrow {
			font-size: 9px;
		}
		.curve-legend {
			font-size: 8px;
			gap: 3px;
		}
		.curve-legend i {
			margin-left: 2px;
			width: 8px;
		}
		.session-bar {
			gap: 17px;
		}
		.saved-indicator {
			display: none;
		}
		footer {
			padding: 22px 0;
			gap: 15px;
		}
		footer > span:nth-child(2) {
			font-size: 9px;
		}
		.journal-intro {
			padding: 35px 0;
		}
		.journal-intro h1 {
			font-size: 48px;
		}
		.journal-intro p {
			font-size: 12px;
		}
		.methods-grid {
			grid-template-columns: 1fr;
		}
		.research-next {
			padding: 22px;
		}
		.error-banner {
			flex-direction: column;
			align-items: flex-start;
		}
	}
</style>
