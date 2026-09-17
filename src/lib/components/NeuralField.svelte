<script lang="ts">
	import Icon from '$lib/components/Icon.svelte';
	import { createNeuralField, type FieldPoint, type FieldStats } from '$lib/scene/neural-field';

	interface Props {
		points: FieldPoint[];
		edges: Array<[number, number]>;
		selected: number | null;
		onselect: (id: number) => void;
		mode: 'activation' | 'effect';
		loading?: boolean;
		theme?: 'dark' | 'light';
		layerFilter?: number | null;
		geometryLabel?: string;
		onstats?: (stats: FieldStats) => void;
	}
	let {
		points,
		edges,
		selected,
		onselect,
		mode,
		loading = false,
		theme = 'dark',
		layerFilter = null,
		geometryLabel = 'Functional geometry',
		onstats
	}: Props = $props();
	let rotating = $state(false);
	let failure = $state('');
	let hover = $state.raw<{ point: FieldPoint; x: number; y: number } | null>(null);
	let anchor = $state.raw<{ id: number; layer: number; x: number; y: number } | null>(null);
	let field: ReturnType<typeof createNeuralField> | undefined;
	const visibleCount = $derived(
		points.filter((point) => layerFilter === null || point.layer === layerFilter).length
	);

	function attachField(canvas: HTMLCanvasElement) {
		try {
			const view = createNeuralField(canvas, {
				onselect: (id) => onselect(id),
				onhover: (point, x, y) => {
					hover = point ? { point, x, y } : null;
				},
				onanchor: (value) => {
					anchor = value;
				},
				onerror: (message) => {
					failure = message;
				},
				onstats: (stats) => onstats?.(stats)
			});
			field = view;
			return () => {
				view.destroy();
				field = undefined;
			};
		} catch {
			failure =
				'This browser could not start the 3D view. Measurements remain available in the inspector and journal.';
		}
	}
	// Measurements update independently of scene lifecycle, preserving camera and interrupted motion.
	function updateField() {
		field?.update({ points, edges, selected, mode, theme, layerFilter });
	}
	function toggleRotation() {
		rotating = !rotating;
		field?.rotate(rotating);
	}
</script>

<div class="neural-field" data-theme={theme} aria-busy={loading}>
	<canvas
		{@attach attachField}
		{@attach updateField}
		aria-label={`${geometryLabel}. Three-dimensional view of measured neurons. Drag to orbit, scroll to zoom, or select a neuron with the inspector. Lines indicate similarity, not causal connections.`}
	></canvas>
	<div class="field-heading" aria-hidden="true">
		<Icon name="cube" size={13} />
		<span>{geometryLabel}</span><span class="dimension">3D</span>
	</div>
	<div class="field-count" aria-live="polite">
		<span class={['status-dot', { loading }]}></span>
		{loading ? 'Measuring' : `${visibleCount} units`}
	</div>
	{#if failure}
		<div class="empty-field" role="status">
			<Icon name="cube" size={24} />
			<p>Graphics unavailable</p>
			<small>{failure}</small>
		</div>
	{:else if !points.length}
		<div class="empty-field" role="status">
			<Icon name="cube" size={27} />
			<p>{loading ? 'Measuring the model' : 'No anatomy recorded'}</p>
			<small
				>{loading
					? 'Building fingerprints from the fixed calibration prompts.'
					: 'Initialize a model or open a reference experiment.'}</small
			>
		</div>
	{/if}
	{#if anchor && !hover && !failure}
		<div class="selected-label" style:left="{anchor.x}px" style:top="{anchor.y}px">
			<span class="label-leader"></span><span
				>L{anchor.layer + 1} <b>U{anchor.id.toString().padStart(3, '0')}</b></span
			>
		</div>
	{/if}
	{#if hover && !failure}
		<div class="node-label" style:left="{hover.x}px" style:top="{hover.y}px">
			<div>
				<span>L{hover.point.layer + 1} · U{hover.point.id.toString().padStart(3, '0')}</span><strong
					>{mode === 'effect'
						? (hover.point.effect ?? 0).toFixed(4)
						: hover.point.activation.toFixed(3)}</strong
				>
			</div>
			<small>{mode === 'effect' ? 'Effect RMS' : 'Token activation'} · click to inspect</small>
		</div>
	{/if}
	<div class="view-caption" aria-hidden="true">
		<span>{edges.length ? 'Similarity links · not causal' : 'Measured units'}</span>
		<small
			>Size · compressed |{mode === 'effect' ? 'effect' : 'activation'}|
			<span class="gesture-hint"> / Drag to orbit</span></small
		>
	</div>
	<div class="view-actions">
		<button
			onclick={toggleRotation}
			aria-pressed={rotating}
			aria-label={rotating ? 'Pause camera orbit' : 'Start camera orbit'}
			disabled={!!failure}
			title={rotating ? 'Pause orbit' : 'Slow orbit'}
		>
			<Icon name={rotating ? 'pause' : 'orbit'} size={14} /><span
				>{rotating ? 'Pause' : 'Orbit'}</span
			>
		</button>
		<button
			onclick={() => field?.reset()}
			disabled={!!failure}
			title="Frame all units"
			aria-label="Reset camera"
		>
			<Icon name="reset" size={14} /><span>Frame</span>
		</button>
	</div>
</div>

<style>
	.neural-field {
		--field-ink: #c7d0cb;
		--field-muted: #81918a;
		--field-line: #344139;
		--field-panel: #19211ee8;
		--field-accent: #bdcec0;
		position: relative;
		width: 100%;
		height: 100%;
		min-height: 220px;
		overflow: hidden;
		background: radial-gradient(ellipse at 47% 42%, #1a211e 0%, #121816 66%, #111614 100%);
		color: var(--field-ink);
		isolation: isolate;
	}
	.neural-field[data-theme='light'] {
		--field-ink: #3e5148;
		--field-muted: #758379;
		--field-line: #c4d0c6;
		--field-panel: #f8faf5ed;
		--field-accent: #526e5a;
		background: radial-gradient(ellipse at 48% 42%, #fcfdf8 0%, #eff2eb 75%, #e9eee7 100%);
	}
	canvas {
		display: block;
		width: 100%;
		height: 100%;
		position: absolute;
		inset: 0;
		touch-action: none;
		cursor: grab;
	}
	canvas:active {
		cursor: grabbing;
	}
	.field-heading,
	.field-count {
		position: absolute;
		top: 15px;
		display: flex;
		align-items: center;
		gap: 8px;
		font: 10px/1.2 var(--mono, 'IBM Plex Mono', monospace);
		pointer-events: none;
	}
	.field-heading {
		left: 17px;
		max-width: calc(100% - 130px);
		color: var(--field-muted);
	}
	.field-heading > span:first-of-type {
		overflow: hidden;
		white-space: nowrap;
		text-overflow: ellipsis;
	}
	.dimension {
		color: var(--field-muted);
		opacity: 0.6;
		font-size: 9px;
		border-left: 1px solid var(--field-line);
		padding-left: 8px;
	}
	.field-count {
		right: 17px;
		color: var(--field-muted);
		font-size: 9px;
	}
	.status-dot {
		height: 4px;
		width: 4px;
		border-radius: 50%;
		background: #96ab9b;
	}
	.status-dot.loading {
		background: #c4b38f;
	}
	.empty-field {
		position: absolute;
		inset: 0;
		display: flex;
		align-items: center;
		justify-content: center;
		flex-direction: column;
		text-align: center;
		pointer-events: none;
		padding: 60px 25px;
		color: var(--field-muted);
	}
	.empty-field p {
		font: 500 13px/1.4 var(--sans, sans-serif);
		margin: 15px 0 6px;
		color: var(--field-ink);
	}
	.empty-field small {
		max-width: 260px;
		font-size: 11px;
		line-height: 1.6;
	}
	.view-caption {
		position: absolute;
		left: 17px;
		bottom: 15px;
		display: flex;
		flex-direction: column;
		gap: 5px;
		pointer-events: none;
		color: var(--field-muted);
	}
	.view-caption > span {
		font-size: 10px;
	}
	.view-caption small {
		font: 9px/1.2 var(--mono, monospace);
		opacity: 0.75;
	}
	.view-actions {
		position: absolute;
		bottom: 15px;
		right: 17px;
		display: flex;
		gap: 5px;
	}
	button {
		display: flex;
		gap: 6px;
		align-items: center;
		justify-content: center;
		border: 1px solid var(--field-line);
		background: var(--field-panel);
		color: var(--field-muted);
		min-height: 29px;
		padding: 0 8px;
		border-radius: 5px;
		font-family: inherit;
		font-size: 10px;
		line-height: 1.2;
		cursor: pointer;
		transition:
			color 120ms,
			border-color 120ms;
	}
	button:hover,
	button[aria-pressed='true'] {
		color: var(--field-ink);
		border-color: var(--field-accent);
	}
	button:focus-visible {
		outline: 2px solid var(--field-accent);
		outline-offset: 3px;
	}
	button:disabled {
		opacity: 0.4;
		cursor: default;
	}
	.node-label {
		position: absolute;
		pointer-events: none;
		width: 150px;
		transform: translate(-50%, calc(-100% - 13px));
		background: var(--field-panel);
		border: 1px solid var(--field-line);
		backdrop-filter: blur(12px);
		padding: 9px 10px;
		border-radius: 5px;
		color: var(--field-ink);
		font: 10px/1.4 var(--mono, monospace);
	}
	.node-label > div {
		display: flex;
		gap: 8px;
		justify-content: space-between;
	}
	.node-label strong {
		font-weight: 500;
	}
	.node-label small {
		display: block;
		margin-top: 4px;
		font-size: 8px;
		color: var(--field-muted);
	}
	.selected-label {
		position: absolute;
		display: flex;
		align-items: center;
		pointer-events: none;
		transform: translate(12px, -50%);
		white-space: nowrap;
		font: 9px/1 var(--mono, monospace);
		color: var(--field-muted);
	}
	.selected-label b {
		color: var(--field-ink);
		margin-left: 4px;
		font-weight: 500;
	}
	.selected-label > span:last-child {
		background: var(--field-panel);
		border: 1px solid var(--field-line);
		padding: 5px 6px;
		border-radius: 3px;
	}
	.label-leader {
		height: 1px;
		width: 16px;
		background: var(--field-line);
	}
	@media (max-width: 520px) {
		.view-actions button {
			width: 30px;
			padding: 0;
		}
		.view-actions button span,
		.gesture-hint {
			display: none;
		}
		.field-heading {
			left: 12px;
			gap: 5px;
			font-size: 9px;
		}
		.field-count {
			right: 12px;
		}
		.view-caption {
			left: 12px;
		}
		.view-actions {
			right: 12px;
		}
	}
	@media (prefers-reduced-motion: reduce) {
		button {
			transition: none;
		}
	}
</style>
