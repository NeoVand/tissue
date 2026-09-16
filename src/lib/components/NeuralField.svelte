<script lang="ts">
	import { createNeuralField, type FieldPoint } from '$lib/scene/neural-field';

	interface Props {
		points: FieldPoint[];
		edges: Array<[number, number]>;
		selected: number | null;
		onselect: (id: number) => void;
		mode: 'activation' | 'effect';
		loading?: boolean;
	}

	let { points, edges, selected, onselect, mode, loading = false }: Props = $props();
	let rotating = $state(false);
	let failure = $state('');
	let hover = $state<{ point: FieldPoint; x: number; y: number } | null>(null);
	let field: ReturnType<typeof createNeuralField> | undefined;

	function attachField(canvas: HTMLCanvasElement) {
		try {
			const view = createNeuralField(canvas, {
				onselect: (id) => onselect(id),
				onhover: (point, x, y) => {
					hover = point ? { point, x, y } : null;
				},
				onerror: (message) => {
					failure = message;
				}
			});
			field = view;
			return () => {
				view.destroy();
				field = undefined;
			};
		} catch {
			failure =
				'This browser could not start the 3D view. The measurements and neuron inspector remain available.';
		}
	}

	// A second attachment tracks measurements without rebuilding the scene or camera.
	function updateField() {
		field?.update({ points, edges, selected, mode });
	}

	function toggleRotation() {
		rotating = !rotating;
		field?.rotate(rotating);
	}
</script>

<div class="neural-field" aria-busy={loading}>
	<canvas
		{@attach attachField}
		{@attach updateField}
		aria-label="Interactive three-dimensional map of measured neuron fingerprints. Drag to orbit, scroll to zoom, or select a neuron using the inspector."
	></canvas>

	<div class="field-heading" aria-hidden="true">
		<span class="field-cross">+</span>
		<span>FUNCTIONAL SPACE <span class="dimension">/ 3D</span></span>
	</div>
	<div class="field-count" aria-live="polite">
		<span class={['status-dot', { loading }]}></span>
		{loading ? 'Measuring anatomy' : `${points.length.toString().padStart(3, '0')} observed units`}
	</div>

	{#if failure}
		<div class="empty-field" role="status">
			<span class="empty-symbol">↗</span>
			<p>Graphics unavailable</p>
			<small>{failure}</small>
		</div>
	{:else if !points.length}
		<div class="empty-field" role="status">
			<span class="empty-symbol">⊙</span>
			<p>{loading ? 'Looking for a shape.' : 'A shape waiting to emerge.'}</p>
			<small
				>{loading
					? 'Measuring the model’s responses to the probe set.'
					: 'Initialize a model to reveal its measured anatomy.'}</small
			>
		</div>
	{/if}

	{#if hover && !failure}
		<div class="node-label" style:left="{hover.x}px" style:top="{hover.y}px">
			<span>N{hover.point.id.toString().padStart(3, '0')}</span>
			<strong
				>{mode === 'effect'
					? (hover.point.effect ?? 0).toFixed(4)
					: hover.point.activation.toFixed(3)}</strong
			>
			<small>{mode === 'effect' ? 'effect magnitude' : 'activation'}</small>
		</div>
	{/if}

	<div class="view-caption" aria-hidden="true">
		<span class="axis-symbol">↗</span>
		<div>
			<span>Drag to orbit · scroll to zoom</span><small
				>Size · compressed |{mode === 'effect' ? 'effect' : 'activation'}|</small
			>
		</div>
	</div>
	<div class="view-actions">
		<button
			class="rotate-button"
			onclick={toggleRotation}
			aria-pressed={rotating}
			aria-label={rotating ? 'Pause camera orbit' : 'Start camera orbit'}
			disabled={!!failure}
			title="Toggle slow camera rotation"
		>
			<svg viewBox="0 0 20 20" fill="none" aria-hidden="true"
				><path d="M15.9 7a6.5 6.5 0 1 0 .6 4M16 3v4h-4" /></svg
			>
			<span>{rotating ? 'Pause orbit' : 'Orbit'}</span>
		</button>
		<button
			onclick={() => field?.reset()}
			disabled={!!failure}
			title="Reset camera"
			aria-label="Reset camera"
		>
			<svg viewBox="0 0 20 20" fill="none" aria-hidden="true"
				><path d="M7 3H3v4m10-4h4v4M3 13v4h4m10-4v4h-4" /><path d="M7 10h6m-3-3v6" /></svg
			>
			<span>Reset view</span>
		</button>
	</div>
</div>

<style>
	.neural-field {
		position: relative;
		width: 100%;
		height: 100%;
		min-height: 400px;
		overflow: hidden;
		background: radial-gradient(ellipse at 48% 41%, #fffdf7 0%, #f3f1e8 77%);
		color: #58604e;
	}
	canvas {
		display: block;
		width: 100%;
		height: 100%;
		min-height: 400px;
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
		top: 24px;
		font: 10px/1.4 var(--font-mono, 'IBM Plex Mono', monospace);
		letter-spacing: 0.085em;
		pointer-events: none;
	}
	.field-heading {
		left: 26px;
		display: flex;
		align-items: center;
		gap: 13px;
	}
	.field-cross {
		font-size: 21px;
		line-height: 0.5;
		color: #919786;
		font-weight: 300;
	}
	.dimension {
		color: #8b9082;
	}
	.field-count {
		right: 26px;
		display: flex;
		align-items: center;
		gap: 8px;
		letter-spacing: 0.035em;
	}
	.status-dot {
		height: 5px;
		width: 5px;
		border-radius: 50%;
		background: #73876b;
	}
	.status-dot.loading {
		background: #bd7e3d;
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
		padding: 70px 25px;
	}
	.empty-symbol {
		color: #98a287;
		font-family: Georgia, serif;
		font-size: 45px;
		font-weight: 300;
	}
	.empty-field p {
		font: italic 27px/1.2 var(--font-serif, Georgia, serif);
		margin: 14px 0 12px;
		color: #4d5947;
	}
	.empty-field small {
		max-width: 320px;
		color: #7d8274;
		font-size: 12px;
		line-height: 1.6;
	}
	.view-caption {
		position: absolute;
		left: 25px;
		bottom: 24px;
		display: flex;
		align-items: center;
		gap: 12px;
		pointer-events: none;
	}
	.axis-symbol {
		font-family: Georgia, serif;
		font-size: 27px;
		font-weight: 300;
		color: #7c8970;
	}
	.view-caption div {
		display: flex;
		flex-direction: column;
		gap: 5px;
	}
	.view-caption span:not(.axis-symbol) {
		font-size: 10px;
	}
	.view-caption small {
		font-size: 9px;
		color: #909584;
	}
	.view-actions {
		position: absolute;
		bottom: 24px;
		right: 25px;
		display: flex;
		gap: 6px;
	}
	button {
		display: flex;
		gap: 7px;
		align-items: center;
		justify-content: center;
		border: 1px solid #d4d7c9;
		background: #fcfbf4d9;
		color: #606953;
		min-height: 33px;
		padding: 0 10px;
		border-radius: 3px;
		font-family: inherit;
		font-size: 10px;
		line-height: 1.2;
		cursor: pointer;
		transition:
			background 120ms,
			border-color 120ms;
	}
	button:hover {
		border-color: #939d83;
		background: #fffef8;
	}
	button[aria-pressed='true'] {
		background: #e5eac9;
		border-color: #a3b17c;
	}
	button:focus-visible {
		outline: 2px solid #5d7448;
		outline-offset: 3px;
	}
	button:disabled {
		opacity: 0.5;
		cursor: default;
	}
	button svg {
		width: 15px;
		height: 15px;
		stroke: currentColor;
		stroke-width: 1.1;
	}
	.node-label {
		position: absolute;
		pointer-events: none;
		display: flex;
		flex-wrap: wrap;
		width: 131px;
		gap: 9px;
		transform: translate(-50%, calc(-100% - 14px));
		background: #fffef6f2;
		border: 1px solid #d4d7c9;
		box-shadow: 0 5px 20px #39432b0a;
		padding: 10px 12px;
		border-radius: 3px;
		color: #47553b;
		font: 10px/1.3 var(--font-mono, monospace);
	}
	.node-label strong {
		margin-left: auto;
		font-weight: 500;
	}
	.node-label small {
		width: 100%;
		font-size: 9px;
		color: #879075;
	}
	@media (max-width: 600px) {
		.field-heading,
		.field-count {
			top: 18px;
			font-size: 8px;
		}
		.field-heading {
			left: 16px;
			gap: 8px;
		}
		.field-count {
			right: 16px;
		}
		.view-caption {
			left: 16px;
			bottom: 18px;
		}
		.view-actions {
			right: 16px;
			bottom: 18px;
		}
		.view-actions button {
			width: 32px;
			padding: 0;
		}
		.view-actions button span {
			display: none;
		}
		.view-caption span:not(.axis-symbol) {
			font-size: 9px;
		}
	}
	@media (prefers-reduced-motion: reduce) {
		button {
			transition: none;
		}
	}
</style>
