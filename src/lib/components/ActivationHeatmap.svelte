<script lang="ts">
	import { MODEL_CONFIG, type Probe } from '$lib/lab/protocol';

	interface Props {
		probe: Probe | null | undefined;
		selected: number | null;
		token: number;
		onselect: (id: number) => void;
		ontoken: (position: number) => void;
		theme?: 'dark' | 'light';
		layerFilter?: number | null;
	}

	let {
		probe,
		selected,
		token,
		onselect,
		ontoken,
		theme = 'dark',
		layerFilter = null
	}: Props = $props();
	const CHANNELS = MODEL_CONFIG.hidden;
	const TOKENS = MODEL_CONFIG.sequenceLength;
	const HEIGHT = 178;
	const LEFT = 43;
	const RIGHT = 10;
	const TOP = 22;
	const BOTTOM = 16;
	const GAP = 14;
	const SCALE = 2;

	interface Cell {
		id: number;
		token: number;
		x?: number;
		y?: number;
	}
	let hover = $state<Cell | null>(null);
	let graphicsUnavailable = $state(false);
	let layers = $derived(
		layerFilter !== null &&
			Number.isInteger(layerFilter) &&
			layerFilter >= 0 &&
			layerFilter < MODEL_CONFIG.layers
			? [layerFilter]
			: Array.from({ length: MODEL_CONFIG.layers }, (_, layer) => layer)
	);
	let hasProbe = $derived(!!probe && probe.activations.length === TOKENS);
	let selectedCell = $derived(selected === null ? null : { id: selected, token });
	let inspected = $derived(hover ?? selectedCell);
	let inspectedValue = $derived(inspected ? valueAt(inspected.id, inspected.token) : null);
	let selectedDescription = $derived(selectedCell ? describe(selectedCell) : 'No neuron selected.');

	function valueAt(id: number, position: number): number | null {
		const value = probe?.activations[position]?.[id];
		return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null;
	}
	function describe(cell: Cell): string {
		const value = valueAt(cell.id, cell.token);
		return `Layer ${Math.floor(cell.id / CHANNELS) + 1}, channel ${cell.id % CHANNELS}, token ${cell.token + 1} ${probe?.example.tokens[cell.token] ?? ''}. Activation ${value === null ? 'unavailable' : String(value)}.`;
	}
	function dimensions(width: number) {
		const layerWidth = Math.max(
			1,
			(width - LEFT - RIGHT - GAP * (layers.length - 1)) / layers.length
		);
		return {
			layerWidth,
			cellWidth: layerWidth / CHANNELS,
			cellHeight: (HEIGHT - TOP - BOTTOM) / TOKENS
		};
	}
	function locate(event: MouseEvent | PointerEvent): Cell | null {
		if (!hasProbe) return null;
		const bounds = (event.currentTarget as HTMLButtonElement).getBoundingClientRect();
		const x = event.clientX - bounds.left;
		const y = event.clientY - bounds.top;
		const { layerWidth, cellWidth, cellHeight } = dimensions(bounds.width);
		const position = Math.floor((y - TOP) / cellHeight);
		const strip = Math.floor((x - LEFT) / (layerWidth + GAP));
		const channel = Math.floor((x - LEFT - strip * (layerWidth + GAP)) / cellWidth);
		if (
			position < 0 ||
			position >= TOKENS ||
			strip < 0 ||
			strip >= layers.length ||
			channel < 0 ||
			channel >= CHANNELS
		)
			return null;
		return { id: layers[strip] * CHANNELS + channel, token: position, x, y };
	}
	function choose(cell: Cell) {
		onselect(cell.id);
		ontoken(cell.token);
	}
	function selectPointer(event: MouseEvent) {
		const cell = locate(event);
		if (cell) choose(cell);
	}
	function navigate(event: KeyboardEvent) {
		if (!hasProbe) return;
		const keys = ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End', 'Enter', ' '];
		if (!keys.includes(event.key)) return;
		event.preventDefault();
		const first = layers[0] * CHANNELS;
		const last = (layers.at(-1)! + 1) * CHANNELS - 1;
		let id = selected !== null && selected >= first && selected <= last ? selected : first;
		let position = Math.min(TOKENS - 1, Math.max(0, token));
		if (event.key === 'ArrowLeft') id = Math.max(first, id - 1);
		if (event.key === 'ArrowRight') id = Math.min(last, id + 1);
		if (event.key === 'ArrowUp') position = Math.max(0, position - 1);
		if (event.key === 'ArrowDown') position = Math.min(TOKENS - 1, position + 1);
		if (event.key === 'Home') id = first;
		if (event.key === 'End') id = last;
		hover = null;
		choose({ id, token: position });
	}

	function attachCanvas(canvas: HTMLCanvasElement) {
		const context = canvas.getContext('2d');
		if (!context) {
			graphicsUnavailable = true;
			return;
		}
		const base = document.createElement('canvas');
		const paint = base.getContext('2d')!;
		let cachedProbe: Probe | null | undefined;
		let cachedTheme = '';
		let cachedLayers = '';
		let cachedWidth = 0;
		let cachedRatio = 0;
		let accent = '#b6d88e';
		let muted = '#7f8994';
		let ink = '#ecf0ef';

		function draw() {
			if (!context) return;
			const width = canvas.getBoundingClientRect().width;
			if (!width) return;
			const ratio = Math.min(window.devicePixelRatio || 1, 2);
			const { layerWidth, cellWidth, cellHeight } = dimensions(width);
			const layerKey = layers.join(',');
			if (
				cachedProbe !== probe ||
				cachedTheme !== theme ||
				cachedLayers !== layerKey ||
				cachedWidth !== width ||
				cachedRatio !== ratio
			) {
				canvas.width = base.width = Math.round(width * ratio);
				canvas.height = base.height = Math.round(HEIGHT * ratio);
				paint.setTransform(ratio, 0, 0, ratio, 0, 0);
				const style = getComputedStyle(canvas);
				const color = (name: string, fallback: string) =>
					style.getPropertyValue(name).trim() || fallback;
				const background = color('--surface', theme === 'dark' ? '#13171a' : '#f6f7f4');
				const raised = color('--surface-raised', theme === 'dark' ? '#20272a' : '#e8ece7');
				const line = color('--line', theme === 'dark' ? '#293236' : '#d4dad3');
				accent = color('--accent', theme === 'dark' ? '#b6d88e' : '#446d38');
				muted = color('--muted', theme === 'dark' ? '#7f8994' : '#667061');
				ink = color('--ink', theme === 'dark' ? '#ecf0ef' : '#202b22');
				const layerColors = [color('--layer-1', '#d69970'), color('--layer-2', '#88bda9')];
				paint.globalAlpha = 1;
				paint.fillStyle = background;
				paint.fillRect(0, 0, width, HEIGHT);
				paint.font = '9px ui-monospace, SFMono-Regular, Menlo, monospace';
				paint.textBaseline = 'middle';
				paint.textAlign = 'left';
				paint.fillStyle = muted;
				paint.fillText('TOKEN', 4, 10);
				for (let position = 0; position < TOKENS; position++) {
					const y = TOP + position * cellHeight;
					paint.fillStyle = muted;
					paint.fillText(String(position + 1).padStart(2, '0'), 4, y + cellHeight / 2);
					paint.fillStyle = ink;
					paint.fillText(probe?.example.tokens[position] ?? '·', 25, y + cellHeight / 2);
				}
				for (const [strip, layer] of layers.entries()) {
					const x = LEFT + strip * (layerWidth + GAP);
					paint.globalAlpha = 1;
					paint.fillStyle = layerColors[layer];
					paint.fillText(`LAYER ${layer + 1}`, x, 10);
					paint.fillStyle = muted;
					paint.textAlign = 'right';
					paint.fillText('128 MLP UNITS', x + layerWidth, 10);
					paint.textAlign = 'left';
					paint.fillStyle = raised;
					paint.fillRect(x, TOP, layerWidth, HEIGHT - TOP - BOTTOM);
					for (let position = 0; position < TOKENS; position++) {
						const y = TOP + position * cellHeight;
						for (let channel = 0; channel < CHANNELS; channel++) {
							const value = valueAt(layer * CHANNELS + channel, position);
							if (value === null) continue;
							paint.fillStyle = layerColors[layer];
							paint.globalAlpha = Math.tanh(value / SCALE);
							paint.fillRect(
								x + channel * cellWidth,
								y + 0.5,
								Math.max(0.5, cellWidth),
								cellHeight - 1
							);
						}
					}
					paint.globalAlpha = 1;
					paint.strokeStyle = line;
					paint.lineWidth = 1;
					paint.strokeRect(x + 0.5, TOP + 0.5, layerWidth - 1, HEIGHT - TOP - BOTTOM - 1);
					paint.fillStyle = muted;
					for (const channel of [0, 32, 64, 96, 127]) {
						paint.textAlign = channel === 0 ? 'left' : channel === 127 ? 'right' : 'center';
						paint.fillText(String(channel), x + channel * cellWidth + cellWidth / 2, HEIGHT - 5);
					}
					paint.textAlign = 'left';
				}
				cachedProbe = probe;
				cachedTheme = theme;
				cachedLayers = layerKey;
				cachedWidth = width;
				cachedRatio = ratio;
			}
			context.setTransform(ratio, 0, 0, ratio, 0, 0);
			context.clearRect(0, 0, width, HEIGHT);
			context.drawImage(base, 0, 0, width, HEIGHT);
			if (!hasProbe) return;
			const position = Math.min(TOKENS - 1, Math.max(0, token));
			context.strokeStyle = accent;
			context.lineWidth = 1;
			context.globalAlpha = 0.75;
			for (let strip = 0; strip < layers.length; strip++) {
				const x = LEFT + strip * (layerWidth + GAP);
				context.strokeRect(
					x + 0.5,
					TOP + position * cellHeight + 0.5,
					layerWidth - 1,
					cellHeight - 1
				);
			}
			if (selected !== null) {
				const strip = layers.indexOf(Math.floor(selected / CHANNELS));
				if (strip >= 0) {
					const x = LEFT + strip * (layerWidth + GAP) + (selected % CHANNELS) * cellWidth;
					context.strokeRect(
						x + 0.5,
						TOP + 0.5,
						Math.max(1, cellWidth - 1),
						HEIGHT - TOP - BOTTOM - 1
					);
					context.globalAlpha = 1;
					context.strokeStyle = ink;
					context.strokeRect(
						x - 0.5,
						TOP + position * cellHeight - 0.5,
						cellWidth + 1,
						cellHeight + 1
					);
				}
			}
			if (hover) {
				const strip = layers.indexOf(Math.floor(hover.id / CHANNELS));
				if (strip >= 0) {
					context.globalAlpha = 1;
					context.strokeStyle = ink;
					context.strokeRect(
						LEFT + strip * (layerWidth + GAP) + (hover.id % CHANNELS) * cellWidth - 0.5,
						TOP + hover.token * cellHeight - 0.5,
						cellWidth + 1,
						cellHeight + 1
					);
				}
			}
			context.globalAlpha = 1;
		}

		const resize = new ResizeObserver(draw);
		resize.observe(canvas);
		$effect(draw);
		return () => resize.disconnect();
	}
</script>

<section
	class="activation-matrix"
	data-theme={theme}
	aria-label="Token by neuron activation heatmap"
>
	<div class="matrix-heading">
		<div><strong>Activation strips</strong><span>POST-ReLU · TOKEN × CHANNEL</span></div>
		<div
			class="legend"
			title="Fixed intensity: tanh(activation / 2). The same magnitude has the same intensity across prompts and checkpoints."
		>
			{#each [0, 2, 4] as value (value)}
				<span class="legend-chip"><i style:opacity={Math.tanh(value / SCALE)}></i></span><span
					>{value}{value === 4 ? '+' : ''}</span
				>
			{/each}
		</div>
	</div>
	<div class="matrix-body">
		<button
			type="button"
			class="matrix-surface"
			aria-label="Inspect neuron activations. Left and right arrows select units; up and down arrows select tokens. Home and End select the first and last visible unit."
			disabled={!hasProbe}
			onclick={selectPointer}
			onpointermove={(event) => {
				hover = locate(event);
			}}
			onpointerleave={() => {
				hover = null;
			}}
			onkeydown={navigate}
		>
			<canvas {@attach attachCanvas} aria-hidden="true"></canvas>
		</button>
		{#if !hasProbe || graphicsUnavailable}
			<div class="matrix-empty">
				{graphicsUnavailable
					? 'Canvas unavailable. Use the neuron inspector for measured values.'
					: 'Activation measurements appear after a probe is captured.'}
			</div>
		{/if}
	</div>
	<div class="matrix-footer">
		{#if inspected}
			<div class="cell-readout">
				<span class="cell-kind">{hover ? 'HOVER' : 'SELECTED'}</span><span
					>L{Math.floor(inspected.id / CHANNELS) + 1} / {String(inspected.id % CHANNELS).padStart(
						3,
						'0'
					)}</span
				><span>T{inspected.token + 1} <b>{probe?.example.tokens[inspected.token] ?? '—'}</b></span
				><strong title="Unnormalized recorded activation"
					>{inspectedValue === null ? 'unavailable' : String(inspectedValue)}</strong
				>
			</div>
		{:else}<span class="matrix-hint">Select a cell to link its neuron and token to the atlas.</span
			>{/if}
		<span class="scale-note">Intensity = tanh(a / 2)</span>
	</div>
	<span class="sr-only" aria-live="polite" aria-atomic="true">{selectedDescription}</span>
</section>

<style>
	.activation-matrix {
		min-width: 0;
		background: var(--surface, #13171a);
		color: var(--ink, #ecf0ef);
	}
	.matrix-heading {
		height: 29px;
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 16px;
		padding: 0 12px;
	}
	.matrix-heading > div:first-child {
		display: flex;
		align-items: baseline;
		gap: 12px;
		min-width: 0;
	}
	.matrix-heading strong {
		font-size: 11px;
		font-weight: 500;
		white-space: nowrap;
	}
	.matrix-heading > div > span {
		font:
			8px ui-monospace,
			SFMono-Regular,
			Menlo,
			monospace;
		letter-spacing: 0.065em;
		color: var(--muted, #7f8994);
		white-space: nowrap;
	}
	.legend {
		display: flex;
		align-items: center;
		gap: 5px;
	}
	.legend-chip {
		width: 17px;
		height: 5px;
		background: var(--surface-raised, #20272a);
	}
	.legend i {
		display: block;
		width: 100%;
		height: 100%;
		background: var(--layer-2, #88bda9);
	}
	.matrix-body {
		position: relative;
		padding: 0 6px;
	}
	.matrix-surface {
		position: relative;
		display: block;
		width: 100%;
		height: 178px;
		padding: 0;
		border: 0;
		border-radius: 2px;
		background: transparent;
		appearance: none;
		cursor: crosshair;
		text-align: left;
	}
	.matrix-surface:focus-visible {
		outline: 1px solid var(--accent, #b6d88e);
		outline-offset: 2px;
	}
	.matrix-surface:disabled {
		cursor: default;
	}
	canvas {
		display: block;
		width: 100%;
		height: 178px;
		touch-action: manipulation;
	}
	.matrix-empty {
		position: absolute;
		inset: 28px 12px 22px 49px;
		display: grid;
		place-items: center;
		background: var(--surface, #13171a);
		color: var(--muted, #7f8994);
		text-align: center;
		font-size: 11px;
		pointer-events: none;
	}
	.matrix-footer {
		min-height: 27px;
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 10px;
		padding: 0 13px;
		color: var(--muted, #7f8994);
		font:
			9px ui-monospace,
			SFMono-Regular,
			Menlo,
			monospace;
	}
	.cell-readout {
		display: flex;
		align-items: center;
		flex-wrap: wrap;
		gap: 12px;
	}
	.cell-readout strong {
		color: var(--ink, #ecf0ef);
		font-weight: 500;
		font-variant-numeric: tabular-nums;
	}
	.cell-readout b {
		color: var(--ink, #ecf0ef);
		font-weight: 400;
	}
	.cell-kind {
		color: var(--accent, #b6d88e);
		font-size: 8px;
	}
	.scale-note {
		white-space: nowrap;
		font-size: 8px;
	}
	.sr-only {
		position: absolute;
		width: 1px;
		height: 1px;
		padding: 0;
		overflow: hidden;
		clip: rect(0, 0, 0, 0);
		white-space: nowrap;
		border: 0;
	}
	@media (max-width: 600px) {
		.matrix-heading > div:first-child > span {
			display: none;
		}
		.scale-note {
			display: none;
		}
		.cell-readout {
			gap: 8px;
		}
	}
</style>
