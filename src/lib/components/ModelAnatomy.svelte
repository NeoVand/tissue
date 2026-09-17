<script lang="ts">
	import { MODEL_CONFIG as cfg, NEURON_COUNT } from '$lib/lab/protocol';
	import { MODEL_PARAMETER_COUNT } from '$lib/lab/model-inspection';
	import Icon from './Icon.svelte';
	interface Props {
		selected: number | null;
		onlayer: (layer: number | null) => void;
		layerFilter?: number | null;
		compact?: boolean;
	}
	let { selected, onlayer, layerFilter = null, compact = false }: Props = $props();
	let selectedLayer = $derived(selected === null ? null : Math.floor(selected / cfg.hidden));
</script>

<section class="anatomy" class:compact aria-label="Transformer architecture">
	<header>
		{#if compact}<span class="count">{(MODEL_PARAMETER_COUNT / 1000).toFixed(1)}k weights</span
			>{:else}<div>
				<span class="label">Model anatomy</span><span class="count"
					>{MODEL_PARAMETER_COUNT.toLocaleString()} weights</span
				>
			</div>{/if}
		<button
			type="button"
			class:active={layerFilter === null}
			aria-pressed={layerFilter === null}
			onclick={() => onlayer(null)}>All layers</button
		>
	</header>
	<div class="flow">
		<div class="endpoint embedding">
			{#if !compact}<span class="step-index">IN</span>{/if}
			<div>
				<strong>{compact ? 'Embeddings' : 'Token + position embeddings'}</strong><span
					class="formula"
					>{compact ? 'token + position · 14 × 32 each' : '14 × 32 + 14 × 32 → RMS norm'}</span
				>{#if compact}<span class="formula">RMS norm → 32d residual</span>{/if}
			</div>
			{#if !compact}<span class="dimension">32d</span>{/if}
		</div>
		{#each Array.from({ length: cfg.layers }, (_, layer) => layer) as layer (layer)}
			<div class="connector" aria-hidden="true">
				<Icon name="down" size={11} /><span>residual stream · 32 channels</span>
			</div>
			<div
				class="block"
				class:chosen={selectedLayer === layer}
				class:filtered={layerFilter === layer}
				style:--layer-color={layer === 0 ? 'var(--layer-1)' : 'var(--layer-2)'}
			>
				<button
					type="button"
					class="block-heading"
					aria-pressed={layerFilter === layer}
					onclick={() => onlayer(layerFilter === layer ? null : layer)}
				>
					<span class="layer-badge">L{layer + 1}</span><strong
						>{compact ? `layers[${layer}]` : `Transformer block ${layer + 1}`}</strong
					><span class="filter-label"
						>{#if compact}<Icon
								name={layerFilter === layer ? 'check' : 'filter'}
								size={11}
							/>{:else}{layerFilter === layer ? 'Filtered' : 'Show layer'}{/if}</span
					>
				</button>
				<div class="block-path attention">
					<span class="path-kind">ATTN</span><span
						>{#if compact}RMS → causal attention<small>4 heads × 8d → Wᵒ</small>{:else}RMS → causal
							attention <small>4 × 8d</small> → Wᵒ{/if}</span
					><span class="residual" title="Add the block input residual">+ x</span>
				</div>
				<div class="block-path mlp">
					<span class="path-kind">MLP</span><span
						>{#if compact}RMS → 32 → 128 → 32<small
								><b class="represented">ReLU · 128 atlas units</b></small
							>{:else}RMS → W₁ → <b class="represented">ReLU · 128 units</b> → W₂{/if}</span
					><span class="residual" title="Add the post-attention residual">+ x′</span>
				</div>
				<div class="node-range">
					<i></i><span
						>{compact ? 'Nodes' : 'Atlas nodes'}
						{layer * cfg.hidden}–{(layer + 1) * cfg.hidden - 1}</span
					>{#if selectedLayer === layer}<span class="selection"
							>{compact ? 'j' : 'selected · '}{selected! % cfg.hidden}</span
						>{/if}
				</div>
			</div>
		{/each}
		<div class="connector" aria-hidden="true">
			<Icon name="down" size={11} /><span>last token only</span>
		</div>
		<div class="endpoint readout">
			{#if !compact}<span class="step-index">OUT</span>{/if}
			<div>
				<strong>{compact ? 'Last-token readout' : 'Readout → next-token distribution'}</strong><span
					class="formula">lmHead [32 × 14] → softmax</span
				>
			</div>
			{#if !compact}<span class="dimension">14p</span>{/if}
		</div>
	</div>
	<p class="caption">
		<i></i>{#if compact}{NEURON_COUNT} post-ReLU MLP points, reused per token. Attention is not plotted.{:else}The
			atlas contains {NEURON_COUNT} MLP channels after ReLU, reused at every token. Attention heads and
			residual channels are outside this point set.{/if}
	</p>
</section>

<style>
	.anatomy {
		color: var(--ink, #dae2eb);
		background: var(--surface, #111820);
		border: 1px solid var(--line, #26303b);
		border-radius: 8px;
		overflow: hidden;
		font:
			11px/1.4 'DM Sans',
			sans-serif;
	}
	header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 8px;
		padding: 11px 12px;
		border-bottom: 1px solid var(--line, #26303b);
	}
	header > div {
		display: flex;
		flex-wrap: wrap;
		gap: 5px 12px;
		align-items: baseline;
	}
	.label {
		font-size: 11px;
		font-weight: 650;
		letter-spacing: 0.02em;
	}
	.count,
	.formula,
	.dimension,
	.step-index,
	.connector,
	.path-kind,
	.node-range,
	.residual,
	.layer-badge {
		font-family: 'IBM Plex Mono', monospace;
	}
	.count {
		color: var(--muted, #8c9aaa);
		font-size: 9px;
	}
	button {
		font: inherit;
		color: inherit;
		cursor: pointer;
	}
	header button {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		border: 1px solid var(--line, #26303b);
		background: transparent;
		border-radius: 4px;
		padding: 4px 7px;
		font-size: 10px;
		white-space: nowrap;
	}
	header button.active {
		color: var(--accent, #87bda7);
		background: color-mix(in srgb, var(--accent, #87bda7) 8%, transparent);
	}
	button:focus-visible {
		outline: 2px solid var(--accent, #87bda7);
		outline-offset: -2px;
	}
	.flow {
		padding: 11px 12px 8px;
	}
	.endpoint {
		display: flex;
		align-items: center;
		gap: 9px;
		padding: 7px 9px;
		border: 1px solid var(--line, #26303b);
		border-radius: 5px;
		background: var(--surface-raised, #161f29);
	}
	.endpoint > div {
		display: grid;
		gap: 2px;
	}
	.endpoint strong {
		font-weight: 550;
		font-size: 10px;
	}
	.step-index {
		color: var(--muted, #8c9aaa);
		font-size: 8px;
		width: 22px;
	}
	.formula {
		color: var(--muted, #8c9aaa);
		font-size: 9px;
	}
	.dimension {
		margin-left: auto;
		color: var(--muted, #8c9aaa);
		font-size: 9px;
	}
	.connector {
		display: flex;
		gap: 7px;
		align-items: center;
		padding: 3px 11px;
		font-size: 10px;
		color: var(--muted, #8c9aaa);
	}
	.connector span {
		font-size: 8px;
		opacity: 0.8;
	}
	.block {
		border: 1px solid color-mix(in srgb, var(--layer-color, #859cae) 35%, var(--line, #26303b));
		border-radius: 5px;
		overflow: hidden;
		transition: border-color 130ms ease;
	}
	.block.chosen {
		border-color: color-mix(in srgb, var(--layer-color, #859cae) 70%, var(--line, #26303b));
	}
	.block.filtered {
		background: color-mix(in srgb, var(--layer-color, #859cae) 6%, transparent);
	}
	.block-heading {
		display: flex;
		align-items: center;
		width: 100%;
		gap: 7px;
		padding: 7px 8px;
		background: var(--surface-raised, #161f29);
		border: 0;
		border-bottom: 1px solid var(--line, #26303b);
		text-align: left;
	}
	.layer-badge {
		color: var(--layer-color, #859cae);
		font-size: 10px;
	}
	.block-heading strong {
		font-weight: 550;
		font-size: 10px;
	}
	.filter-label {
		margin-left: auto;
		color: var(--muted, #8c9aaa);
		font-size: 9px;
	}
	.block-path {
		display: flex;
		gap: 7px;
		align-items: baseline;
		padding: 5px 8px;
		font-size: 9px;
	}
	.block-path > span:nth-child(2) {
		flex: 1;
	}
	.path-kind {
		width: 29px;
		font-size: 8px;
		color: var(--muted, #8c9aaa);
		flex-shrink: 0;
	}
	.block-path small {
		color: var(--muted, #8c9aaa);
		font-size: 8px;
		white-space: nowrap;
	}
	.represented {
		font-weight: 500;
		color: var(--layer-color, #859cae);
		background: color-mix(in srgb, var(--layer-color, #859cae) 10%, transparent);
		border-radius: 3px;
		padding: 2px 3px;
		white-space: nowrap;
	}
	.residual {
		color: var(--muted, #8c9aaa);
		font-size: 8px;
		white-space: nowrap;
	}
	.node-range {
		display: flex;
		gap: 5px;
		align-items: center;
		padding: 4px 8px 6px 44px;
		color: var(--muted, #8c9aaa);
		font-size: 8px;
	}
	.node-range i,
	.caption i {
		width: 4px;
		height: 4px;
		border-radius: 50%;
		background: var(--layer-color, var(--accent, #87bda7));
		flex-shrink: 0;
	}
	.selection {
		color: var(--layer-color, #859cae);
		margin-left: auto;
	}
	.caption {
		display: flex;
		align-items: baseline;
		gap: 6px;
		margin: 0;
		padding: 0 12px 10px;
		color: var(--muted, #8c9aaa);
		font-size: 9px;
		line-height: 1.55;
	}
	.compact {
		border: 0;
		border-radius: 0;
		background: transparent;
	}
	.compact header {
		padding: 7px 8px;
		border-bottom: 0;
		gap: 4px;
	}
	.compact header button {
		font-size: 9px;
		padding: 3px 4px;
	}
	.compact .count {
		font-size: 9px;
	}
	.compact .flow {
		padding: 0 8px 6px;
	}
	.compact .endpoint {
		padding: 6px;
		gap: 0;
	}
	.compact .endpoint strong {
		font-size: 10px;
	}
	.compact .formula {
		font-size: 8px;
	}
	.compact .connector {
		padding: 1px 7px;
		height: 12px;
	}
	.compact .connector span {
		display: none;
	}
	.compact .block-heading {
		padding: 6px;
		gap: 5px;
	}
	.compact .block-heading strong {
		font:
			9px 'IBM Plex Mono',
			monospace;
	}
	.compact .block-path {
		gap: 4px;
		padding: 5px 5px 2px;
		font-size: 9px;
		align-items: start;
	}
	.compact .block-path small {
		display: block;
		margin-top: 2px;
		font-size: 8px;
	}
	.compact .block-path > span:nth-child(2) {
		min-width: 0;
	}
	.compact .path-kind {
		width: 24px;
		font-size: 8px;
		padding-top: 1px;
	}
	.compact .residual {
		font-size: 8px;
	}
	.compact .represented {
		font-size: 8px;
		padding: 1px 2px;
	}
	.compact .node-range {
		padding: 4px 6px 6px 32px;
		gap: 4px;
		font-size: 8px;
	}
	.compact .caption {
		padding: 0 8px 7px;
		font-size: 9px;
		line-height: 1.5;
	}
	@media (prefers-reduced-motion: reduce) {
		.block {
			transition: none;
		}
	}
</style>
