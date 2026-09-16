<script lang="ts">
	import type { Metrics } from '$lib/lab/protocol';
	let { metrics, selectedStep = 0 }: { metrics: Metrics[]; selectedStep?: number } = $props();
	const width = 620;
	const height = 150;
	const pad = { left: 39, right: 15, top: 16, bottom: 27 };
	let maxStep = $derived(Math.max(1, ...metrics.map((m) => m.step)));
	let maxLoss = $derived(Math.max(Math.log(8), ...metrics.map((m) => m.validationLoss)) * 1.08);
	const x = (step: number) => pad.left + (step / maxStep) * (width - pad.left - pad.right);
	const y = (loss: number) =>
		height - pad.bottom - (loss / maxLoss) * (height - pad.top - pad.bottom);
	let path = $derived(
		metrics.map((m, i) => `${i ? 'L' : 'M'}${x(m.step)},${y(m.validationLoss)}`).join(' ')
	);
	let selected = $derived(metrics.find((m) => m.step === selectedStep));
</script>

<div class="curve">
	<svg
		viewBox="0 0 {width} {height}"
		role="img"
		aria-label="Held-out answer loss in nats by training step"
	>
		{#each [0, 1, 2, 3].filter((tick) => tick <= maxLoss) as tick (tick)}
			<line x1={pad.left} y1={y(tick)} x2={width - pad.right} y2={y(tick)} class="grid" />
			<text x={pad.left - 10} y={y(tick) + 4} text-anchor="end">{tick}</text>
		{/each}
		<line
			x1={pad.left}
			y1={y(Math.log(8))}
			x2={width - pad.right}
			y2={y(Math.log(8))}
			class="baseline"
		/>
		<path d={path} />
		{#if selected}<circle cx={x(selected.step)} cy={y(selected.validationLoss)} r="4" />{/if}
		<text x={pad.left} y={height - 5}>0</text>
		<text x={width - pad.right} y={height - 5} text-anchor="end">{maxStep} steps</text>
		<text x={pad.left} y="10" class="axis-label">nats / answer</text>
	</svg>
</div>

<style>
	.curve {
		width: 100%;
	}
	svg {
		display: block;
		width: 100%;
		overflow: visible;
	}
	text {
		font: 11px var(--mono);
		fill: var(--muted);
	}
	.grid {
		stroke: var(--line);
		stroke-width: 0.7;
	}
	.baseline {
		stroke: var(--clay);
		stroke-width: 1;
		stroke-dasharray: 4 5;
		opacity: 0.6;
	}
	path {
		fill: none;
		stroke: var(--forest);
		stroke-width: 2;
		stroke-linecap: round;
		stroke-linejoin: round;
	}
	circle {
		fill: var(--forest);
		stroke: var(--paper);
		stroke-width: 2;
	}
	.axis-label {
		font-size: 10px;
	}
</style>
