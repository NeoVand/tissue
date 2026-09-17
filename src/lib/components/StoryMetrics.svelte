<script lang="ts">
	import type { StoryMetrics } from '$lib/stories/protocol';
	import Icon from './Icon.svelte';
	let {
		metrics,
		compact = false,
		unit = 'character'
	}: { metrics: StoryMetrics[]; compact?: boolean; unit?: 'character' | 'token' } = $props();
	let latest = $derived(metrics.at(-1));
	let observed = $derived(metrics.filter((metric) => Number.isFinite(metric.validationLoss)));
	let maximumStep = $derived(Math.max(1, ...observed.map((metric) => metric.step)));
	let minimumLoss = $derived(
		Math.max(
			0,
			Math.min(
				...observed.flatMap((metric) => [
					metric.validationLoss,
					metric.unigramLoss,
					...(metric.trainLoss === null ? [] : [metric.trainLoss])
				]),
				0
			)
		)
	);
	let maximumLoss = $derived(
		Math.max(
			1,
			...observed.flatMap((metric) => [
				metric.validationLoss,
				metric.unigramLoss,
				...(metric.trainLoss === null ? [] : [metric.trainLoss])
			])
		) * 1.08
	);
	const x = (step: number) => 34 + (step / maximumStep) * 456;
	const y = (loss: number) => 126 - ((loss - minimumLoss) / (maximumLoss - minimumLoss)) * 110;
	let heldoutLine = $derived(
		observed.map((metric) => `${x(metric.step)},${y(metric.validationLoss)}`).join(' ')
	);
	let trainLine = $derived(
		observed
			.filter((metric) => metric.trainLoss !== null)
			.map((metric) => `${x(metric.step)},${y(metric.trainLoss!)}`)
			.join(' ')
	);
	function value(number: number | null | undefined, digits = 3) {
		return number === null || number === undefined ? '—' : number.toFixed(digits);
	}
</script>

<section class:compact class="story-metrics" aria-label="TinyStories training evidence">
	<div class="metric-strip">
		<div>
			<span>Held-out loss <small>nats / {unit === 'character' ? 'char' : 'token'}</small></span
			><strong>{value(latest?.validationLoss)}</strong><small
				>Train unigram {value(latest?.unigramLoss)}</small
			>
		</div>
		<div>
			<span>Held-out accuracy</span><strong
				>{latest ? `${(latest.validationAccuracy * 100).toFixed(1)}%` : '—'}</strong
			><small>Unigram {latest ? `${(latest.unigramAccuracy * 100).toFixed(1)}%` : '—'}</small>
		</div>
		<div>
			<span>Training loss <small>nats / {unit === 'character' ? 'char' : 'token'}</small></span
			><strong>{value(latest?.trainLoss)}</strong><small>Latest training batch</small>
		</div>
		<div>
			<span>Updates</span><strong>{latest?.step.toLocaleString() ?? '—'}</strong><small
				>{latest
					? `${latest.trainedTokens.toLocaleString()} training ${unit}s`
					: 'No model initialized'}</small
			>
		</div>
	</div>
	{#if !compact}
		<div class="learning-curve">
			<div class="chart-heading">
				<h3><Icon name="chart" size={13} />Fixed held-out evidence</h3>
				<span
					>{latest
						? `${latest.evaluationTokens.toLocaleString()} evaluated ${unit}s`
						: 'Measured after initialization and training bursts'}</span
				>
			</div>
			{#if observed.length}<svg
					viewBox="0 0 512 151"
					role="img"
					aria-label={`Training and held-out ${unit} cross-entropy against the train-unigram baseline`}
				>
					{#each [0, 0.5, 1] as fraction (fraction)}{@const loss =
							minimumLoss + (maximumLoss - minimumLoss) * fraction}<line
							x1="34"
							x2="490"
							y1={y(loss)}
							y2={y(loss)}
						/><text x="27" y={y(loss) + 3} text-anchor="end">{loss.toFixed(1)}</text>{/each}
					{#if latest}<line
							class="baseline"
							x1="34"
							x2="490"
							y1={y(latest.unigramLoss)}
							y2={y(latest.unigramLoss)}
						/>{/if}
					<polyline class="train" points={trainLine} /><polyline
						class="heldout"
						points={heldoutLine}
					/>
					{#each observed as metric (metric.step)}<circle
							class="heldout"
							cx={x(metric.step)}
							cy={y(metric.validationLoss)}
							r="2"
							><title
								>Step {metric.step}: held-out {metric.validationLoss.toFixed(5)} nats per {unit}</title
							></circle
						>{/each}
					<text x="34" y="145">0</text><text x="490" y="145" text-anchor="end"
						>{maximumStep.toLocaleString()} updates</text
					>
				</svg>{:else}<div class="empty-curve">
					Training and evaluation measurements will appear here.
				</div>{/if}
			<div class="legend">
				<span><i class="heldout"></i>Held out</span><span><i class="train"></i>Training batch</span
				><span><i class="baseline"></i>Train unigram</span><span>Lower loss is better</span>
			</div>
		</div>
	{/if}
</section>

<style>
	.story-metrics {
		min-width: 0;
	}
	.metric-strip {
		display: grid;
		grid-template-columns: repeat(4, minmax(0, 1fr));
		background: var(--surface);
	}
	.metric-strip > div {
		display: flex;
		flex-direction: column;
		gap: 7px;
		padding: 14px 17px;
		min-width: 0;
	}
	.metric-strip > div + div {
		border-left: 1px solid var(--line);
	}
	.metric-strip span {
		color: var(--muted);
		font: 9px var(--mono);
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
	}
	.metric-strip span small {
		font-size: 8px;
		opacity: 0.75;
	}
	.metric-strip strong {
		font: 23px var(--mono);
		letter-spacing: -0.8px;
		font-weight: 450;
	}
	.metric-strip > div > small {
		font: 8px/1.5 var(--mono);
		color: var(--muted);
	}
	.learning-curve {
		padding: 14px 17px;
		border-top: 1px solid var(--line);
	}
	.chart-heading {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 10px;
		margin-bottom: 8px;
	}
	h3 {
		display: flex;
		align-items: center;
		gap: 6px;
		font-size: 11px;
		font-weight: 500;
		margin: 0;
	}
	.chart-heading > span {
		font: 8px var(--mono);
		color: var(--muted);
	}
	svg {
		width: 100%;
		height: 170px;
		display: block;
	}
	line {
		stroke: var(--line);
		stroke-width: 0.65;
	}
	text {
		fill: var(--muted);
		font: 8px var(--mono);
	}
	polyline {
		fill: none;
		stroke-width: 1.4;
		stroke-linejoin: round;
	}
	.heldout {
		stroke: var(--accent);
		fill: var(--accent);
	}
	polyline.heldout {
		fill: none;
	}
	.train {
		stroke: var(--layer-2);
		opacity: 0.7;
	}
	.baseline {
		stroke: var(--warning);
		stroke-dasharray: 3 3;
		stroke-width: 0.9;
	}
	.legend {
		display: flex;
		gap: 14px;
		flex-wrap: wrap;
		font: 8px var(--mono);
		color: var(--muted);
	}
	.legend span {
		display: flex;
		align-items: center;
		gap: 5px;
	}
	.legend span:last-child {
		margin-left: auto;
	}
	.legend i {
		width: 12px;
		height: 2px;
		background: var(--accent);
	}
	.legend i.train {
		background: var(--layer-2);
	}
	.legend i.baseline {
		height: 1px;
		background: var(--warning);
	}
	.empty-curve {
		min-height: 140px;
		display: flex;
		align-items: center;
		justify-content: center;
		color: var(--muted);
		font-size: 11px;
	}
	.compact .metric-strip {
		border-bottom: 1px solid var(--line);
	}
	@media (max-width: 700px) {
		.metric-strip {
			grid-template-columns: repeat(2, minmax(0, 1fr));
		}
		.metric-strip > div:nth-child(3) {
			border-left: 0;
		}
		.metric-strip > div:nth-child(n + 3) {
			border-top: 1px solid var(--line);
		}
		.chart-heading {
			flex-wrap: wrap;
		}
		.metric-strip > div {
			padding: 12px;
		}
		.metric-strip strong {
			font-size: 21px;
		}
	}
</style>
