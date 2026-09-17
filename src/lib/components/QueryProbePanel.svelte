<script lang="ts">
	import type { QueryMeasurement } from '$lib/lab/query-protocol';
	import Icon from './Icon.svelte';
	let { measurement, selected }: { measurement: QueryMeasurement; selected: number | null } =
		$props();
	let split = $state<'calibration' | 'test'>('test');
	let groupIndex = $state(0);
	const queries = ['a', 'b', 'c'];
	const answers = Array.from({ length: 8 }, (_, index) => index);
	let groups = $derived(measurement.groups[split]);
	let index = $derived(Math.min(groupIndex, Math.max(0, groups.length - 1)));
	let group = $derived(groups[index]);
	let rows = $derived.by(() => {
		if (selected === null) return [];
		const data = measurement.splits[split];
		return queries.map((query, q) => {
			const offset = (index * 3 + q) * 8;
			const effects = data.effects[selected].slice(offset, offset + 8);
			return {
				query,
				effects,
				activation: data.activations[selected][index * 3 + q],
				rms: Math.sqrt(effects.reduce((sum, value) => sum + value * value, 0) / 8)
			};
		});
	});
	let measuredExtent = $derived(Math.max(0, ...rows.flatMap((row) => row.effects.map(Math.abs))));
	let extent = $derived(measuredExtent || 1);
	let ticks = $derived(measuredExtent === 0 ? [0] : [-1, 0, 1]);
	const plotX = (answer: number) => 45 + answer * 48;
	function plotY(value: number): number {
		return 87 - (value / extent) * 61;
	}
	function formatted(value: number): string {
		if (value === 0) return '0.000';
		return Math.abs(value) < 0.001 ? value.toExponential(2) : value.toFixed(3);
	}
	function signed(value: number): string {
		return `${value > 0 ? '+' : ''}${formatted(value)}`;
	}
	function step(direction: number): void {
		groupIndex = (index + direction + groups.length) % groups.length;
	}
</script>

<section class="paired-probes" aria-label="Paired query measurements">
	<div class="section-heading">
		<div>
			<Icon name="activity" size={14} />
			<h2>Same assignment, three queries</h2>
		</div>
		<div class="probe-controls">
			<div class="segmented" aria-label="Probe split">
				<button
					class:active={split === 'calibration'}
					aria-pressed={split === 'calibration'}
					onclick={() => {
						split = 'calibration';
						groupIndex = 0;
					}}>Calibration</button
				>
				<button
					class:active={split === 'test'}
					aria-pressed={split === 'test'}
					onclick={() => {
						split = 'test';
						groupIndex = 0;
					}}>Held out</button
				>
			</div>
			<div class="stepper">
				<button
					aria-label="Previous assignment"
					onclick={() => step(-1)}
					disabled={groups.length < 2}><Icon name="left" size={13} /></button
				>
				<span>{String(index + 1).padStart(2, '0')} / {String(groups.length).padStart(2, '0')}</span>
				<button aria-label="Next assignment" onclick={() => step(1)} disabled={groups.length < 2}
					><Icon name="right" size={13} /></button
				>
			</div>
		</div>
	</div>
	<div class="probe-body">
		<div class="prompts">
			<p class="subtle">
				{split === 'test'
					? 'Held-out assignment · used only to score neighborhoods'
					: 'Calibration assignment · used to choose neighborhoods'}
			</p>
			{#if group}
				{#each group.examples as example, q (example.id)}
					<div class="prompt-row" style:--query-color={`var(--query-${q})`}>
						<span class="query-label">?{example.query}</span>
						<code
							>{example.tokens.slice(0, -1).join('')}<strong>{example.tokens.at(-1)}</strong></code
						>
						<Icon name="right" size={12} /><span class="answer" title="Correct answer"
							>{example.answer}</span
						>
					</div>
				{/each}
			{/if}
			<p class="prompt-note">
				The first 13 tokens are identical. Only the final query token changes.
			</p>
			{#if selected !== null}
				<div class="raw-stats">
					<span></span><span>Final-token activation</span><span>Effect RMS</span>
					{#each rows as row, q (row.query)}
						<strong style:color={`var(--query-${q})`}>?{row.query}</strong><span
							>{formatted(row.activation)}</span
						><span>{formatted(row.rms)}</span>
					{/each}
				</div>
			{/if}
		</div>
		<div class="effect-plot">
			<div class="plot-heading">
				<h3>Raw intervention effect</h3>
				<span>Δ answer probability</span>
			</div>
			{#if selected !== null}
				<svg
					viewBox="0 0 410 184"
					role="img"
					aria-label={`Unit ${selected} intervention effects for queries a, b and c, on ${split} assignment ${index + 1}`}
				>
					{#each ticks as tick (tick)}
						<line
							x1="45"
							x2="381"
							y1={plotY(tick * extent)}
							y2={plotY(tick * extent)}
							class:zero={tick === 0}
						/>
						<text x="37" y={plotY(tick * extent) + 3} text-anchor="end"
							>{signed(tick * extent)}</text
						>
					{/each}
					{#each rows as row, q (row.query)}
						<polyline
							points={row.effects
								.map((value, answer) => `${plotX(answer)},${plotY(value)}`)
								.join(' ')}
							style:stroke={`var(--query-${q})`}
						/>
						{#each row.effects as effect, answer (answer)}
							<circle cx={plotX(answer)} cy={plotY(effect)} r="2.7" style:fill={`var(--query-${q})`}
								><title>Query {row.query}, answer {answer}: {signed(effect)}</title></circle
							>
						{/each}
					{/each}
					{#each answers as answer (answer)}<text x={plotX(answer)} y="168" text-anchor="middle"
							>{answer}</text
						>{/each}
				</svg>
				<div class="plot-legend">
					{#each queries as query, q (query)}<span
							><i style:background={`var(--query-${q})`}></i>Query {query}</span
						>{/each}<span>Answer value 0–7</span>
				</div>
				{#if measuredExtent === 0}<p class="zero-effects">
						All 24 measured answer-probability changes are zero on this assignment.
					</p>{/if}
				<details class="raw-values">
					<summary>Inspect numeric values</summary>
					<table>
						<caption>Lesioned minus intact probability · 6 significant digits</caption><thead
							><tr
								><th>Answer</th>{#each queries as query (query)}<th>?{query}</th>{/each}</tr
							></thead
						><tbody
							>{#each answers as answer (answer)}<tr
									><th>{answer}</th>{#each rows as row (row.query)}<td
											>{row.effects[answer].toPrecision(6)}</td
										>{/each}</tr
								>{/each}</tbody
						>
					</table>
				</details>
			{:else}
				<div class="choose-unit">
					<Icon name="target" size={22} />
					<p>Select a unit to inspect its measured effects.</p>
				</div>
			{/if}
			<p class="plot-note">
				Silence this MLP channel at every token position. Negative values reduce an answer’s
				probability; the eight values need not sum to zero because six other vocabulary tokens
				exist. Axis range follows the selected unit and assignment.
			</p>
		</div>
	</div>
</section>

<style>
	.paired-probes {
		--query-0: var(--layer-1);
		--query-1: var(--layer-2);
		--query-2: var(--warning);
		border: 1px solid var(--line);
		border-radius: 7px;
		background: var(--surface);
		overflow: hidden;
		min-width: 0;
	}
	.section-heading,
	.section-heading > div,
	.probe-controls,
	.stepper,
	.prompt-row,
	.plot-heading,
	.plot-legend,
	.plot-legend span {
		display: flex;
		align-items: center;
	}
	.section-heading {
		padding: 12px 16px;
		justify-content: space-between;
		gap: 12px;
		border-bottom: 1px solid var(--line);
	}
	.section-heading > div {
		gap: 8px;
	}
	h2,
	h3,
	p {
		margin: 0;
	}
	h2 {
		font-size: 12px;
		font-weight: 550;
	}
	h3 {
		font-size: 11px;
		font-weight: 550;
	}
	.probe-controls {
		gap: 12px;
		flex-wrap: wrap;
	}
	.segmented {
		display: flex;
		border: 1px solid var(--line);
		border-radius: 5px;
		padding: 2px;
	}
	.segmented button {
		border: 0;
		border-radius: 3px;
		padding: 5px 8px;
		font-size: 10px;
		color: var(--muted);
		background: transparent;
	}
	.segmented button.active {
		background: var(--surface-hover);
		color: var(--ink);
	}
	.stepper {
		gap: 5px;
		font: 9px var(--mono);
		color: var(--muted);
	}
	.stepper button {
		display: flex;
		padding: 4px;
		border: 0;
		background: transparent;
		border-radius: 3px;
	}
	.stepper button:hover {
		background: var(--surface-hover);
	}
	.probe-body {
		display: grid;
		grid-template-columns: 1fr 1fr;
	}
	.prompts,
	.effect-plot {
		padding: 15px 18px;
		min-width: 0;
	}
	.prompts {
		border-right: 1px solid var(--line);
	}
	.subtle,
	.prompt-note,
	.plot-note {
		color: var(--muted);
		font-size: 10px;
		line-height: 1.6;
	}
	.subtle {
		margin-bottom: 12px;
	}
	.prompt-row {
		gap: 9px;
		margin: 7px 0;
		padding: 8px 10px;
		border: 1px solid color-mix(in srgb, var(--query-color) 28%, var(--line));
		border-radius: 5px;
		background: color-mix(in srgb, var(--query-color) 3%, transparent);
		white-space: nowrap;
	}
	.query-label {
		font: 9px var(--mono);
		color: var(--query-color);
	}
	code {
		margin-right: auto;
		font: 11px var(--mono);
		letter-spacing: 0.04em;
	}
	code strong {
		border: 1px solid var(--query-color);
		border-radius: 3px;
		color: var(--query-color);
		padding: 1px 3px;
		margin-left: 2px;
		font-weight: 500;
	}
	.answer {
		font: 12px var(--mono);
		color: var(--query-color);
	}
	.prompt-note {
		margin-top: 10px;
	}
	.raw-stats {
		display: grid;
		grid-template-columns: 35px 1fr 1fr;
		gap: 8px;
		margin-top: 17px;
		font: 10px var(--mono);
	}
	.raw-stats > span:nth-child(2),
	.raw-stats > span:nth-child(3) {
		color: var(--muted);
		font-size: 9px;
	}
	.raw-stats strong {
		font-weight: 500;
	}
	.plot-heading {
		justify-content: space-between;
		gap: 8px;
	}
	.plot-heading > span {
		font: 9px var(--mono);
		color: var(--muted);
	}
	svg {
		width: 100%;
		height: 190px;
		display: block;
		overflow: visible;
	}
	svg line {
		stroke: var(--line);
		stroke-width: 0.7;
	}
	svg line.zero {
		stroke: var(--muted);
		stroke-dasharray: 2 3;
		opacity: 0.6;
	}
	svg polyline {
		stroke-width: 1.4;
		fill: none;
		stroke-linejoin: round;
	}
	svg text {
		fill: var(--muted);
		font: 8px var(--mono);
	}
	.plot-legend {
		gap: 14px;
		font: 9px var(--mono);
		color: var(--muted);
		margin-bottom: 10px;
		flex-wrap: wrap;
	}
	.plot-legend span {
		gap: 5px;
	}
	.plot-legend span:last-child {
		margin-left: auto;
	}
	.plot-legend i {
		width: 11px;
		height: 2px;
		border-radius: 2px;
	}
	.zero-effects {
		color: var(--muted);
		font-size: 10px;
		line-height: 1.5;
		margin-bottom: 8px;
	}
	.raw-values {
		color: var(--muted);
		font-size: 9px;
		margin-bottom: 10px;
	}
	summary {
		cursor: pointer;
		padding: 3px 0;
	}
	table {
		width: 100%;
		border-collapse: collapse;
		font: 9px var(--mono);
		margin: 7px 0;
	}
	caption {
		text-align: left;
		padding: 5px 0 8px;
	}
	th,
	td {
		text-align: right;
		padding: 5px;
		border-bottom: 1px solid var(--line);
	}
	th {
		font-weight: 500;
	}
	.choose-unit {
		min-height: 210px;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: 12px;
		color: var(--muted);
		font-size: 11px;
	}
	@media (max-width: 1050px) {
		.section-heading {
			align-items: flex-start;
			flex-wrap: wrap;
		}
		.prompt-row {
			gap: 6px;
			padding: 8px;
		}
		code {
			font-size: 10px;
		}
	}
	@media (max-width: 720px) {
		.probe-body {
			grid-template-columns: 1fr;
		}
		.prompts {
			border-right: 0;
			border-bottom: 1px solid var(--line);
		}
		.section-heading {
			padding: 12px;
		}
		.prompts,
		.effect-plot {
			padding: 14px;
		}
		.probe-controls {
			width: 100%;
			justify-content: space-between;
		}
	}
</style>
