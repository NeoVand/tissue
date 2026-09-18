<script lang="ts">
	import report from '../../../static/experiments/fingerprint-comparison.json';

	const groups = report.runs.map((run) => ({
		...run,
		maps: [
			{ name: 'Activation', ...run.activation },
			{ name: 'Intervention', ...run.effect }
		]
	}));
	const percent = (value: number | null, digits = 1) =>
		value === null ? 'Unavailable' : `${(100 * value).toFixed(digits)}%`;
	const earlyOverlaps = report.history
		.filter((run) => run.checkpoint === 500)
		.map((run) => run.overlap.share);
	const earlyOverlapRange = `${percent(Math.min(...earlyOverlaps))}–${percent(Math.max(...earlyOverlaps))}`;
</script>

<section class="comparison" aria-label="Recorded fingerprint comparison">
	<header>
		<h3>Layer membership shapes the neighborhoods</h3>
		<span>Recorded step 2,000 · k = {report.k}</span>
	</header>
	<div class="table-scroll">
		<table>
			<caption>
				Two reference seeds at 2,000 updates. Neighbors come from original fingerprint distances.
			</caption>
			<thead>
				<tr>
					<th scope="col">Seed</th>
					<th scope="col">Fingerprint</th>
					<th scope="col">Retained in 3D</th>
					<th scope="col">Same layer</th>
					<th scope="col">Shared across maps</th>
				</tr>
			</thead>
			{#each groups as group (group.seed)}
				<tbody>
					{#each group.maps as map, index (map.name)}
						<tr>
							{#if index === 0}
								<th scope="rowgroup" rowspan="2" class="seed">{group.seed}</th>
							{/if}
							<th scope="row" class="map-name">
								{map.name}<span>{map.validNeurons} units</span>
							</th>
							<td title="Fraction of projected neighbors within the original sixth-neighbor radius">
								{percent(map.neighborRetention)}
							</td>
							<td
								title={`${map.withinLayer.sameLayerLinks} of ${map.withinLayer.directedLinks} directed neighbor selections`}
							>
								{percent(map.withinLayer.share)}
							</td>
							{#if index === 0}
								<td
									rowspan="2"
									class="overlap"
									title={`${group.overlap.sharedNeighborLinks} shared of ${group.overlap.directedActivationNeighbors} activation-neighbor selections; ${group.overlap.commonValidNeurons} common focal units`}
								>
									{percent(group.overlap.share)}
									<span>uniform {percent(group.overlap.nullExpectedShare, 2)}</span>
								</td>
							{/if}
						</tr>
					{/each}
				</tbody>
			{/each}
		</table>
	</div>
	<footer>
		<p>
			Same-layer uniform expectation: 49.80–49.83%. The maps differ, with more overlap than
			independent uniform draws.
		</p>
		<p>At step 500, overlap was {earlyOverlapRange}; it decreased in both seeds by step 2,000.</p>
		<p>
			These are fixed reference measurements, not metrics for the current model. Geometry alone does
			not identify concepts.
		</p>
	</footer>
</section>

<style>
	.comparison {
		padding: 12px 16px 10px;
		color: var(--ink);
		font-size: 13px;
		font-variant-numeric: tabular-nums;
	}
	header {
		display: flex;
		align-items: baseline;
		justify-content: space-between;
		gap: 12px;
		margin-bottom: 8px;
	}
	h3 {
		margin: 0;
		font-size: 12px;
		font-weight: 550;
	}
	header > span {
		flex-shrink: 0;
		color: var(--muted);
		font-size: 12px;
	}
	.table-scroll {
		overflow-x: auto;
	}
	table {
		width: 100%;
		min-width: 560px;
		border-collapse: collapse;
		text-align: right;
	}
	caption {
		position: absolute;
		width: 1px;
		height: 1px;
		overflow: hidden;
		clip-path: inset(50%);
		white-space: nowrap;
	}
	th,
	td {
		padding: 5px 10px;
		font-weight: 400;
	}
	thead th {
		padding-top: 0;
		padding-bottom: 6px;
		color: var(--muted);
		font-size: 12px;
	}
	tbody {
		border-top: 1px solid var(--line);
	}
	thead th:first-child,
	.seed {
		padding-left: 0;
		text-align: left;
	}
	thead th:nth-child(2),
	.map-name {
		text-align: left;
	}
	.map-name span {
		margin-left: 8px;
		color: var(--muted);
		font-size: 12px;
	}
	.seed {
		width: 32px;
		color: var(--muted);
	}
	.overlap {
		border-left: 1px solid var(--line);
	}
	.overlap span {
		display: block;
		margin-top: 3px;
		color: var(--muted);
		font-size: 12px;
	}
	footer {
		margin-top: 8px;
		color: var(--muted);
		font-size: 12px;
		line-height: 1.45;
	}
	footer p {
		margin: 2px 0 0;
	}
	@media (max-width: 640px) {
		header {
			align-items: flex-start;
			flex-direction: column;
			gap: 3px;
		}
	}
</style>
