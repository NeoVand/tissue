<script lang="ts">
	import type { Snippet } from 'svelte';
	import { QUERY_METHODS, type QueryAnalysis, type QueryMethod } from '$lib/lab/query-analysis';
	import type { QueryMeasurement } from '$lib/lab/query-protocol';
	import type { Backend } from '$lib/lab/protocol';
	import Icon from './Icon.svelte';
	import NeuralField from './NeuralField.svelte';
	import QueryProbePanel from './QueryProbePanel.svelte';

	interface Props {
		savedStudies?: Snippet;
		measurement: QueryMeasurement | null;
		analysis: QueryAnalysis | null;
		busy: boolean;
		activity?: 'measurement' | 'analysis' | 'loading';
		status: string;
		progress: { completed: number; total: number };
		theme: 'dark' | 'light';
		selected: number | null;
		onselect: (id: number) => void;
		onrun: () => void;
		oncancel: () => void;
		onreference: (seed: 42 | 7) => void;
		onexport: () => void;
		onimport?: () => void;
		source?: 'current' | 'reference';
		canRun?: boolean;
		checkpoint?: { seed: number; step: number; backend: Backend } | null;
	}
	let {
		savedStudies,
		measurement,
		analysis,
		busy,
		activity = 'measurement',
		status,
		progress,
		theme,
		selected,
		onselect,
		onrun,
		oncancel,
		onreference,
		onexport,
		onimport,
		source = 'current',
		canRun = true,
		checkpoint = null
	}: Props = $props();
	let method = $state<QueryMethod>('query-effect');
	let scope = $state<number | null>(null);
	let unit = $derived(
		selected === null ? null : (analysis?.units.find((unit) => unit.id === selected) ?? null)
	);
	let summary = $derived(
		scope === null ? analysis?.overall : analysis?.layers.find((layer) => layer.layer === scope)
	);
	let map = $derived(analysis?.calibrationGeometry);
	let points = $derived(
		map?.positions.flatMap((position, id) =>
			map.valid[id]
				? [
						{
							id,
							position,
							layer: Math.floor(id / 128),
							activation: 0,
							effect: analysis?.units[id]?.strengths.calibrationQueryEffectRms ?? 0
						}
					]
				: []
		) ?? []
	);
	let edges = $derived<[number, number][]>(
		unit?.methods[method].neighbors.map((id) => [unit.id, id]) ?? []
	);
	let methodResult = $derived(unit?.methods[method]);
	let candidateStrengths = $derived(
		unit?.candidatePool.map((id) => analysis!.units[id].strengths.calibrationFullEffectRms) ?? []
	);
	let candidateMinimum = $derived(
		candidateStrengths.length ? Math.min(...candidateStrengths) : undefined
	);
	let candidateMaximum = $derived(
		candidateStrengths.length ? Math.max(...candidateStrengths) : undefined
	);
	let maximumStrengthRatio = $derived(
		unit && candidateStrengths.length
			? Math.max(
					...candidateStrengths.map((strength) =>
						Math.max(
							strength / unit.strengths.calibrationFullEffectRms,
							unit.strengths.calibrationFullEffectRms / strength
						)
					)
				)
			: null
	);
	let currentDiffers = $derived(
		!!measurement &&
			!!checkpoint &&
			(measurement.seed !== checkpoint.seed || measurement.step !== checkpoint.step)
	);
	let percent = $derived(
		progress.total ? Math.min(100, (100 * progress.completed) / progress.total) : 0
	);
	let measuredAt = $derived(
		measurement
			? new Date(measurement.capturedAt).toLocaleString(undefined, {
					dateStyle: 'medium',
					timeStyle: 'short'
				})
			: ''
	);
	const statusLabels = {
		'calibration-unresolved': 'Calibration direction unresolved',
		'insufficient-candidates': 'Too few eligible matched candidates',
		'test-unresolved': 'Held-out query direction unresolved',
		scored: 'Held-out score available'
	};
	function value(number: number | null | undefined, digits = 3): string {
		return number === null || number === undefined ? '—' : number.toFixed(digits);
	}
	function signed(number: number | null | undefined): string {
		return number === null || number === undefined
			? '—'
			: `${number > 0 ? '+' : ''}${number.toFixed(3)}`;
	}
	function rate(number: number | null | undefined): string {
		return number === null || number === undefined ? '—' : `${(number * 100).toFixed(1)}%`;
	}
	function magnitude(number: number | undefined): string {
		if (number === undefined) return '—';
		return number !== 0 && Math.abs(number) < 0.001 ? number.toExponential(2) : number.toFixed(4);
	}
	function address(id: number): string {
		return `L${Math.floor(id / 128) + 1} / ${String(id % 128).padStart(3, '0')}`;
	}
	function selectRelative(delta: number): void {
		onselect(((selected ?? (delta > 0 ? -1 : 0)) + delta + 256) % 256);
	}
</script>

<div class="query-study">
	<header class="study-heading">
		<div class="heading-copy">
			<span class="eyebrow">Study 002 · Paired-query transfer</span>
			<h1>Paired-query transfer</h1>
			<p>
				Hold the assignment fixed. Change only the queried variable. Test whether a neighborhood’s
				effect similarity carries to unseen assignments.
			</p>
		</div>
		<div class="study-actions">
			{#if busy}<button class="secondary" onclick={oncancel}
					><Icon name="close" size={14} />{activity === 'measurement'
						? 'Cancel measurement'
						: 'Cancel'}</button
				>{:else}<button class="primary" onclick={onrun} disabled={!canRun}
					><Icon name="flask" size={14} />Measure current checkpoint</button
				>{/if}
			<div class="secondary-actions">
				<button onclick={() => onreference(42)} disabled={busy}>Reference 42</button><button
					onclick={() => onreference(7)}
					disabled={busy}>Reference 7</button
				>{#if onimport}<button
						class="icon-button"
						onclick={onimport}
						disabled={busy}
						aria-label="Import query study"
						title="Import query study"><Icon name="upload" size={14} /></button
					>{/if}<button
					class="icon-button"
					onclick={onexport}
					disabled={!measurement || busy}
					aria-label="Export query study"
					title="Export raw evidence and provenance"><Icon name="download" size={14} /></button
				>
			</div>
			<span class="resident"
				>Current checkpoint {checkpoint
					? `· seed ${checkpoint.seed} / step ${checkpoint.step.toLocaleString()} / ${checkpoint.backend.toUpperCase()}`
					: 'not ready'}</span
			>
		</div>
	</header>

	{#if savedStudies}<details class="lab-disclosure protocol-disclosure">
			<summary>Saved query studies</summary>{@render savedStudies()}
		</details>{/if}
	<details class="lab-disclosure protocol-disclosure">
		<summary
			>Study protocol <small>Paired prompts, matched controls & held-out evaluation</small></summary
		>
		<div class="protocol-strip">
			<div>
				<span class="protocol-number">01</span>
				<p>
					<strong>Paired prompts</strong><span
						>16 calibration + 16 held-out assignments × 3 queries</span
					>
				</p>
			</div>
			<div>
				<span class="protocol-number">02</span>
				<p>
					<strong>Controlled neighbors</strong><span
						>Up to 32 same-layer, strength-matched candidates → 6 neighbors</span
					>
				</p>
			</div>
			<div>
				<span class="protocol-number">03</span>
				<p>
					<strong>Held-out comparison</strong><span
						>Query-effect cosine versus the same pool’s random expectation</span
					>
				</p>
			</div>
		</div>
	</details>
	{#if busy}
		<div class="measurement-progress" role="status">
			<div>
				<Icon name="activity" size={14} /><span
					>{status ||
						(activity === 'measurement'
							? 'Measuring paired-query interventions'
							: activity === 'analysis'
								? 'Analyzing paired-query evidence'
								: 'Loading query study')}</span
				>{#if activity === 'measurement'}<span class="progress-count"
						>{progress.completed.toLocaleString()} / {progress.total.toLocaleString()}</span
					>{/if}
			</div>
			{#if activity === 'measurement'}<progress
					value={progress.completed}
					max={Math.max(1, progress.total)}
					aria-label="Measurement progress">{percent.toFixed(0)}%</progress
				>{/if}
			<p>
				{#if activity === 'measurement'}The worker measures intact responses and single-unit
					interventions. The checkpoint, optimizer and training random state are preserved.{:else if activity === 'analysis'}Computing
					calibration geometry and held-out comparisons from the raw measurements.{:else}Loading raw
					evidence and provenance. Analysis is recomputed locally from the saved measurements.{/if}
			</p>
		</div>
	{/if}

	{#if measurement && analysis}
		<div class="provenance">
			<span class="source"
				><Icon name={source === 'reference' ? 'book' : 'flask'} size={12} />{source === 'reference'
					? 'Recorded reference'
					: 'Browser measurement'}</span
			><span>Seed {measurement.seed}</span><span>Step {measurement.step.toLocaleString()}</span
			><span>{measurement.backend.toUpperCase()}</span><time datetime={measurement.capturedAt}
				>{measuredAt}</time
			><span title={measurement.checkpointHash}
				>SHA-256 {measurement.checkpointHash.slice(0, 12)}</span
			>{#if currentDiffers}<span class="different"
					>Shown study differs from the current checkpoint</span
				>{/if}
			<span
				>Intact accuracy · cal {rate(measurement.splits.calibration.accuracy)} / held out {rate(
					measurement.splits.test.accuracy
				)}</span
			>
		</div>
		<div class="research-grid">
			<section class="map-panel panel">
				<div class="panel-heading">
					<div>
						<Icon name="cube" size={14} />
						<h2>Calibration contrast geometry</h2>
					</div>
					<span class="small-label">Frozen before held-out scoring</span>
				</div>
				<div class="map-view">
					<NeuralField
						{points}
						{edges}
						{selected}
						{onselect}
						{theme}
						mode="effect"
						geometryLabel="Query-effect contrasts / 3D PCA"
					/>
				</div>
				<div class="map-diagnostics">
					<span><i style:background="var(--layer-1)"></i>L1</span><span
						><i style:background="var(--layer-2)"></i>L2</span
					><span>{points.length}/256 resolved</span><span
						>Variance <strong>{rate(map?.explainedVariance)}</strong></span
					><span>Neighbors retained <strong>{rate(map?.neighborRetention)}</strong></span>
				</div>
				<p class="map-note">
					Coordinates use only calibration query-effect contrasts. Size shows their raw RMS. Links
					show the selected method’s matched neighbors; they are similarity relations, not causal
					connections. Projection diagnostics describe global 6-neighbor geometry.
				</p>
				<details class="contrast-definition">
					<summary>What is a query-effect contrast?</summary>
					<p>
						For each assignment and answer value, let Eₐ, Eᵦ and E𝒸 be the raw intervention effects
						under the three queries.
					</p>
					<code>h₁ = (Eᵦ − Eₐ) / √2<br />h₂ = (2E𝒸 − Eₐ − Eᵦ) / √6</code>
					<p>
						These two coordinates remove the component shared by all three queries and preserve
						their centered variation. Concatenate across calibration assignments and answer values,
						then normalize each unit’s vector for directional similarity. PCA subsequently centers
						the cloud across units.
					</p>
				</details>
				{#if map?.boundaryDegenerate}<p class="map-warning">
						The third and fourth PCA components are tied; this 3D subspace is not unique.
					</p>{/if}
			</section>

			<section class="scores-panel panel">
				<div class="panel-heading">
					<div>
						<Icon name="chart" size={14} />
						<h2>Does similarity transfer?</h2>
					</div>
					<div class="segmented" aria-label="Score cohort">
						<button
							class:active={scope === null}
							aria-pressed={scope === null}
							onclick={() => (scope = null)}>All</button
						>{#each [0, 1] as layer (layer)}<button
								class:active={scope === layer}
								aria-pressed={scope === layer}
								onclick={() => (scope = layer)}>L{layer + 1}</button
							>{/each}
					</div>
				</div>
				<div class="score-content">
					<p class="score-intro">
						Mean cosine of held-out query-effect contrasts. Every method is scored on the same focal
						units.
					</p>
					<div class="cohort">
						<strong>{summary?.scoredUnits ?? 0}<small> / {summary?.totalUnits ?? 0}</small></strong
						><span>scored focal units</span><span
							>Matched random <b>{value(summary?.randomMeanScore)}</b></span
						>
					</div>
					<div class="table-scroll">
						<table class="score-table">
							<caption class="sr-only">Held-out neighborhood comparison</caption><thead
								><tr
									><th>Calibration method</th><th>Cosine</th><th
										title="Mean held-out cosine minus exact matched-pool random expectation"
										>Δ random</th
									><th title="Same score after shuffling held-out unit identities within layers"
										>Shuffled Δ</th
									></tr
								></thead
							><tbody
								>{#each QUERY_METHODS as entry (entry.id)}<tr class:chosen={method === entry.id}
										><th
											><button
												onclick={() => (method = entry.id)}
												aria-pressed={method === entry.id}
												><i></i>{entry.label}{#if entry.primary}<span class="primary-label"
														>primary</span
													>{/if}</button
											></th
										><td>{value(summary?.methods[entry.id].meanScore)}</td><td
											>{signed(summary?.methods[entry.id].meanDelta)}</td
										><td>{signed(summary?.methods[entry.id].meanShuffledDelta)}</td></tr
									>{/each}<tr class="control-row"
									><th>Matched random expectation</th><td>{value(summary?.randomMeanScore)}</td><td
										>—</td
									><td>—</td></tr
								></tbody
							>
						</table>
					</div>
					<div class="selection-caption">
						<span
							>Viewing {QUERY_METHODS.find(
								(entry) => entry.id === method
							)?.label.toLowerCase()}</span
						><span>Resolved test neighbors {rate(summary?.methods[method].meanCoverage)}</span>
					</div>
					<p class="score-note">
						Positive Δ means more similar than the strength-matched random expectation; negative Δ
						means less similar. These are descriptive comparisons, not confidence intervals or a
						test of predicting new outputs.
					</p>
					<details class="audit">
						<summary>Eligibility and controls</summary>
						<dl>
							<div>
								<dt>Calibration directions unresolved</dt>
								<dd>{summary?.calibrationUnresolvedUnits}</dd>
							</div>
							<div>
								<dt>Insufficient matched candidates</dt>
								<dd>{summary?.insufficientCandidateUnits}</dd>
							</div>
							<div>
								<dt>Held-out focal direction unresolved</dt>
								<dd>{summary?.testUnresolvedFocals}</dd>
							</div>
							<div>
								<dt>Matched random test coverage</dt>
								<dd>{rate(summary?.randomMeanCoverage)}</dd>
							</div>
							<div>
								<dt>Shuffled random expectation</dt>
								<dd>{value(summary?.shuffledRandomMeanScore)}</dd>
							</div>
						</dl>
						<p>
							Candidate strength is calibration full-effect RMS. Matching is approximate; no maximum
							strength ratio is enforced. Ranking, directions and candidate eligibility are fixed
							using calibration measurements. The numerical RMS floor is {analysis.design.rmsFloor.toExponential(
								0
							)}. Unresolved test neighbors contribute zero in the fixed six-neighbor denominator;
							coverage reports how many have defined directions. Random expectation averages the
							same cosine-or-zero score over the entire candidate pool. The single deterministic
							shuffle reassigns held-out identities within each layer and resolution stratum,
							preserving coverage. It is a descriptive control, not a p-value.
						</p>
					</details>
				</div>
			</section>
		</div>

		<section class="unit-panel panel">
			<div class="panel-heading">
				<div>
					<Icon name="target" size={14} />
					<h2>One unit, auditable neighbors</h2>
				</div>
				<div class="unit-stepper">
					<button aria-label="Previous query-study unit" onclick={() => selectRelative(-1)}
						><Icon name="left" size={13} /></button
					><label for="query-unit">Atlas ID</label><input
						id="query-unit"
						type="number"
						min="0"
						max="255"
						value={selected ?? ''}
						placeholder="0–255"
						aria-label="Query-study unit ID"
						oninput={(event) => {
							const next = event.currentTarget.valueAsNumber;
							if (Number.isInteger(next) && next >= 0 && next < 256) onselect(next);
						}}
					/><button aria-label="Next query-study unit" onclick={() => selectRelative(1)}
						><Icon name="right" size={13} /></button
					>
				</div>
			</div>
			{#if unit}
				<div class="unit-content">
					<div class="unit-address">
						<strong style:color={unit.layer === 0 ? 'var(--layer-1)' : 'var(--layer-2)'}
							>{address(unit.id)}</strong
						><code>layers[{unit.layer}] · channel {unit.channel}</code><span
							>{#if unit.status === 'calibration-unresolved'}Not scored: no resolved calibration
								direction for {unit.unresolvedCalibrationMethods
									.map((id) => QUERY_METHODS.find((entry) => entry.id === id)?.label ?? id)
									.join(', ')}. The common cohort requires all four methods above RMS {analysis.design.rmsFloor.toExponential(
									0
								)}.{:else}{statusLabels[unit.status]}{/if}</span
						>
					</div>
					<div class="unit-stat">
						<span>Held-out cosine</span><strong>{value(methodResult?.score)}</strong><small
							>Δ matched {signed(methodResult?.delta)}</small
						>
					</div>
					<div class="unit-stat">
						<span>Matched random</span><strong>{value(unit.randomScore)}</strong><small
							>{unit.candidatePoolSize} candidate units</small
						>
					</div>
					<div class="unit-stat">
						<span>Query-effect RMS</span><strong
							>{magnitude(unit.strengths.calibrationQueryEffectRms)}</strong
						><small>Held out {magnitude(unit.strengths.testQueryEffectRms)}</small>
					</div>
					<div class="neighbor-section">
						<span
							>{QUERY_METHODS.find((entry) => entry.id === method)?.label} · chosen on calibration</span
						>
						<div class="neighbor-chips">
							{#each methodResult?.neighbors ?? [] as id (id)}<button
									onclick={() => onselect(id)}
									title={`Inspect ${address(id)}, atlas ID ${id}`}
									>{id}<small>{address(id)}</small></button
								>{:else}<span class="subtle">No defined neighborhood for this unit.</span>{/each}
						</div>
					</div>
				</div>
				<details class="pool-details">
					<summary>Inspect the shared candidate pool and measurement checks</summary>
					<div class="pool-body">
						<div>
							<p>
								Same-layer candidates nearest in log calibration full-effect RMS. Every method
								chooses from this same pool.
							</p>
							<div class="pool-chips">
								{#each unit.candidatePool as id (id)}<button
										onclick={() => onselect(id)}
										title={address(id)}>{id}</button
									>{/each}
							</div>
						</div>
						<dl>
							<div>
								<dt>Calibration full-effect RMS</dt>
								<dd>{magnitude(unit.strengths.calibrationFullEffectRms)}</dd>
							</div>
							<div>
								<dt>Candidate full-effect RMS range</dt>
								<dd>{magnitude(candidateMinimum)} – {magnitude(candidateMaximum)}</dd>
							</div>
							<div>
								<dt>Maximum symmetric strength ratio to focal</dt>
								<dd>{value(maximumStrengthRatio, 2)}{maximumStrengthRatio === null ? '' : '×'}</dd>
							</div>
							<div>
								<dt>Selected-neighbor coverage</dt>
								<dd>
									{methodResult?.resolvedTestNeighbors ?? 0} / {methodResult?.neighbors.length ?? 0}
								</dd>
							</div>
							<div>
								<dt>Prefix activation max difference</dt>
								<dd>
									{magnitude(
										Math.max(
											measurement.splits.calibration.prefixMaxDifference,
											measurement.splits.test.prefixMaxDifference
										)
									)}
								</dd>
							</div>
							<div>
								<dt>Probability mass max error</dt>
								<dd>
									{magnitude(
										Math.max(
											measurement.splits.calibration.probabilityMassMaxError,
											measurement.splits.test.probabilityMassMaxError
										)
									)}
								</dd>
							</div>
							<div>
								<dt>Checkpoint preserved</dt>
								<dd>{measurement.checkpointPreserved ? 'Verified exact' : 'Unverified'}</dd>
							</div>
						</dl>
					</div>
				</details>
			{:else}<div class="unit-empty">
					<Icon name="target" size={16} /><span
						>Select a point or enter an atlas ID to inspect its raw effects and matched neighbors.</span
					>
				</div>{/if}
		</section>
		<QueryProbePanel {measurement} {selected} />
		<footer class="study-footer">
			<span>Single checkpoint · single seed · exploratory result</span><span
				>{measurement.groups.calibration.length * 3} calibration + {measurement.groups.test.length *
					3} held-out prompts · {value(measurement.elapsedMs / 1000, 1)} s measurement</span
			>
		</footer>
	{:else}
		<section class="empty-study panel">
			<div class="empty-symbol"><Icon name="network" size={32} /></div>
			<span class="eyebrow">A hypothesis ready to test</span>
			<h2>Do query-sensitive effects define useful neighbors?</h2>
			<p>
				Measure all 256 MLP channels under three queries for each fixed assignment. Choose
				neighborhoods on calibration data, then compare their effect similarity on held-out
				assignments.
			</p>
			<div class="empty-protocol">
				<span><Icon name="target" size={15} />Single-channel ablations</span><span
					><Icon name="layers" size={15} />Shared candidate pools</span
				><span><Icon name="chart" size={15} />Exact matched random control</span>
			</div>
			<p class="empty-note">
				Start from the current checkpoint or open a recorded reference. No measurement is displayed
				until it has completed.
			</p>
		</section>
	{/if}
</div>

<style>
	.query-study {
		padding: 24px;
		max-width: 1800px;
		margin: 0 auto;
		color: var(--ink);
	}
	h1,
	h2,
	p {
		margin: 0;
	}
	h1 {
		font-size: 25px;
		font-weight: 520;
		letter-spacing: -0.7px;
		line-height: 1.25;
		margin: 7px 0 9px;
	}
	h2 {
		font-size: 12px;
		font-weight: 550;
	}
	.study-heading {
		display: flex;
		align-items: flex-start;
		justify-content: space-between;
		gap: 24px;
		margin-bottom: 20px;
	}
	.heading-copy {
		max-width: 690px;
	}
	.eyebrow {
		font: 12px var(--mono);
		text-transform: uppercase;
		letter-spacing: 1.3px;
		color: var(--muted);
	}
	.heading-copy p {
		color: var(--muted);
		font-size: 12px;
		line-height: 1.65;
		max-width: 650px;
	}
	.study-actions {
		display: flex;
		flex-direction: column;
		align-items: flex-end;
		gap: 8px;
		flex-shrink: 0;
	}
	.study-actions > button {
		display: flex;
		align-items: center;
		gap: 8px;
		min-height: 34px;
		padding: 8px 12px;
		font-size: 13px;
	}
	.secondary-actions {
		display: flex;
		gap: 6px;
	}
	.secondary-actions button {
		border: 1px solid var(--line);
		border-radius: 5px;
		padding: 6px 10px;
		background: var(--surface);
		color: var(--muted);
		font-size: 12px;
	}
	.secondary-actions button:hover {
		border-color: var(--faint);
		color: var(--ink);
	}
	.secondary-actions .icon-button {
		display: flex;
		align-items: center;
		justify-content: center;
		padding: 6px;
		width: 29px;
	}
	.resident {
		color: var(--muted);
		font: 12px var(--mono);
	}
	.protocol-strip {
		display: grid;
		grid-template-columns: 1fr 1.2fr 1.1fr;
		gap: 14px;
		margin-bottom: 14px;
	}
	.protocol-strip > div {
		display: flex;
		align-items: flex-start;
		gap: 10px;
		border: 1px solid var(--line);
		border-radius: 6px;
		padding: 13px 14px;
		background: var(--surface);
	}
	.protocol-number {
		color: var(--faint);
		font: 12px var(--mono);
		padding-top: 2px;
	}
	.protocol-strip p {
		display: flex;
		flex-direction: column;
		gap: 5px;
	}
	.protocol-strip strong {
		font-size: 13px;
		font-weight: 550;
	}
	.protocol-strip p span {
		font-size: 12px;
		line-height: 1.5;
		color: var(--muted);
	}
	.measurement-progress {
		margin-bottom: 14px;
		border: 1px solid color-mix(in srgb, var(--accent) 45%, var(--line));
		border-radius: 6px;
		padding: 12px 15px;
		background: var(--surface);
	}
	.measurement-progress > div {
		display: flex;
		gap: 8px;
		align-items: center;
		font-size: 13px;
	}
	.progress-count {
		margin-left: auto;
		font: 12px var(--mono);
		color: var(--muted);
	}
	progress {
		width: 100%;
		height: 4px;
		margin: 10px 0;
		accent-color: var(--accent);
	}
	.measurement-progress p {
		margin-top: 8px;
		font-size: 12px;
		color: var(--muted);
		line-height: 1.5;
	}
	.provenance {
		display: flex;
		flex-wrap: wrap;
		gap: 10px 17px;
		padding: 6px 1px 14px;
		color: var(--muted);
		font: 12px var(--mono);
		align-items: center;
	}
	.source {
		color: var(--accent);
		display: flex;
		align-items: center;
		gap: 6px;
	}
	.provenance .different {
		color: var(--warning);
		margin-left: auto;
	}
	.panel {
		border: 1px solid var(--line);
		border-radius: 7px;
		background: var(--surface);
		min-width: 0;
		overflow: hidden;
	}
	.research-grid {
		display: grid;
		grid-template-columns: minmax(0, 1.25fr) minmax(440px, 1fr);
		gap: 14px;
		margin-bottom: 14px;
	}
	.panel-heading {
		min-height: 46px;
		padding: 10px 14px;
		border-bottom: 1px solid var(--line);
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 10px;
	}
	.panel-heading > div:first-child {
		display: flex;
		align-items: center;
		gap: 7px;
	}
	.small-label {
		color: var(--muted);
		font: 12px var(--mono);
	}
	.map-view {
		height: 320px;
	}
	.map-diagnostics {
		display: flex;
		gap: 12px;
		align-items: center;
		padding: 10px 14px;
		border-top: 1px solid var(--line);
		flex-wrap: wrap;
		font: 12px var(--mono);
		color: var(--muted);
	}
	.map-diagnostics span {
		display: flex;
		align-items: center;
		gap: 4px;
	}
	.map-diagnostics span:nth-child(4) {
		margin-left: auto;
	}
	.map-diagnostics i {
		width: 4px;
		height: 4px;
		border-radius: 50%;
	}
	.map-diagnostics strong {
		font-weight: 450;
		color: var(--ink);
	}
	.map-note,
	.map-warning {
		color: var(--muted);
		font-size: 12px;
		line-height: 1.55;
		padding: 0 14px 12px;
	}
	.contrast-definition {
		margin: 0 14px 12px;
	}
	.contrast-definition p {
		font-size: 12px;
		line-height: 1.6;
		color: var(--muted);
		padding: 6px 0;
	}
	.contrast-definition code {
		display: block;
		font: 12px/1.8 var(--mono);
		padding: 5px 0;
	}
	.map-warning {
		color: var(--warning);
	}
	.segmented {
		display: flex;
		border: 1px solid var(--line);
		border-radius: 5px;
		padding: 2px;
		gap: 2px;
	}
	.segmented button {
		border: 0;
		border-radius: 3px;
		background: transparent;
		color: var(--muted);
		font-size: 12px;
		padding: 4px 7px;
	}
	.segmented button.active {
		background: var(--surface-hover);
		color: var(--ink);
	}
	.score-content {
		padding: 15px 16px;
	}
	.score-intro {
		font-size: 13px;
		color: var(--muted);
		line-height: 1.55;
	}
	.cohort {
		display: flex;
		align-items: baseline;
		gap: 8px;
		margin: 15px 0;
	}
	.cohort > strong {
		font: 23px var(--mono);
		letter-spacing: -1px;
	}
	.cohort small {
		font-size: 12px;
		color: var(--muted);
	}
	.cohort > span {
		color: var(--muted);
		font-size: 12px;
	}
	.cohort > span:last-child {
		margin-left: auto;
		display: flex;
		flex-direction: column;
		gap: 4px;
		align-items: flex-end;
	}
	.cohort b {
		color: var(--ink);
		font: 15px var(--mono);
	}
	.table-scroll {
		overflow-x: auto;
	}
	.score-table {
		width: 100%;
		border-collapse: collapse;
		font-size: 12px;
		white-space: nowrap;
	}
	.score-table th {
		text-align: left;
		font-weight: 400;
	}
	.score-table thead th {
		padding: 8px 6px;
		border-bottom: 1px solid var(--line);
		color: var(--muted);
		font-size: 12px;
	}
	.score-table td {
		text-align: right;
		font: 13px var(--mono);
		padding: 12px 6px;
	}
	.score-table thead th:not(:first-child) {
		text-align: right;
	}
	.score-table tbody tr {
		border-bottom: 1px solid color-mix(in srgb, var(--line) 70%, transparent);
	}
	.score-table tbody th button {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 10px 4px;
		border: 0;
		background: transparent;
		color: var(--muted);
		width: 100%;
		text-align: left;
		font-size: 12px;
	}
	.score-table th button i {
		width: 5px;
		height: 5px;
		border: 1px solid var(--faint);
		border-radius: 50%;
	}
	.score-table .chosen th button {
		color: var(--ink);
	}
	.score-table .chosen th button i {
		background: var(--accent);
		border-color: var(--accent);
	}
	.primary-label {
		color: var(--muted);
		font: 12px var(--mono);
		border: 1px solid var(--line);
		border-radius: 3px;
		padding: 2px 3px;
	}
	.control-row {
		color: var(--muted);
	}
	.control-row th {
		padding: 11px 4px;
		font-size: 12px;
	}
	.selection-caption {
		display: flex;
		justify-content: space-between;
		gap: 8px;
		color: var(--muted);
		font: 12px var(--mono);
		padding: 10px 0;
	}
	.score-note {
		font-size: 12px;
		color: var(--muted);
		line-height: 1.6;
		margin: 3px 0 10px;
	}
	summary {
		cursor: pointer;
		font-size: 12px;
		color: var(--muted);
		padding: 6px 0;
	}
	.audit p {
		font-size: 12px;
		color: var(--muted);
		line-height: 1.6;
		margin-top: 10px;
	}
	dl {
		margin: 7px 0;
		font-size: 12px;
	}
	dl > div {
		display: flex;
		justify-content: space-between;
		gap: 15px;
		margin: 7px 0;
	}
	dt {
		color: var(--muted);
	}
	dd {
		margin: 0;
		font-family: var(--mono);
	}
	.unit-panel {
		margin-bottom: 14px;
	}
	.unit-stepper {
		display: flex;
		align-items: center;
		gap: 7px;
	}
	.unit-stepper label {
		font: 12px var(--mono);
		color: var(--muted);
	}
	.unit-stepper button {
		border: 0;
		background: transparent;
		display: flex;
		padding: 4px;
		border-radius: 3px;
	}
	.unit-stepper button:hover {
		background: var(--surface-hover);
	}
	.unit-stepper input {
		width: 57px;
		padding: 5px 6px;
		text-align: center;
		font: 12px var(--mono);
	}
	.unit-content {
		display: grid;
		grid-template-columns: 1.2fr 0.7fr 0.7fr 0.85fr 1.7fr;
		gap: 18px;
		padding: 17px;
		align-items: start;
	}
	.unit-address,
	.unit-stat,
	.neighbor-section {
		display: flex;
		flex-direction: column;
		gap: 7px;
	}
	.unit-address > strong {
		font: 16px var(--mono);
	}
	.unit-address code {
		color: var(--muted);
		font: 12px var(--mono);
	}
	.unit-address > span,
	.unit-stat span,
	.neighbor-section > span {
		color: var(--muted);
		font-size: 12px;
		line-height: 1.5;
	}
	.unit-stat strong {
		font: 16px var(--mono);
		font-weight: 400;
	}
	.unit-stat small {
		color: var(--muted);
		font: 12px var(--mono);
	}
	.neighbor-chips {
		display: flex;
		gap: 5px;
		flex-wrap: wrap;
	}
	.neighbor-chips button {
		display: flex;
		flex-direction: column;
		gap: 4px;
		border: 1px solid color-mix(in srgb, var(--accent) 30%, var(--line));
		border-radius: 4px;
		background: transparent;
		padding: 6px;
		font: 12px var(--mono);
	}
	.neighbor-chips small {
		color: var(--muted);
		font-size: 12px;
	}
	.neighbor-chips button:hover,
	.pool-chips button:hover {
		background: var(--surface-hover);
		border-color: var(--accent);
	}
	.subtle {
		font-size: 12px;
		color: var(--muted);
	}
	.pool-details {
		margin: 0 17px 12px;
	}
	.pool-body {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 25px;
		padding: 6px 0;
	}
	.pool-body p {
		color: var(--muted);
		font-size: 12px;
		line-height: 1.6;
	}
	.pool-chips {
		display: flex;
		gap: 5px;
		flex-wrap: wrap;
		margin-top: 10px;
	}
	.pool-chips button {
		border: 1px solid var(--line);
		border-radius: 4px;
		background: transparent;
		padding: 4px 6px;
		font: 12px var(--mono);
	}
	.unit-empty {
		padding: 24px 17px;
		display: flex;
		gap: 10px;
		color: var(--muted);
		font-size: 13px;
		align-items: center;
	}
	.study-footer {
		display: flex;
		justify-content: space-between;
		gap: 12px;
		margin-top: 15px;
		color: var(--faint);
		font: 12px var(--mono);
	}
	.empty-study {
		padding: 65px 24px;
		text-align: center;
		min-height: 390px;
		display: flex;
		flex-direction: column;
		align-items: center;
	}
	.empty-symbol {
		color: var(--accent);
		border: 1px solid color-mix(in srgb, var(--accent) 35%, var(--line));
		border-radius: 12px;
		width: 65px;
		height: 65px;
		display: flex;
		align-items: center;
		justify-content: center;
		margin-bottom: 23px;
		background: color-mix(in srgb, var(--accent) 4%, transparent);
	}
	.empty-study h2 {
		font-size: 21px;
		font-weight: 450;
		margin: 10px 0 14px;
		letter-spacing: -0.4px;
	}
	.empty-study > p {
		max-width: 620px;
		color: var(--muted);
		font-size: 12px;
		line-height: 1.7;
	}
	.empty-protocol {
		display: flex;
		gap: 23px;
		flex-wrap: wrap;
		justify-content: center;
		margin: 24px 0;
	}
	.empty-protocol > span {
		display: flex;
		align-items: center;
		gap: 7px;
		font: 12px var(--mono);
		color: var(--muted);
	}
	.empty-study p.empty-note {
		font-size: 12px;
		color: var(--faint);
	}
	.sr-only {
		position: absolute;
		width: 1px;
		height: 1px;
		overflow: hidden;
		clip-path: inset(50%);
	}
	@media (max-width: 1200px) {
		.query-study {
			padding: 18px;
		}
		.research-grid {
			grid-template-columns: minmax(0, 1fr) minmax(420px, 1fr);
		}
		.unit-content {
			grid-template-columns: 1.2fr 1fr 1fr 1fr;
		}
		.neighbor-section {
			grid-column: 1/-1;
		}
		.study-heading {
			gap: 16px;
		}
		h1 {
			font-size: 22px;
		}
		.protocol-strip {
			gap: 10px;
		}
	}
	@media (max-width: 960px) {
		.research-grid {
			grid-template-columns: 1fr;
		}
		.map-view {
			height: 370px;
		}
		.study-heading {
			flex-direction: column;
		}
		.study-actions {
			align-items: flex-start;
			flex-direction: row;
			flex-wrap: wrap;
		}
		.resident {
			width: 100%;
		}
		.protocol-strip {
			grid-template-columns: 1fr;
			gap: 7px;
		}
		.protocol-strip > div {
			padding: 10px 12px;
		}
		.protocol-strip p {
			flex-direction: row;
			gap: 12px;
			align-items: baseline;
		}
		.study-footer {
			flex-wrap: wrap;
		}
	}
	@media (max-width: 600px) {
		.query-study {
			padding: 14px 10px;
		}
		h1 {
			font-size: 22px;
		}
		.heading-copy p {
			font-size: 13px;
		}
		.protocol-strip p {
			flex-direction: column;
			gap: 4px;
		}
		.unit-content {
			grid-template-columns: 1fr 1fr;
			gap: 20px;
		}
		.unit-address {
			grid-column: 1/-1;
		}
		.unit-stepper {
			gap: 4px;
		}
		.panel-heading {
			flex-wrap: wrap;
		}
		.small-label {
			font-size: 12px;
		}
		.pool-body {
			grid-template-columns: 1fr;
			gap: 10px;
		}
		.map-diagnostics {
			gap: 9px;
			font-size: 12px;
		}
		.map-diagnostics span:nth-child(4) {
			margin-left: 0;
		}
		.selection-caption {
			flex-wrap: wrap;
		}
		.score-content {
			padding: 13px 10px;
		}
		.score-table td {
			font-size: 12px;
			padding: 11px 4px;
		}
		.score-table thead th {
			font-size: 12px;
			padding: 8px 4px;
		}
		.score-table tbody th button {
			font-size: 12px;
		}
		.primary-label {
			display: none;
		}
		.empty-study {
			padding: 40px 16px;
		}
		.empty-study h2 {
			font-size: 19px;
		}
		.provenance .different {
			margin-left: 0;
		}
	}

	.query-study {
		max-width: 1600px;
		margin: auto;
		padding: 32px 24px;
	}
	.study-heading {
		gap: 32px;
		margin-bottom: 24px;
	}
	.heading-copy p {
		font-size: 15px;
		max-width: 640px;
		line-height: 1.7;
	}
	.protocol-disclosure {
		margin-bottom: 24px;
		border: 1px solid var(--line);
		border-radius: 10px;
	}
	.protocol-strip {
		padding: 16px;
		margin: 0;
	}
	.research-grid {
		gap: 20px;
		margin-bottom: 24px;
		grid-template-columns: minmax(0, 1fr) minmax(440px, 0.9fr);
	}
	.panel {
		border-radius: 12px;
	}
	.panel-heading {
		padding: 16px 18px;
		flex-wrap: wrap;
	}
	.map-view {
		height: 440px;
	}
	.score-content {
		padding: 20px;
	}
	.score-intro,
	.map-note,
	.map-warning {
		font-size: 14px;
		line-height: 1.65;
	}
	.segmented button {
		min-height: 32px;
		padding: 6px 10px;
	}
	@media (max-width: 1100px) {
		.research-grid {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	@media (max-width: 600px) {
		.query-study {
			padding: 24px 12px;
		}
		.protocol-strip {
			grid-template-columns: 1fr;
		}
		.heading-copy p {
			font-size: 14px;
		}
	}
</style>
