<script lang="ts">
	import Icon from './Icon.svelte';
	import {
		MODEL_CONFIG as cfg,
		NEURON_COUNT,
		type Checkpoint,
		type Probe
	} from '$lib/lab/protocol';
	import {
		inspectNeuron,
		inspectActivations,
		type NeuronInspection,
		type ActivationProfile
	} from '$lib/lab/model-inspection';
	interface Props {
		selected: number | null;
		checkpoint: Checkpoint | undefined;
		probe: Probe | null | undefined;
		token: number;
		onselect?: (id: number) => void;
		theme?: 'dark' | 'light';
		compact?: boolean;
	}
	let {
		selected,
		checkpoint,
		probe,
		token,
		onselect,
		theme = 'dark',
		compact = false
	}: Props = $props();
	let parameters = $derived.by((): { unit: NeuronInspection | null; error: string } => {
		if (selected === null || !checkpoint) return { unit: null, error: '' };
		try {
			return { unit: inspectNeuron(checkpoint, selected), error: '' };
		} catch (reason) {
			return { unit: null, error: reason instanceof Error ? reason.message : String(reason) };
		}
	});
	let activation = $derived.by((): { profile: ActivationProfile | null; error: string } => {
		if (selected === null || !probe) return { profile: null, error: '' };
		try {
			return { profile: inspectActivations(probe, selected), error: '' };
		} catch (reason) {
			return { profile: null, error: reason instanceof Error ? reason.message : String(reason) };
		}
	});
	let unit = $derived(parameters.unit);
	let activity = $derived(activation.profile);
	let profiles = $derived(
		unit
			? [
					{ ...unit.incoming, label: 'Incoming', description: '32 residual channels → this unit' },
					{ ...unit.outgoing, label: 'Outgoing', description: 'This unit → 32 residual channels' }
				]
			: []
	);
	let selectedLayer = $derived(selected === null ? 0 : Math.floor(selected / cfg.hidden));
	let selectedChannel = $derived(selected === null ? 0 : selected % cfg.hidden);
	let currentActivation = $derived(
		activity && token >= 0 && token < activity.values.length ? activity.values[token] : null
	);
	let mismatchedSteps = $derived(unit && activity && unit.step !== activity.step);
	function number(value: number | null | undefined, digits = 3) {
		if (value === null || value === undefined) return '—';
		if (value !== 0 && Math.abs(value) < 0.001) return value.toExponential(2);
		return value.toFixed(digits);
	}
	function zeroCenteredHeight(value: number, scale: number) {
		return scale > 0 ? Math.abs(value / scale) * 29 : 0;
	}
</script>

<section
	class="inspector"
	class:compact
	data-theme={theme}
	style:--unit-color={selectedLayer === 0 ? 'var(--layer-1)' : 'var(--layer-2)'}
	aria-label="Selected neuron parameters and activation"
>
	{#if !compact}<header>
			<span class="section-label">Unit inspector</span><span class="mono meta">MLP / post-ReLU</span
			>
		</header>{/if}
	{#if selected === null}
		<div class="empty">
			<div class="empty-symbol" aria-hidden="true"><Icon name="target" size={25} /></div>
			<strong>Select a neuron</strong>
			<p>Inspect its place in the model, both weight vectors, and its response at every token.</p>
			<span class="mono">256 channels · 64 weights per channel</span>
		</div>
	{:else}
		<div class="identity">
			<div>
				<span class="unit-code"
					>L{selectedLayer + 1}<span> / </span>{String(selectedChannel).padStart(3, '0')}</span
				><span class="meta mono">atlas index {selected}</span>
			</div>
			{#if onselect && !compact}<div class="selection-controls">
					<button
						type="button"
						aria-label="Select previous neuron"
						disabled={selected <= 0}
						onclick={() => selected !== null && onselect?.(selected - 1)}>←</button
					><button
						type="button"
						aria-label="Select next neuron"
						disabled={selected >= NEURON_COUNT - 1}
						onclick={() => selected !== null && onselect?.(selected + 1)}>→</button
					>
				</div>{/if}
		</div>
		<div class="observation-stamps">
			<span>Weights <b class="mono">{unit ? `step ${unit.step}` : 'unavailable'}</b></span><span
				>Activity <b class="mono">{activity ? `step ${activity.step}` : 'unavailable'}</b></span
			>
		</div>
		{#if mismatchedSteps}<p class="notice">
				These measurements are from different checkpoints. The activation trace is historical; the
				weights are from step {unit?.step}.
			</p>{/if}
		{#if parameters.error || activation.error}<p class="notice error" role="status">
				{parameters.error || activation.error}
			</p>{/if}
		<div class="activation-section">
			<div class="subheading">
				<strong>Response across the prompt</strong><span class="mono current-value"
					>{number(currentActivation)}</span
				>
			</div>
			{#if activity}
				<div
					class="activation-chart"
					role="img"
					aria-label={`Neuron ${selected} activation at each token; maximum ${number(activity.statistics.max)}. Selected token ${token + 1} activation ${number(currentActivation)}.`}
				>
					{#each activity.values as value, index (index)}<div
							class:current={index === token}
							title={`Token ${index + 1} (${activity.tokens[index]}): ${value}`}
						>
							<span class="activation-column"
								><i
									style:height={`${activity.statistics.max > 0 ? (value / activity.statistics.max) * 100 : 0}%`}
								></i></span
							><span class="token-text">{activity.tokens[index]}</span><span class="token-index"
								>{index + 1}</span
							>
						</div>{/each}
				</div>
				<div class="trace-footer">
					<span>max <b>{number(activity.statistics.max)}</b></span><span
						>mean <b>{number(activity.statistics.mean)}</b></span
					><span>active <b>{activity.statistics.nonzero}/{cfg.sequenceLength}</b></span>
				</div>
				<p class="trace-context">
					{#if activity.lesionNeuron === null}Intact forward pass.{:else}L{Math.floor(
							activity.lesionNeuron / cfg.hidden
						) + 1} · {activity.lesionNeuron % cfg.hidden} is silenced at every token.{/if} Post-ReLU values
					include that intervention.
				</p>
			{:else}<p class="unavailable">
					A measured probe is needed to show this unit’s activation.
				</p>{/if}
		</div>
		{#each profiles as profile (profile.label)}
			{@const extent = Math.max(Math.abs(profile.statistics.min), Math.abs(profile.statistics.max))}
			<div class="weight-section">
				<div class="subheading">
					<strong>{profile.label} weights</strong><span class="meta mono"
						>{profile.descriptor.shape.join(' × ')}</span
					>
				</div>
				<div class="tensor-address mono">
					<span>{profile.descriptor.name}</span><b>{profile.selection}</b>
				</div>
				<div class="weight-chart">
					<svg
						viewBox="0 0 320 64"
						role="img"
						aria-label={`${profile.label} weight vector over 32 residual dimensions, signed scale minus ${number(extent)} to plus ${number(extent)}`}
					>
						<line x1="0" x2="320" y1="32" y2="32" class="zero-line" />
						{#each profile.values as value, index (index)}<rect
								x={index * 10 + 1.5}
								y={value >= 0 ? 32 - zeroCenteredHeight(value, extent) : 32}
								width="7"
								height={zeroCenteredHeight(value, extent)}
								rx="1"
								class:negative={value < 0}><title>Dimension {index}: {value}</title></rect
							>{/each}
					</svg>
					<div class="weight-axis mono">
						<span>0</span><span>residual dimension</span><span>31</span>
					</div>
				</div>
				<div class="weight-statistics">
					<div><span>L₂ norm</span><b>{number(profile.statistics.l2)}</b></div>
					<div><span>RMS</span><b>{number(profile.statistics.rms)}</b></div>
					<div>
						<span>min / max</span><b
							>{number(profile.statistics.min)} / {number(profile.statistics.max)}</b
						>
					</div>
				</div>
				<p class="weight-caption">{profile.description}. Chart scale ±{number(extent)}.</p>
			</div>
		{:else}<p class="unavailable weight-empty">
				Load a complete checkpoint to inspect parameter values.
			</p>{/each}
		<details class="computation">
			<summary>How this unit contributes</summary>
			<div>
				<code>hⱼ(t) = ReLU(RMS(xₜ) · W₁[:, j]) × mⱼ</code><code>Δxₜ = hⱼ(t) × W₂[j, :]</code>
				<p>
					The MLP adds the sum of these output vectors to the residual stream. There are no MLP
					biases. Weight magnitude alone does not establish the effect on a prediction; use a
					measured intervention.
				</p>
				<p class="mono">
					j = {selectedChannel} · mⱼ = {activity?.lesionNeuron === selected ? 0 : 1}{#if !activity}
						(intact definition){/if}
				</p>
			</div>
		</details>
	{/if}
</section>

<style>
	.inspector {
		container-type: inline-size;
		--profile-negative: #be9294;
		color: var(--ink, #dae2eb);
		background: var(--surface, #111820);
		border: 1px solid var(--line, #26303b);
		border-radius: 8px;
		overflow: hidden;
		font:
			11px/1.45 'DM Sans',
			sans-serif;
	}
	.inspector[data-theme='light'] {
		--profile-negative: #a76166;
	}
	header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 11px 12px;
		gap: 8px;
		border-bottom: 1px solid var(--line, #26303b);
	}
	.section-label,
	.subheading strong {
		font-size: 11px;
		font-weight: 600;
	}
	.mono,
	.unit-code,
	.token-text,
	.token-index,
	.trace-footer,
	.weight-statistics b,
	.computation code {
		font-family: 'IBM Plex Mono', monospace;
	}
	.meta {
		color: var(--muted, #8c9aaa);
		font-size: 9px;
	}
	.empty {
		display: grid;
		justify-items: center;
		padding: 20px 18px;
		text-align: center;
		gap: 7px;
	}
	.empty-symbol {
		font-size: 25px;
		color: var(--muted, #8c9aaa);
	}
	.empty strong {
		font-size: 11px;
		font-weight: 550;
	}
	.empty p {
		max-width: 220px;
		margin: 0;
		color: var(--muted, #8c9aaa);
		font-size: 10px;
		line-height: 1.65;
	}
	.empty > span {
		font-size: 8px;
		color: var(--muted, #8c9aaa);
		margin-top: 3px;
	}
	.identity {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 8px;
		padding: 12px 12px 9px;
	}
	.identity > div:first-child {
		display: flex;
		gap: 12px;
		align-items: baseline;
	}
	.unit-code {
		font-size: 18px;
		font-weight: 500;
		color: var(--unit-color, #a7b8d4);
		white-space: nowrap;
	}
	.unit-code > span {
		color: var(--line, #26303b);
		font-weight: 400;
	}
	.selection-controls {
		display: flex;
		gap: 4px;
	}
	.selection-controls button {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 25px;
		height: 23px;
		border: 1px solid var(--line, #26303b);
		border-radius: 4px;
		background: var(--surface-raised, #161f29);
		color: var(--muted, #8c9aaa);
		font:
			11px 'IBM Plex Mono',
			monospace;
		cursor: pointer;
	}
	.selection-controls button:disabled {
		opacity: 0.3;
		cursor: default;
	}
	button:focus-visible,
	summary:focus-visible {
		outline: 2px solid var(--accent, #87bda7);
		outline-offset: -2px;
	}
	.observation-stamps {
		display: flex;
		gap: 14px;
		padding: 0 12px 10px;
		color: var(--muted, #8c9aaa);
		font-size: 9px;
	}
	.observation-stamps b {
		font-size: 9px;
		color: var(--ink, #dae2eb);
		font-weight: 400;
		margin-left: 4px;
	}
	.notice {
		margin: 0;
		padding: 8px 12px;
		border-top: 1px solid var(--line, #26303b);
		color: var(--muted, #8c9aaa);
		background: var(--surface-raised, #161f29);
		font-size: 9px;
		line-height: 1.6;
	}
	.notice.error {
		color: var(--profile-negative);
	}
	.activation-section,
	.weight-section {
		padding: 10px 12px;
		border-top: 1px solid var(--line, #26303b);
	}
	.subheading {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 8px;
		margin-bottom: 8px;
	}
	.current-value {
		color: var(--unit-color, #a7b8d4);
		font-size: 15px;
	}
	.activation-chart {
		display: grid;
		grid-template-columns: repeat(14, minmax(0, 1fr));
		gap: 3px;
	}
	.activation-chart > div {
		display: flex;
		flex-direction: column;
		align-items: stretch;
		text-align: center;
		gap: 2px;
	}
	.activation-column {
		position: relative;
		display: block;
		height: 44px;
		border-bottom: 1px solid var(--line, #26303b);
		background: var(--surface-raised, #161f29);
		border-radius: 2px 2px 0 0;
	}
	.activation-column i {
		position: absolute;
		display: block;
		width: 100%;
		bottom: 0;
		background: color-mix(in srgb, var(--unit-color, #a7b8d4) 55%, transparent);
		border-radius: 2px 2px 0 0;
	}
	.current .activation-column {
		outline: 1px solid var(--unit-color, #a7b8d4);
		outline-offset: 1px;
	}
	.current .activation-column i {
		background: var(--unit-color, #a7b8d4);
	}
	.token-text {
		font-size: 9px;
		color: var(--muted, #8c9aaa);
	}
	.current .token-text {
		color: var(--unit-color, #a7b8d4);
	}
	.token-index {
		font-size: 7px;
		color: var(--muted, #8c9aaa);
		opacity: 0.6;
	}
	.trace-footer {
		display: flex;
		gap: 12px;
		flex-wrap: wrap;
		margin-top: 8px;
		color: var(--muted, #8c9aaa);
		font-size: 8px;
	}
	.trace-footer b {
		font-weight: 400;
		color: var(--ink, #dae2eb);
	}
	.trace-context,
	.unavailable {
		color: var(--muted, #8c9aaa);
		font-size: 9px;
		line-height: 1.6;
		margin: 7px 0 0;
	}
	.tensor-address {
		font-size: 9px;
		display: flex;
		justify-content: space-between;
		gap: 5px;
	}
	.tensor-address span {
		color: var(--muted, #8c9aaa);
	}
	.tensor-address b {
		color: var(--unit-color, #a7b8d4);
		font-weight: 400;
		white-space: nowrap;
	}
	.weight-chart {
		margin: 6px 0 8px;
	}
	.weight-chart svg {
		display: block;
		width: 100%;
		height: 56px;
		overflow: visible;
	}
	.weight-chart rect {
		fill: var(--unit-color, #a7b8d4);
		opacity: 0.78;
	}
	.weight-chart rect.negative {
		fill: var(--profile-negative);
	}
	.zero-line {
		stroke: var(--line, #26303b);
		stroke-width: 1;
	}
	.weight-axis {
		display: flex;
		justify-content: space-between;
		color: var(--muted, #8c9aaa);
		font-size: 7px;
	}
	.weight-axis span:nth-child(2) {
		opacity: 0.8;
	}
	.weight-statistics {
		display: grid;
		grid-template-columns: 1fr 1fr 1.7fr;
		gap: 6px;
	}
	.weight-statistics > div {
		display: grid;
		gap: 3px;
	}
	.weight-statistics span {
		color: var(--muted, #8c9aaa);
		font-size: 8px;
	}
	.weight-statistics b {
		font-size: 9px;
		font-weight: 400;
		white-space: nowrap;
	}
	.weight-caption {
		margin: 7px 0 0;
		font-size: 8px;
		color: var(--muted, #8c9aaa);
	}
	.weight-empty {
		padding: 8px 12px 13px;
		border-top: 1px solid var(--line, #26303b);
	}
	.computation {
		border-top: 1px solid var(--line, #26303b);
	}
	.computation summary {
		cursor: pointer;
		padding: 9px 12px;
		color: var(--muted, #8c9aaa);
		font-size: 9px;
	}
	.computation > div {
		padding: 0 12px 10px;
	}
	.computation code {
		display: block;
		font-size: 8px;
		margin-bottom: 5px;
		color: var(--ink, #dae2eb);
		white-space: normal;
	}
	.computation p {
		margin: 7px 0 0;
		color: var(--muted, #8c9aaa);
		font-size: 9px;
		line-height: 1.6;
	}
	.compact {
		border: 0;
		border-radius: 0;
		background: transparent;
	}
	.compact .identity {
		padding: 8px 9px 5px;
	}
	.compact .unit-code {
		font-size: 15px;
	}
	.compact .observation-stamps {
		padding: 0 9px 7px;
		gap: 10px;
	}
	.compact .activation-section,
	.compact .weight-section {
		padding: 8px 9px;
	}
	.compact .subheading {
		margin-bottom: 6px;
	}
	.compact .activation-column {
		height: 37px;
	}
	.compact .weight-chart {
		margin: 5px 0 6px;
	}
	.compact .weight-chart svg {
		height: 46px;
	}
	.compact .trace-context {
		font-size: 8px;
	}
	.compact .computation summary {
		padding: 8px 9px;
	}
	@container (max-width: 260px) {
		.weight-statistics {
			grid-template-columns: repeat(2, minmax(0, 1fr));
			row-gap: 7px;
		}
		.weight-statistics > div:last-child {
			grid-column: 1 / -1;
		}
		.observation-stamps {
			flex-wrap: wrap;
			gap: 3px 10px;
		}
		.tensor-address {
			flex-wrap: wrap;
			gap: 2px;
		}
	}
</style>
