<script lang="ts">
	import { STORY_CHARACTERS, type StoryProbe, type StoryConfig } from '$lib/stories/protocol';
	import Icon from './Icon.svelte';
	let {
		probe,
		lesioned,
		selected,
		token,
		config,
		busy,
		onselect,
		ontoken,
		onlesion
	}: {
		probe: StoryProbe | null;
		lesioned: StoryProbe | null;
		selected: number | null;
		token: number;
		config: StoryConfig;
		busy: boolean;
		onselect: (id: number) => void;
		ontoken: (position: number) => void;
		onlesion: () => void;
	} = $props();
	let count = $derived(config.layers * config.hidden);
	let layer = $derived(selected === null ? null : Math.floor(selected / config.hidden));
	let channel = $derived(selected === null ? null : selected % config.hidden);
	let activations = $derived(
		probe && selected !== null
			? probe.prompt.tokenIds.map(
					(_, index) => probe.activations[index * probe.unitCount + selected]
				)
			: []
	);
	let maximum = $derived(Math.max(0, ...activations));
	let currentActivation = $derived(activations[Math.min(token, activations.length - 1)]);
	let topTokens = $derived.by(() => {
		if (!probe) return [];
		return Array.from(probe.probabilities, (probability, id) => ({
			id,
			probability,
			delta:
				lesioned && lesioned.step === probe.step && lesioned.lesionNeuron === selected
					? lesioned.probabilities[id] - probability
					: null
		}))
			.sort((a, b) => b.probability - a.probability)
			.slice(0, 8);
	});
	let lesionMatches = $derived(
		!!probe &&
			!!lesioned &&
			lesioned.step === probe.step &&
			lesioned.lesionNeuron === selected &&
			lesioned.prompt.text === probe.prompt.text
	);
	let effectRows = $derived.by(() => {
		if (!probe || !lesioned || !lesionMatches) return [];
		return Array.from(probe.probabilities, (probability, id) => ({
			id,
			delta: lesioned.probabilities[id] - probability
		}))
			.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))
			.slice(0, 6);
	});
	function character(id: number): string {
		return id === 0 ? '↵' : id === 1 ? '␣' : STORY_CHARACTERS[id];
	}
	function literal(id: number): string {
		return id === 0 ? 'newline' : id === 1 ? 'space' : STORY_CHARACTERS[id];
	}
	function numeric(value: number | undefined): string {
		return value === undefined
			? '—'
			: value !== 0 && Math.abs(value) < 0.001
				? value.toExponential(2)
				: value.toFixed(3);
	}
	function step(delta: number): void {
		onselect(((selected ?? (delta > 0 ? -1 : 0)) + delta + count) % count);
	}
</script>

<aside class="story-probe" aria-label="TinyStories unit inspector">
	<div class="panel-heading">
		<h2><Icon name="target" size={14} />Unit inspector</h2>
		<span>Post-ReLU MLP</span>
	</div>
	<div class="unit-control">
		<label for="story-unit">Atlas ID</label><button
			class="icon-button"
			aria-label="Previous story unit"
			onclick={() => step(-1)}><Icon name="left" size={12} /></button
		><input
			id="story-unit"
			type="number"
			min="0"
			max={count - 1}
			value={selected ?? ''}
			placeholder={`0–${count - 1}`}
			aria-label="Story unit ID"
			oninput={(event) => {
				const value = event.currentTarget.valueAsNumber;
				if (Number.isInteger(value) && value >= 0 && value < count) onselect(value);
			}}
		/><button class="icon-button" aria-label="Next story unit" onclick={() => step(1)}
			><Icon name="right" size={12} /></button
		>
	</div>
	{#if selected !== null}
		<div class="address">
			<strong>L{(layer ?? 0) + 1} / {String(channel).padStart(4, '0')}</strong><span
				>layers[{layer}] · channel {channel}</span
			><small
				>{probe
					? `Activity measured at step ${probe.step}`
					: 'Run a prompt to measure activity'}</small
			>
		</div>
		<div class="activation-summary">
			<span>At character {Math.min(token + 1, probe?.prompt.tokenIds.length ?? 0)}</span><strong
				>{numeric(currentActivation)}</strong
			>
		</div>
		{#if activations.length}
			<svg
				class="activation-strip"
				viewBox="0 0 250 72"
				role="img"
				aria-label={`Activation of unit ${selected} at every character of the prompt`}
			>
				<line x1="0" x2="250" y1="58" y2="58" />
				{#each activations as activation, i (i)}<rect
						x={(i / activations.length) * 250}
						y={58 - (maximum ? (activation / maximum) * 50 : 0)}
						width={Math.max(0.5, 250 / activations.length - 1)}
						height={maximum ? (activation / maximum) * 50 : 0}
						class:chosen={i === token}
						><title>Position {i + 1}: {activation.toPrecision(6)}</title></rect
					>{/each}
				<text x="0" y="70">1</text><text x="250" y="70" text-anchor="end"
					>{activations.length} characters</text
				>
			</svg>
			<p class="scale">Peak {numeric(maximum)} · bar scale follows this unit</p>
		{:else}<div class="inspector-empty">
				Prompt activations will appear after a forward pass.
			</div>{/if}
	{:else}<div class="inspector-empty">
			<Icon name="target" size={23} />
			<p>Select a point or enter a unit ID.</p>
		</div>{/if}
	{#if probe}
		<div class="token-panel">
			<div class="minor-heading">
				<span>Probe positions</span><span>{probe.prompt.tokenIds.length} / {config.context}</span>
			</div>
			<div class="token-grid">
				{#each probe.prompt.tokenIds as id, i (i)}<button
						class:chosen={i === token}
						onclick={() => ontoken(i)}
						aria-pressed={i === token}
						aria-label={`Probe character ${i + 1}: ${literal(id)}`}
						title={`Position ${i + 1} · ${literal(id)}`}>{character(id)}</button
					>{/each}
			</div>
			{#if probe.prompt.truncatedCharacters}<p class="scale">
					Showing the last {probe.prompt.tokenIds.length} characters; {probe.prompt
						.truncatedCharacters} earlier characters were outside the context.
				</p>{/if}
		</div>
		<div class="prediction-panel">
			<div class="minor-heading">
				<span>Next-character distribution</span><span>Intact · step {probe.step}</span>
			</div>
			<div class="probabilities">
				{#each topTokens as next (next.id)}<div>
						<code title={literal(next.id)}>{character(next.id)}</code><span
							class="probability-track"
							><i style:width={`${Math.max(0, next.probability * 100)}%`}></i></span
						><span>{(next.probability * 100).toFixed(1)}%</span>
					</div>{/each}
			</div>
		</div>
	{/if}
	<div class="intervention">
		<button class="secondary" onclick={onlesion} disabled={busy || !probe || selected === null}
			><Icon name="target" size={13} />Silence selected unit</button
		>
		<p>Exact forward pass with this channel set to zero at all prompt positions.</p>
		{#if lesionMatches}<div class="lesion-result">
				<div class="minor-heading">
					<span>Largest probability changes</span><span>Δ lesioned − intact</span>
				</div>
				{#each effectRows as effect (effect.id)}<div>
						<code title={literal(effect.id)}>{character(effect.id)}</code><span
							>{effect.delta > 0 ? '+' : ''}{numeric(100 * effect.delta)} pp</span
						>
					</div>{/each}<small>Unit {selected} · step {lesioned?.step} · same prompt</small>
				{#if effectRows.length && Math.abs(effectRows[0].delta) < 1e-6}<p class="scale">
						Maximum change below 1e−6 probability. Repeat intact controls before interpreting
						effects this small.
					</p>{/if}
			</div>{/if}
	</div>
</aside>

<style>
	.story-probe {
		min-width: 0;
		background: var(--surface);
	}
	.panel-heading {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 8px;
		padding: 13px;
		border-bottom: 1px solid var(--line);
	}
	h2 {
		display: flex;
		align-items: center;
		gap: 7px;
		margin: 0;
		font-size: 11px;
		font-weight: 550;
	}
	.panel-heading > span {
		color: var(--muted);
		font: 8px var(--mono);
	}
	.unit-control {
		display: flex;
		align-items: center;
		gap: 5px;
		padding: 10px 12px;
		border-bottom: 1px solid var(--line);
	}
	.unit-control label {
		color: var(--muted);
		font: 9px var(--mono);
		margin-right: auto;
	}
	.unit-control input {
		width: 73px;
		padding: 6px;
		text-align: center;
		font: 10px var(--mono);
	}
	.unit-control button {
		width: 23px;
		height: 25px;
	}
	.address {
		display: flex;
		flex-direction: column;
		gap: 6px;
		padding: 14px 14px 9px;
	}
	.address strong {
		font: 16px var(--mono);
		color: var(--accent);
	}
	.address span,
	.address small {
		font: 8px var(--mono);
		color: var(--muted);
	}
	.activation-summary {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 8px 14px;
	}
	.activation-summary span {
		font-size: 10px;
		color: var(--muted);
	}
	.activation-summary strong {
		font: 16px var(--mono);
	}
	.activation-strip {
		width: calc(100% - 28px);
		height: 78px;
		display: block;
		margin: 0 14px;
	}
	.activation-strip line {
		stroke: var(--line);
		stroke-width: 0.5;
	}
	.activation-strip rect {
		fill: var(--layer-2);
		opacity: 0.5;
	}
	.activation-strip rect.chosen {
		fill: var(--accent);
		opacity: 1;
	}
	.activation-strip text {
		fill: var(--muted);
		font: 7px var(--mono);
	}
	.scale {
		color: var(--muted);
		font: 8px/1.6 var(--mono);
		margin: 7px 14px 12px;
	}
	.inspector-empty {
		min-height: 145px;
		display: flex;
		flex-direction: column;
		justify-content: center;
		align-items: center;
		text-align: center;
		gap: 12px;
		padding: 20px;
		color: var(--muted);
		font-size: 11px;
	}
	.inspector-empty p {
		margin: 0;
	}
	.token-panel,
	.prediction-panel {
		padding: 12px 14px;
		border-top: 1px solid var(--line);
	}
	.minor-heading {
		display: flex;
		justify-content: space-between;
		gap: 8px;
		font: 8px/1.4 var(--mono);
		color: var(--muted);
		margin-bottom: 10px;
	}
	.token-grid {
		display: flex;
		flex-wrap: wrap;
		gap: 3px;
		max-height: 138px;
		overflow: auto;
		padding: 2px;
	}
	.token-grid button {
		border: 1px solid transparent;
		border-radius: 3px;
		background: var(--surface-raised);
		font: 10px var(--mono);
		width: 18px;
		height: 23px;
		padding: 0;
		color: var(--muted);
	}
	.token-grid button.chosen {
		color: var(--ink);
		border-color: var(--accent);
		background: var(--surface-hover);
	}
	.token-panel .scale {
		margin: 10px 0 0;
	}
	.probabilities > div {
		display: flex;
		gap: 8px;
		align-items: center;
		margin: 7px 0;
	}
	.probabilities code {
		font: 10px var(--mono);
		width: 15px;
	}
	.probabilities > div > span:last-child {
		width: 38px;
		text-align: right;
		font: 8px var(--mono);
		color: var(--muted);
	}
	.probability-track {
		flex: 1;
		height: 4px;
		background: var(--surface-raised);
		border-radius: 2px;
		overflow: hidden;
	}
	.probability-track i {
		height: 100%;
		background: var(--accent);
		display: block;
		border-radius: 2px;
	}
	.intervention {
		border-top: 1px solid var(--line);
		padding: 12px 14px;
	}
	.intervention > button {
		width: 100%;
		font-size: 10px;
	}
	.intervention > p {
		color: var(--muted);
		font-size: 9px;
		line-height: 1.6;
		margin: 9px 0 0;
	}
	.lesion-result {
		margin-top: 14px;
		border: 1px solid color-mix(in srgb, var(--warning) 30%, var(--line));
		border-radius: 5px;
		padding: 10px;
	}
	.lesion-result > div:not(.minor-heading) {
		display: flex;
		justify-content: space-between;
		padding: 4px 0;
		font: 9px var(--mono);
	}
	.lesion-result code {
		font: 10px var(--mono);
	}
	.lesion-result > small {
		display: block;
		margin-top: 8px;
		color: var(--muted);
		font: 8px var(--mono);
	}
</style>
