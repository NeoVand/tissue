<script lang="ts">
	import {
		tokenStoryUnitAddress,
		type TokenStoryProbe,
		type TokenStoryConfig,
		type TokenStoryAtlas
	} from '$lib/token-stories/protocol';
	import type { TokenStoryTokenizer } from '$lib/token-stories/tokenizer';
	import type { LiveTokenStoryFrame } from '$lib/token-stories/live-protocol';
	import Icon from './Icon.svelte';
	let {
		probe,
		liveFrame = null,
		tokenizer,
		atlas = null,
		lesioned,
		selected,
		token,
		config,
		busy,
		onselect,
		ontoken,
		onlesion,
		oncontext
	}: {
		probe: TokenStoryProbe | null;
		liveFrame?: LiveTokenStoryFrame | null;
		tokenizer: TokenStoryTokenizer | null;
		atlas?: TokenStoryAtlas | null;
		lesioned: TokenStoryProbe | null;
		selected: number | null;
		token: number;
		config: TokenStoryConfig;
		busy: boolean;
		onselect: (id: number) => void;
		ontoken: (position: number) => void;
		onlesion: () => void;
		oncontext?: (text: string) => void;
	} = $props();
	let count = $derived(config.layers * config.hidden);
	let layer = $derived(selected === null ? null : Math.floor(selected / config.hidden));
	let channel = $derived(selected === null ? null : selected % config.hidden);
	let address = $derived(
		selected === null
			? null
			: (probe?.unitAddresses[selected] ?? tokenStoryUnitAddress(config, selected))
	);
	let calibration = $derived(
		atlas && selected !== null
			? Array.from(
					atlas.fingerprints.subarray(
						selected * atlas.dimensions,
						(selected + 1) * atlas.dimensions
					)
				)
			: []
	);
	let calibrationPeak = $derived(Math.max(0, ...calibration));
	let strongestWindows = $derived(
		atlas
			? atlas.examples
					.map((text, index) => ({
						text,
						index,
						peak: Math.max(0, ...calibration.slice(index * 16, index * 16 + 16))
					}))
					.sort((a, b) => b.peak - a.peak)
					.slice(0, 3)
			: []
	);
	let activations = $derived(
		probe && selected !== null
			? probe.prompt.tokenIds.map(
					(_, index) => probe.activations[index * probe.unitCount + selected]
				)
			: []
	);
	let maximum = $derived(Math.max(0, ...activations));
	let currentActivation = $derived(
		liveFrame && selected !== null
			? liveFrame.activations[selected]
			: activations[Math.min(token, activations.length - 1)]
	);
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
			lesioned.modelId === probe.modelId &&
			lesioned.tokenizerId === probe.tokenizerId &&
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
	function tokenPiece(id: number): string {
		return tokenizer?.tokenPiece(id).replaceAll(' ', '·').replaceAll('\n', '↵') ?? String(id);
	}
	function literal(id: number): string {
		return tokenizer?.tokenPiece(id) ?? String(id);
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
	function calibrationLabel(index: number): string {
		if (!atlas) return '';
		const window = Math.floor(index / 16),
			knot = index % 16;
		const position = atlas.tokenPositions[window][knot];
		return `Window ${window + 1}, token ${position + 1} ${JSON.stringify(literal(atlas.tokenIds[window][position]))}: activation ${calibration[index].toPrecision(5)}`;
	}
</script>

<aside class="token-story-probe" aria-label="TinyStories unit inspector">
	<div class="panel-heading">
		<h2><Icon name="target" size={14} />Unit inspector</h2>
		<span>Post-ReLU MLP</span>
	</div>
	<div class="unit-control">
		<label for="token-story-unit">Atlas ID</label><button
			class="icon-button"
			aria-label="Previous subword unit"
			onclick={() => step(-1)}><Icon name="left" size={12} /></button
		><input
			id="token-story-unit"
			type="number"
			min="0"
			max={count - 1}
			value={selected ?? ''}
			placeholder={`0–${count - 1}`}
			aria-label="Subword unit ID"
			oninput={(event) => {
				const value = event.currentTarget.valueAsNumber;
				if (Number.isInteger(value) && value >= 0 && value < count) onselect(value);
			}}
		/><button class="icon-button" aria-label="Next subword unit" onclick={() => step(1)}
			><Icon name="right" size={12} /></button
		>
	</div>
	{#if selected !== null}
		<div class="address">
			<strong>L{(layer ?? 0) + 1} / {String(channel).padStart(4, '0')}</strong><span
				>layers[{layer}] · channel {channel}</span
			><small
				>{liveFrame
					? `Live trace · step ${liveFrame.step}`
					: probe
						? `Activity measured at step ${probe.step}`
						: 'Run a prompt to measure activity'}</small
			>
		</div>
		{#if address}<div class="tensor-address">
				<span>Exact feed-forward channel</span>
				<code>{address.incomingTensor}[:, {address.incomingColumn}]</code>
				<div>
					<span>{config.width}</span><Icon name="right" size={10} /><b>ReLU · C{address.channel}</b
					><Icon name="right" size={10} /><span>{config.width}</span>
				</div>
				<code>{address.outgoingTensor}[{address.outgoingRow}, :]</code>
			</div>{/if}
		<div class="activation-summary">
			<span
				>{liveFrame
					? `Final input token ${liveFrame.position + 1}`
					: probe
						? `At token ${Math.min(token + 1, probe.prompt.tokenIds.length)}`
						: 'No prompt measured'}</span
			><strong>{numeric(currentActivation)}</strong>
		</div>
		{#if liveFrame}
			<p class="scale">
				Measured at the final input position, predicting output token {liveFrame.index + 1}. Earlier
				positions are not captured in this trace.
			</p>
		{:else if activations.length}
			<svg
				class="activation-strip"
				viewBox="0 0 250 72"
				role="img"
				aria-label={`Activation of unit ${selected} at every token of the prompt`}
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
					>{activations.length} tokens</text
				>
			</svg>
			<p class="scale">Peak {numeric(maximum)} · bar scale follows this unit</p>
		{:else}<div class="inspector-empty">
				{atlas
					? 'Recorded calibration responses below. Resume the checkpoint to probe new text.'
					: 'Prompt activations will appear after a forward pass.'}
			</div>{/if}
		{#if !probe && !liveFrame && atlas && calibration.length}<div class="calibration-panel">
				<div class="minor-heading">
					<span>Recorded calibration · step {atlas.step}</span><span>8 × 16</span>
				</div>
				<div
					class="calibration-grid"
					role="img"
					aria-label={`Unit ${selected}: raw responses at all 128 calibration coordinates`}
				>
					{#each calibration as value, index (index)}<span
							title={calibrationLabel(index)}
							style:opacity={0.12 + 0.88 * (calibrationPeak ? value / calibrationPeak : 0)}
						></span>{/each}
				</div>
				<p class="scale">
					All 128 coordinates · shade follows this unit’s peak {numeric(calibrationPeak)}. Rows are
					fixed text windows.
				</p>
				<p class="scale">
					Probing the text below retokenizes it as a new prompt; it does not replay the recorded
					window.
				</p>
				{#each strongestWindows as window (window.index)}<button
						class="context-example"
						onclick={() => oncontext?.(window.text)}
						disabled={!oncontext || busy}
						title="Retokenized as a new prompt; this does not replay the saved calibration window."
						><span>Window {window.index + 1} · peak {numeric(window.peak)}</span><q>{window.text}</q
						><small>Probe this text <Icon name="right" size={10} /></small></button
					>{/each}
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
						aria-label={`Probe token ${i + 1}: ${literal(id)}`}
						title={`Position ${i + 1} · ${literal(id)}`}>{tokenPiece(id)}</button
					>{/each}
			</div>
			{#if probe.prompt.truncatedTokens}<p class="scale">
					Showing the last {probe.prompt.tokenIds.length} tokens; {probe.prompt.truncatedTokens} earlier
					tokens were outside the context.
				</p>{/if}
		</div>
		<div class="prediction-panel">
			<div class="minor-heading">
				<span>Next-token distribution</span><span>Final context position</span>
			</div>
			<p class="distribution-note">
				Top eight of {config.vocabularySize.toLocaleString()} tokens · intact step {probe.step}.
				Moving the activation cursor does not change this final-position prediction.
			</p>
			<div class="probabilities">
				{#each topTokens as next (next.id)}<div>
						<code title={literal(next.id)}>{tokenPiece(next.id)}</code><span
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
						<code title={literal(effect.id)}>{tokenPiece(effect.id)}</code><span
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
	.tensor-address {
		margin: 0 12px 8px;
		padding: 9px;
		border: 1px solid var(--line);
		border-radius: 5px;
		display: flex;
		flex-direction: column;
		gap: 7px;
	}
	.tensor-address > span {
		font: 12px var(--mono);
		color: var(--faint);
	}
	.tensor-address code {
		font: 12px var(--mono);
		overflow-wrap: anywhere;
		color: var(--muted);
	}
	.tensor-address > div {
		display: flex;
		align-items: center;
		gap: 9px;
		font: 12px var(--mono);
		color: var(--muted);
	}
	.tensor-address b {
		color: var(--accent);
		font-weight: 500;
	}
	.calibration-panel {
		padding: 12px 14px;
		border-top: 1px solid var(--line);
	}
	.calibration-grid {
		display: grid;
		grid-template-columns: repeat(16, minmax(0, 1fr));
		gap: 3px;
	}
	.calibration-grid > span {
		height: 9px;
		border-radius: 2px;
		background: var(--layer-1);
	}
	.calibration-panel .scale {
		margin: 7px 0 10px;
	}
	.context-example {
		display: flex;
		flex-direction: column;
		gap: 6px;
		text-align: left;
		width: 100%;
		background: transparent;
		border: 1px solid var(--line);
		border-radius: 5px;
		padding: 8px;
		margin-top: 6px;
		color: var(--ink);
	}
	.context-example > span,
	.context-example small {
		color: var(--muted);
		font: 12px var(--mono);
	}
	.context-example q {
		font-size: 12px;
		line-height: 1.6;
		display: -webkit-box;
		-webkit-line-clamp: 3;
		line-clamp: 3;
		-webkit-box-orient: vertical;
		overflow: hidden;
	}
	.context-example small {
		display: flex;
		align-items: center;
		gap: 5px;
	}
	.distribution-note {
		font: 12px/1.6 var(--mono);
		color: var(--muted);
		margin: 10px 0 0;
	}
	.token-story-probe {
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
		font-size: 13px;
		font-weight: 550;
	}
	.panel-heading > span {
		color: var(--muted);
		font: 12px var(--mono);
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
		font: 12px var(--mono);
		margin-right: auto;
	}
	.unit-control input {
		width: 73px;
		padding: 6px;
		text-align: center;
		font: 12px var(--mono);
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
		font: 12px var(--mono);
		color: var(--muted);
	}
	.activation-summary {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 8px 14px;
	}
	.activation-summary span {
		font-size: 12px;
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
		font: 12px var(--mono);
	}
	.scale {
		color: var(--muted);
		font: 12px/1.6 var(--mono);
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
		font-size: 13px;
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
		font: 12px/1.4 var(--mono);
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
		font: 12px var(--mono);
		min-width: 24px;
		max-width: 100%;
		min-height: 25px;
		padding: 3px 5px;
		overflow-wrap: anywhere;
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
		font: 12px var(--mono);
		width: 86px;
		flex-shrink: 0;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.probabilities > div > span:last-child {
		width: 38px;
		text-align: right;
		font: 12px var(--mono);
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
		font-size: 12px;
	}
	.intervention > p {
		color: var(--muted);
		font-size: 12px;
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
		font: 12px var(--mono);
	}
	.lesion-result code {
		font: 12px var(--mono);
		max-width: 130px;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.lesion-result > small {
		display: block;
		margin-top: 8px;
		color: var(--muted);
		font: 12px var(--mono);
	}
</style>
