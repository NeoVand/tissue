<script lang="ts">
	import type { TokenStoryGeneration } from '$lib/token-stories/protocol';
	import type { LiveTokenStoryFrame } from '$lib/token-stories/live-protocol';
	import type { TokenStoryTokenizer } from '$lib/token-stories/tokenizer';
	import Icon from './Icon.svelte';
	let {
		prompt,
		sample,
		tokenizer,
		frame,
		frames,
		layer,
		layers,
		selected,
		tokens,
		prefix,
		running,
		replaying,
		paused,
		stopping,
		complete,
		blocked,
		canStart,
		startLabel = 'Generate live',
		availability = '',
		pace,
		limit,
		temperature,
		samplingSeed,
		topK,
		onprompt,
		onstart,
		onpause,
		onnextlayer,
		onnexttoken,
		onstop,
		onpace,
		onlayer,
		onframe,
		onreplay,
		onclear
	}: {
		prompt: string;
		sample: TokenStoryGeneration | null;
		tokenizer: TokenStoryTokenizer | null;
		frame: LiveTokenStoryFrame | null;
		frames: LiveTokenStoryFrame[];
		layer: number | null;
		layers: number;
		selected: number | null;
		tokens: number[];
		prefix: string;
		running: boolean;
		replaying: boolean;
		paused: boolean;
		stopping: boolean;
		complete: boolean;
		blocked: boolean;
		canStart: boolean;
		startLabel?: string;
		availability?: string;
		pace: number;
		limit: number;
		temperature: number;
		samplingSeed: number;
		topK: number;
		onprompt: (text: string) => void;
		onstart: () => void;
		onpause: () => void;
		onnextlayer: () => void;
		onnexttoken: () => void;
		onstop: () => void;
		onpace: (milliseconds: number) => void;
		onlayer: (layer: number) => void;
		onframe: (index: number) => void;
		onreplay: () => void;
		onclear: () => void;
	} = $props();
	let playing = $derived(running || replaying);
	let primaryLabel = $derived(
		playing ? `${paused ? 'Resume' : 'Pause'} ${replaying ? 'replay' : 'generation'}` : startLabel
	);
	let state = $derived(
		playing
			? paused
				? 'paused'
				: replaying
					? 'replaying'
					: 'playing'
			: complete
				? 'complete'
				: 'idle'
	);
	let output = $derived(tokenizer?.decode(tokens) ?? '');
	let selectedValue = $derived(frame && selected !== null ? frame.activations[selected] : null);
	let nextTokens = $derived(
		frame
			? Array.from(frame.probabilities, (probability, id) => ({ probability, id }))
					.sort((a, b) => b.probability - a.probability)
					.slice(0, 4)
			: []
	);
	function piece(id: number): string {
		return tokenizer?.tokenPiece(id).replaceAll(' ', '·').replaceAll('\n', '↵') ?? String(id);
	}
</script>

<section
	class="live-panel"
	class:playing
	aria-label="Live token generation"
	data-state={state}
	data-frame={frame?.index ?? -1}
	data-layer={layer ?? -1}
	data-emitted={tokens.length}
>
	<div class="live-heading">
		<h2><Icon name="activity" size={13} />Live generation</h2>
		<span class:lit={playing}
			>{playing
				? replaying
					? 'Recorded playback'
					: 'Live inference'
				: frames.length
					? 'Recorded trace available'
					: 'Measured MLP activity'}</span
		><small
			>Next generation · {limit} tokens · seed {samplingSeed} · T {temperature} · top-k {topK}</small
		>
	</div>
	{#if playing || tokens.length || (complete && frames.length)}<div
			class="live-output"
			aria-label="Live generated text"
		>
			<span>{prefix}</span>{output}{#if playing}<i aria-hidden="true"></i>{/if}<small
				>{tokens.length} output tokens{running
					? ' · live'
					: replaying
						? ' · replay'
						: ' · recorded'}</small
			>
		</div>{/if}
	<div class="live-prefix">
		<label for="live-generation-prefix">Next prefix</label><input
			id="live-generation-prefix"
			value={prompt}
			oninput={(event) => onprompt(event.currentTarget.value)}
			disabled={blocked}
			maxlength="100000"
			spellcheck="false"
		/><button
			class="primary"
			onclick={playing ? onpause : onstart}
			disabled={playing ? stopping : !canStart}
			><Icon name={playing && !paused ? 'pause' : 'play'} size={12} />{primaryLabel}</button
		>
	</div>
	{#if availability && !playing}<p class="generation-availability" role="status">
			{availability}
		</p>{/if}
	<details class="playback-settings" open={frames.length > 0 || playing}>
		<summary>Playback controls <span>Layer stepping & speed</span></summary>
		<div class="playback-controls">
			<div class="transport-controls">
				<button onclick={onnextlayer} disabled={stopping || (!playing && !frame)}
					><Icon name="right" size={12} />Next layer</button
				>
				<button onclick={onnexttoken} disabled={stopping || (!playing && !frames.length)}
					><Icon name="right" size={12} />Next token</button
				>
				{#if playing}<button class="stop" onclick={onstop} disabled={stopping}
						><Icon name="close" size={12} />{replaying ? 'Stop replay' : 'Stop generation'}</button
					>{/if}
			</div>
			<div class="pace-controls">
				<span>Per layer</span
				>{#each [{ label: 'Fast', ms: 40 }, { label: 'Read', ms: 160 }, { label: 'Slow', ms: 400 }] as option (option.ms)}<button
						class:chosen={pace === option.ms}
						aria-pressed={pace === option.ms}
						onclick={() => onpace(option.ms)}>{option.label} · {option.ms} ms</button
					>{/each}
			</div>
		</div>
		<div class="layer-track">
			<span>Input</span><Icon
				name="right"
				size={10}
			/>{#each Array.from({ length: layers }, (_, index) => index) as index (index)}<button
					class:focused={layer === index}
					disabled={playing || !frame}
					onclick={() => onlayer(index)}
					aria-label={`Inspect recorded layer ${index + 1}`}
					aria-pressed={layer === index}>L{index + 1}</button
				><Icon name="right" size={10} />{/each}<span
				class:focused={frame && tokens.length > frame.index}>Next token</span
			><small
				>{frame
					? `Frame ${frame.index + 1} · step ${frame.step}`
					: 'Awaiting a measured frame'}</small
			>
		</div>
	</details>
	{#if frame}<div class="live-readout">
			<span>Input ends <code>{JSON.stringify(frame.context.pieces.at(-1))}</code></span><span
				>{selected === null
					? 'Select a channel'
					: `U${selected} · L${Math.floor(selected / frame.config.hidden) + 1} C${selected % frame.config.hidden}`}</span
			><output aria-label="Selected channel activation"
				>{selectedValue === null ? '—' : selectedValue.toPrecision(5)}</output
			>
		</div>{/if}
	<details class="frame-evidence">
		<summary
			>Measured input & evidence <span
				>{sample && frames.length
					? `Saved step ${sample.step} · seed ${sample.samplingSeed} · ${sample.tokenIds.length}/${sample.requestedTokens} tokens`
					: frame
						? `${frame.context.tokenIds.length} input tokens · final position ${frame.position + 1}`
						: 'How to read this view'}</span
			></summary
		>
		{#if sample && frames.length}<p class="sample-settings">
				Recorded trace · step {sample.step} · sampling seed {sample.samplingSeed} · temperature {sample.temperature}
				· top-k {sample.topK} · {sample.tokenIds.length} output tokens / {sample.requestedTokens} requested.
				The prefix and settings above are for the next generation.
			</p>{/if}
		{#if frame}
			<div class="measured-context">
				<div>
					<span
						>Exact model input · {frame.context.tokenIds.length} tokens{frame.context
							.truncatedTokens
							? ` · ${frame.context.truncatedTokens} outside context`
							: ''}</span
					><strong
						>{selected === null
							? 'Select a unit to inspect'
							: `U${selected} · L${Math.floor(selected / frame.config.hidden) + 1} C${selected % frame.config.hidden}`}</strong
					>
				</div>
				<div class="context-tail">
					{#if frame.context.tokenIds.length > 14}<small>…</small
						>{/if}{#each frame.context.tokenIds.slice(-14) as id, index (index)}<code
							class:final={index === Math.min(14, frame.context.tokenIds.length) - 1}
							title={`Vocabulary ID ${id}`}>{piece(id)}</code
						>{/each}
				</div>
				<div class="full-context">
					<p>{frame.context.text || 'Beginning-of-story token only'}</p>
					<div class="context-tail">
						{#each frame.context.tokenIds as id, index (index)}<code
								title={`Position ${index + 1}; vocabulary ID ${id}`}>{piece(id)}</code
							>{/each}
					</div>
				</div>
			</div>
		{/if}
		{#if nextTokens.length}<div class="next-distribution">
				<span>Raw probabilities · after all layers</span>{#each nextTokens as next (next.id)}<code
						title={`Vocabulary ID ${next.id}`}
						>{piece(next.id)} <b>{(next.probability * 100).toFixed(1)}%</b></code
					>{/each}
			</div>{/if}
		<div class="playback-note">
			Measured post-ReLU MLP values at the final input token predict the next token. Layer pacing is
			presentation timing, not GPU execution. Attention and residual-stream internals are not shown.
		</div>
	</details>
	{#if frames.length}<div class="trace-history">
			<div>
				<span>{frames.length} frames</span><button onclick={onreplay} disabled={blocked}
					><Icon name="play" size={11} />Replay trace</button
				>{#if frame}<button onclick={onclear} disabled={playing}
						><Icon name="close" size={11} />Return to probe</button
					>{/if}
			</div>
			<div class="trace-tokens">
				{#each frames as stored, index (index)}<button
						class:chosen={frame?.index === stored.index}
						onclick={() => onframe(index)}
						disabled={blocked}
						aria-label={`Inspect recorded token ${stored.index + 1}`}
						title={`Output ${stored.index + 1} · input ends with ${JSON.stringify(stored.context.pieces.at(-1))}`}
						>{piece(stored.sampledToken)}</button
					>{/each}
			</div>
		</div>{/if}
</section>

<style>
	.generation-availability {
		margin: 10px 0 0;
		font-size: 13px;
		line-height: 1.6;
		color: var(--muted);
	}
	.live-panel {
		border-top: 1px solid var(--line);
		border-bottom: 1px solid var(--line);
		background: var(--surface);
		padding: 9px 12px;
		min-width: 0;
	}
	.live-heading {
		display: flex;
		align-items: center;
		gap: 10px;
		flex-wrap: wrap;
		margin-bottom: 8px;
	}
	h2 {
		display: flex;
		align-items: center;
		gap: 6px;
		margin: 0;
		font-size: 13px;
		font-weight: 550;
	}
	.live-heading > span {
		font: 12px var(--mono);
		color: var(--muted);
		border: 1px solid var(--line);
		border-radius: 3px;
		padding: 3px 5px;
	}
	.live-heading > span.lit {
		color: var(--accent);
		border-color: color-mix(in srgb, var(--accent) 50%, var(--line));
	}
	.live-heading small {
		font: 12px var(--mono);
		color: var(--faint);
		margin-left: auto;
	}
	.live-prefix {
		display: grid;
		grid-template-columns: auto minmax(0, 1fr) auto;
		gap: 8px;
		align-items: center;
	}
	.live-prefix label {
		font: 12px var(--mono);
		color: var(--muted);
	}
	.live-prefix input {
		min-width: 0;
		font: 12px var(--mono);
		padding: 6px 8px;
	}
	.live-prefix > button {
		font-size: 12px;
		white-space: nowrap;
	}
	.playback-controls {
		display: flex;
		align-items: center;
		gap: 8px;
		flex-wrap: wrap;
		margin-top: 7px;
	}
	.transport-controls,
	.pace-controls {
		display: flex;
		align-items: center;
		gap: 4px;
		flex-wrap: wrap;
	}
	.pace-controls {
		margin-left: auto;
	}
	.pace-controls > span {
		font: 12px var(--mono);
		color: var(--faint);
		margin-right: 4px;
	}
	button {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		gap: 4px;
		cursor: pointer;
	}
	.playback-controls button,
	.trace-history button {
		background: transparent;
		border: 1px solid var(--line);
		border-radius: 4px;
		padding: 5px 6px;
		color: var(--muted);
		font: 12px var(--mono);
	}
	.playback-controls button.chosen {
		color: var(--accent);
		border-color: color-mix(in srgb, var(--accent) 45%, var(--line));
	}
	.playback-controls button.stop {
		color: var(--warning);
	}
	.layer-track {
		display: flex;
		align-items: center;
		gap: 6px;
		margin-top: 8px;
		flex-wrap: wrap;
		color: var(--faint);
		font: 12px var(--mono);
	}
	.layer-track button {
		font: 12px var(--mono);
		padding: 5px 8px;
		background: var(--surface-raised);
		border: 1px solid var(--line);
		border-radius: 4px;
		color: var(--muted);
		opacity: 1;
	}
	.layer-track button.focused {
		color: var(--ink);
		border-color: var(--accent);
		background: color-mix(in srgb, var(--accent) 16%, var(--surface));
	}
	.layer-track > span.focused {
		color: var(--accent);
	}
	.layer-track small {
		margin-left: auto;
		color: var(--muted);
		font: 12px var(--mono);
	}
	.measured-context {
		margin-top: 12px;
		border-top: 1px solid var(--line);
		padding-top: 10px;
	}
	.measured-context > div:first-child {
		display: flex;
		justify-content: space-between;
		gap: 10px;
		flex-wrap: wrap;
		font: 12px/1.6 var(--mono);
		color: var(--muted);
	}
	.measured-context strong {
		font-weight: 400;
		color: var(--accent);
	}
	.context-tail {
		display: flex;
		gap: 3px;
		flex-wrap: wrap;
		align-items: center;
		margin-top: 7px;
	}
	.context-tail code {
		font: 12px var(--mono);
		padding: 4px 5px;
		border: 1px solid var(--line);
		border-radius: 3px;
		overflow-wrap: anywhere;
	}
	.context-tail code.final {
		border-color: var(--accent);
		color: var(--accent);
	}
	.live-readout {
		display: flex;
		align-items: center;
		gap: 8px;
		margin-top: 7px;
		color: var(--muted);
		font: 12px var(--mono);
	}
	.live-readout > span:first-child {
		margin-right: auto;
	}
	.live-readout code {
		color: var(--accent);
		font: 12px var(--mono);
	}
	.live-readout output {
		font: 13px var(--mono);
		color: var(--ink);
	}
	details {
		margin-top: 7px;
		font: 12px/1.7 var(--mono);
		color: var(--muted);
	}
	summary {
		cursor: pointer;
	}
	.frame-evidence > summary > span {
		color: var(--faint);
		margin-left: 9px;
	}
	.frame-evidence[open] {
		padding-bottom: 6px;
	}
	details p {
		max-height: 100px;
		overflow: auto;
		white-space: pre-wrap;
		overflow-wrap: anywhere;
	}
	.live-output {
		margin-top: 7px;
		border: 1px solid var(--line);
		border-radius: 5px;
		background: var(--bg);
		padding: 7px 9px;
		font: 13px/1.65 var(--mono);
		white-space: pre-wrap;
		overflow-wrap: anywhere;
		max-height: 64px;
		overflow: auto;
	}
	.live-output > span {
		color: var(--muted);
	}
	.live-output i {
		display: inline-block;
		width: 4px;
		height: 12px;
		background: var(--accent);
		opacity: 0.7;
		margin-left: 3px;
		vertical-align: middle;
	}
	.live-output small {
		display: inline;
		color: var(--faint);
		font: 12px var(--mono);
		margin-left: 8px;
	}
	.next-distribution {
		display: flex;
		gap: 7px;
		align-items: center;
		flex-wrap: wrap;
		margin-top: 10px;
		font: 12px var(--mono);
		color: var(--muted);
	}
	.next-distribution code {
		border: 1px solid var(--line);
		padding: 3px 5px;
		border-radius: 3px;
		font: 12px var(--mono);
	}
	.next-distribution b {
		color: var(--ink);
		font-weight: 400;
	}
	.playback-note {
		color: var(--muted);
		font: 12px/1.7 var(--mono);
		margin-top: 10px;
	}
	.trace-history {
		display: grid;
		grid-template-columns: auto minmax(0, 1fr);
		align-items: center;
		gap: 8px;
		border-top: 1px solid var(--line);
		padding-top: 7px;
		margin-top: 7px;
	}
	.trace-history > div:first-child {
		display: flex;
		align-items: center;
		gap: 5px;
		white-space: nowrap;
	}
	.trace-history span {
		font: 12px var(--mono);
		color: var(--muted);
		margin-right: auto;
	}
	.trace-tokens {
		display: flex;
		flex-wrap: nowrap;
		gap: 3px;
		min-width: 0;
		max-height: 36px;
		overflow: auto;
	}
	.trace-tokens button {
		flex-shrink: 0;
		white-space: nowrap;
	}
	.trace-tokens button.chosen {
		color: var(--accent);
		border-color: var(--accent);
	}
	button:disabled {
		cursor: default;
		opacity: 0.4;
	}
	button:focus-visible {
		outline: 1px solid var(--accent);
		outline-offset: 2px;
	}
	@media (max-width: 700px) {
		.trace-history {
			grid-template-columns: minmax(0, 1fr);
		}
		.live-readout {
			flex-wrap: wrap;
		}
		.live-prefix {
			grid-template-columns: minmax(0, 1fr) auto;
		}
		.live-prefix label {
			grid-column: 1/-1;
		}
		.live-heading small {
			width: 100%;
			margin-left: 0;
		}
		.pace-controls {
			margin-left: 0;
		}
		.layer-track small {
			width: 100%;
			margin-left: 0;
		}
	}

	.live-panel {
		padding: 16px 18px;
		border-top: 1px solid var(--line);
		background: var(--surface);
	}
	.live-heading {
		flex-wrap: wrap;
		gap: 8px;
		margin-bottom: 12px;
	}
	.live-heading h2 {
		font-size: 14px;
	}
	.live-heading > small {
		font:
			12px 'DM Sans',
			sans-serif;
	}
	.live-prefix {
		gap: 10px;
	}
	.live-prefix input {
		font-size: 13px;
		min-height: 38px;
	}
	.playback-controls {
		gap: 12px;
		flex-wrap: wrap;
		margin: 12px 0;
	}
	.transport-controls,
	.pace-controls {
		flex-wrap: wrap;
	}
	.transport-controls button,
	.pace-controls button {
		min-height: 32px;
		padding: 6px 9px;
		font:
			12px 'DM Sans',
			sans-serif;
	}
	.layer-track {
		flex-wrap: wrap;
		gap: 6px;
	}
	.layer-track button {
		min-width: 34px;
		min-height: 30px;
	}
	.live-output {
		font:
			15px/1.7 'DM Sans',
			sans-serif;
		max-height: 140px;
	}
	.trace-tokens {
		max-height: 84px;
	}
	.trace-tokens button {
		padding: 6px 8px;
	}
	.frame-evidence {
		margin-top: 12px;
	}
	@media (max-width: 600px) {
		.live-panel {
			padding: 14px 12px;
		}
		.live-heading > small {
			display: none;
		}
	}

	.playback-settings {
		margin-top: 12px;
	}
	.playback-settings > summary {
		font-size: 12px;
		color: var(--muted);
	}
	.playback-settings > summary span {
		margin-left: 8px;
	}
	.live-output {
		margin: 0 0 14px;
		padding: 10px 0;
		background: transparent;
		border: 0;
		border-radius: 0;
	}
	.live-panel[data-state='playing'] .live-prefix input,
	.live-panel[data-state='playing'] .live-prefix label,
	.live-panel[data-state='replaying'] .live-prefix input,
	.live-panel[data-state='replaying'] .live-prefix label,
	.live-panel[data-state='paused'] .live-prefix input,
	.live-panel[data-state='paused'] .live-prefix label {
		display: none;
	}
	.live-panel[data-state='playing'] .live-prefix,
	.live-panel[data-state='replaying'] .live-prefix,
	.live-panel[data-state='paused'] .live-prefix {
		display: flex;
		justify-content: flex-end;
	}

	.live-panel.playing {
		display: grid;
		grid-template-columns: minmax(0, 1fr) auto;
		gap: 0 12px;
	}
	.playing .live-heading {
		grid-column: 1;
		grid-row: 1;
		margin-bottom: 0;
	}
	.playing .live-heading > small {
		display: none;
	}
	.playing .live-prefix {
		grid-column: 2;
		grid-row: 1;
	}
	.playing .live-output {
		grid-column: 1 / -1;
		grid-row: 2;
		margin: 4px 0;
	}
	.playing .playback-settings,
	.playing .live-readout,
	.playing .frame-evidence,
	.playing .trace-history {
		grid-column: 1 / -1;
	}
	@media (max-width: 600px) {
		.playing .live-heading > span {
			display: none;
		}
	}
</style>
