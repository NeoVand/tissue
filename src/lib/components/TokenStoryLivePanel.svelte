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
		availability = '',
		pace,
		limit,
		temperature,
		samplingSeed,
		topK,
		onprompt,
		onnextlayer,
		onnexttoken,
		onstop,
		onpace,
		onlayer,
		onframe,
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
		availability?: string;
		pace: number;
		limit: number;
		temperature: number;
		samplingSeed: number;
		topK: number;
		onprompt: (text: string) => void;
		onnextlayer: () => void;
		onnexttoken: () => void;
		onstop: () => void;
		onpace: (milliseconds: number) => void;
		onlayer: (layer: number) => void;
		onframe: (index: number) => void;
		onclear: () => void;
	} = $props();
	let playing = $derived(running || replaying);
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
		<h2>
			<Icon name="activity" size={16} />{playing
				? replaying
					? 'Recorded playback'
					: 'Live generation'
				: 'Generation'}
		</h2>
		<span
			>{playing
				? paused
					? 'Paused'
					: 'Running'
				: frames.length
					? `${frames.length} measured frames`
					: `${limit} tokens · T ${temperature} · top-k ${topK}`}</span
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
	<div class="live-prefix" hidden={playing}>
		<label for="live-generation-prefix">Prompt</label><input
			id="live-generation-prefix"
			value={prompt}
			oninput={(event) => onprompt(event.currentTarget.value)}
			disabled={blocked}
			maxlength="100000"
			spellcheck="false"
		/>
	</div>
	{#if availability && !playing}<p class="generation-availability" role="status">
			{availability}
		</p>{/if}
	{#if frame || playing}
		<div class="playback-controls">
			<div class="transport-controls">
				<button onclick={onnextlayer} disabled={stopping || (!playing && !frame)}
					><Icon name="right" size={14} />Next layer</button
				>
				<button onclick={onnexttoken} disabled={stopping || (!playing && !frames.length)}
					><Icon name="right" size={14} />Next token</button
				>
				{#if playing}<button onclick={onstop} disabled={stopping}
						><Icon name="close" size={14} />{replaying ? 'Stop replay' : 'Stop generation'}</button
					>{/if}
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
		</div>
	{/if}
	{#if frame}<div class="live-readout">
			<span>Input ends <code>{JSON.stringify(frame.context.pieces.at(-1))}</code></span><span
				>{selected === null
					? 'Select a channel'
					: `U${selected} · L${Math.floor(selected / frame.config.hidden) + 1} C${selected % frame.config.hidden}`}</span
			><output aria-label="Selected channel activation"
				>{selectedValue === null ? '—' : selectedValue.toPrecision(5)}</output
			>
		</div>{/if}
	<div class="console-details">
		<details class="playback-settings">
			<summary>Playback controls</summary>
			<div class="pace-controls">
				<span>Time per layer</span>
				{#each [{ label: 'Fast', ms: 40 }, { label: 'Read', ms: 160 }, { label: 'Slow', ms: 400 }] as option (option.ms)}
					<button
						class:chosen={pace === option.ms}
						aria-pressed={pace === option.ms}
						onclick={() => onpace(option.ms)}>{option.label} · {option.ms} ms</button
					>
				{/each}
			</div>
			<p>
				Next generation: {limit} tokens · seed {samplingSeed} · temperature {temperature} · top-k {topK}.
				Change these in Model → Generation settings.
			</p>
		</details>
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
				Measured post-ReLU MLP values at the final input token predict the next token. Layer pacing
				is presentation timing, not GPU execution. Attention and residual-stream internals are not
				shown.
			</div>
		</details>
		{#if frames.length}<details class="trace-history">
				<summary>Recorded tokens <span>{frames.length} frames</span></summary>
				<div>
					{#if frame}<button onclick={onclear} disabled={playing}
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
			</details>{/if}
	</div>
</section>

<style>
	.live-panel {
		padding: 16px 20px 12px;
		background: var(--surface);
		border-bottom: 1px solid var(--line);
		min-width: 0;
	}
	.live-heading {
		display: flex;
		align-items: center;
		gap: 12px;
		margin-bottom: 10px;
	}
	h2 {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: 14px;
		font-weight: 550;
		margin: 0;
	}
	.live-heading > span {
		color: var(--muted);
		font-size: 12px;
		margin-left: auto;
	}
	.playing h2 {
		color: var(--accent);
	}
	.live-prefix {
		display: flex;
		align-items: center;
		gap: 14px;
	}
	.live-prefix label {
		font-size: 13px;
		color: var(--muted);
	}
	.live-prefix input {
		flex: 1;
		min-width: 0;
		font-size: 15px;
		padding: 9px 12px;
	}
	.generation-availability {
		color: var(--muted);
		font-size: 12px;
		line-height: 1.6;
		margin: 8px 0;
	}
	.live-output {
		font-size: 17px;
		line-height: 1.6;
		max-height: 104px;
		overflow-y: auto;
		overflow-wrap: anywhere;
		padding: 0 0 8px;
	}
	.live-output > span {
		color: var(--muted);
	}
	.live-output small {
		display: block;
		font-size: 12px;
		color: var(--muted);
	}
	.live-output i {
		display: inline-block;
		width: 5px;
		height: 16px;
		background: var(--accent);
		margin-left: 4px;
		vertical-align: middle;
		border-radius: 2px;
	}
	.playback-controls {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
		flex-wrap: wrap;
		margin: 6px 0;
	}
	.transport-controls,
	.pace-controls,
	.layer-track {
		display: flex;
		align-items: center;
		gap: 6px;
		flex-wrap: wrap;
	}
	button {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		gap: 5px;
		cursor: pointer;
		border: 1px solid var(--line);
		border-radius: 7px;
		background: transparent;
		color: var(--muted);
		padding: 6px 9px;
		font-size: 12px;
	}
	button:hover:not(:disabled) {
		border-color: var(--accent);
		color: var(--ink);
	}
	button:disabled {
		opacity: 0.45;
		cursor: default;
	}
	button.chosen,
	button.focused {
		color: var(--accent);
		border-color: var(--accent);
		background: var(--accent-soft);
	}
	.layer-track {
		font-size: 12px;
		color: var(--muted);
	}
	.layer-track button {
		min-width: 29px;
		padding: 5px 7px;
	}
	.layer-track small {
		font-size: 12px;
	}
	.layer-track > small,
	.layer-track > span:last-of-type {
		display: none;
	}
	.live-readout {
		display: flex;
		align-items: center;
		gap: 12px;
		flex-wrap: wrap;
		color: var(--muted);
		font-size: 12px;
		margin: 9px 0;
	}
	.live-readout output {
		color: var(--accent);
		font: 13px var(--mono);
	}
	.live-readout code {
		color: var(--ink);
	}
	.console-details {
		display: flex;
		align-items: flex-start;
		gap: 8px 20px;
		flex-wrap: wrap;
		margin-top: 8px;
	}
	details {
		min-width: 0;
		color: var(--muted);
		font-size: 13px;
		line-height: 1.6;
	}
	details[open] {
		flex: 1 1 100%;
		padding-bottom: 8px;
	}
	summary {
		cursor: pointer;
		font-size: 12px;
	}
	summary > span {
		margin-left: 8px;
		color: var(--faint);
	}
	.frame-evidence:not([open]) > summary > span,
	.trace-history:not([open]) > summary > span {
		display: none;
	}
	.pace-controls {
		margin-top: 12px;
	}
	.pace-controls > span {
		margin-right: 8px;
	}
	.measured-context {
		border: 1px solid var(--line);
		border-radius: 8px;
		padding: 12px;
		margin: 12px 0;
	}
	.measured-context > div:first-child {
		display: flex;
		flex-wrap: wrap;
		justify-content: space-between;
		gap: 8px;
	}
	.context-tail,
	.next-distribution {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		margin: 10px 0;
	}
	.context-tail code {
		padding: 3px 6px;
		border: 1px solid var(--line);
		border-radius: 5px;
	}
	.context-tail code.final {
		border-color: var(--accent);
		color: var(--accent);
	}
	.full-context p {
		white-space: pre-wrap;
		overflow-wrap: anywhere;
	}
	.next-distribution code {
		border: 1px solid var(--line);
		border-radius: 5px;
		padding: 3px 6px;
	}
	.playback-note {
		color: var(--faint);
	}
	.trace-history > div:first-of-type {
		display: flex;
		gap: 8px;
		margin: 10px 0;
	}
	.trace-tokens {
		display: flex;
		gap: 5px;
		flex-wrap: wrap;
	}
	.trace-tokens button {
		font-family: var(--mono);
	}
	@media (max-width: 600px) {
		.live-panel {
			padding: 14px 12px;
		}
		.live-prefix {
			gap: 8px;
		}
		.live-output {
			font-size: 16px;
		}
		.live-heading {
			flex-wrap: wrap;
		}
		.live-heading > span {
			margin-left: 0;
		}
	}
</style>
