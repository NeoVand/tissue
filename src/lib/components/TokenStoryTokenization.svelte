<script lang="ts">
	import type { TokenStoryTokenizer } from '$lib/token-stories/tokenizer';
	let {
		text,
		tokenizer,
		context,
		compact = false
	}: {
		text: string;
		tokenizer: TokenStoryTokenizer | null;
		context: number;
		compact?: boolean;
	} = $props();
	let encoded = $derived.by(() => {
		if (!tokenizer) return { ids: [] as number[], error: '' };
		try {
			return { ids: tokenizer.encode(text, { bos: true }), error: '' };
		} catch (reason) {
			return {
				ids: [] as number[],
				error: reason instanceof Error ? reason.message : String(reason)
			};
		}
	});
	let omitted = $derived(Math.max(0, encoded.ids.length - context));
	let visible = $derived(
		encoded.ids.slice(Math.max(0, encoded.ids.length - Math.max(context, 40)))
	);
	let retainedText = $derived(tokenizer?.decode(encoded.ids.slice(-context)) ?? '');
	function piece(id: number): string {
		return tokenizer?.tokenPiece(id).replaceAll(' ', '·').replaceAll('\n', '↵') ?? String(id);
	}
</script>

<div class:compact class="tokenization" aria-label="Subword tokenization preview">
	{#if encoded.error}<p class="token-error">{encoded.error}</p>
	{:else if tokenizer}
		<div class="encoding-meta">
			<span
				><b>{text.length}</b> characters <span aria-hidden="true">→</span>
				<b>{Math.max(0, encoded.ids.length - 1)}</b> text tokens + BOS</span
			><span>{Math.min(encoded.ids.length, context)} / {context} context tokens</span>
		</div>
		{#if !compact}<div class="pieces">
				{#each visible as id, index (`${index}-${id}`)}{@const absolute =
						encoded.ids.length - visible.length + index}<span
						class:outside={absolute < omitted}
						class:special={id === tokenizer.bosId || id === tokenizer.eosId}
						title={`Token ${absolute + 1} · vocabulary ID ${id}${absolute < omitted ? ' · outside context' : ''}`}
						>{piece(id)}</span
					>{/each}
			</div>{/if}
		{#if omitted}<p class="coverage">
				{omitted} leading tokens outside the current context. The retained suffix covers {retainedText.length}
				/ {text.length} text characters.
			</p>{:else if !compact}<p class="coverage">
				All text fits. Dots mark spaces; each chip is one token. Tokenization is deterministic and
				requires no model.
			</p>{/if}
	{:else}<p class="coverage">Loading the recorded training-vocabulary tokenizer…</p>{/if}
</div>

<style>
	.tokenization {
		padding: 11px 12px 13px;
		border-top: 1px solid var(--line);
		background: var(--surface);
		min-width: 0;
	}
	.encoding-meta {
		display: flex;
		justify-content: space-between;
		gap: 8px;
		flex-wrap: wrap;
		color: var(--muted);
		font: 12px/1.6 var(--mono);
	}
	.encoding-meta b {
		color: var(--ink);
		font-weight: 500;
	}
	.pieces {
		display: flex;
		flex-wrap: wrap;
		gap: 3px;
		margin-top: 8px;
		max-height: 124px;
		overflow: auto;
	}
	.pieces > span {
		border: 1px solid var(--line);
		border-radius: 4px;
		background: var(--surface-raised);
		padding: 4px 6px;
		color: var(--ink);
		font: 12px/1.3 var(--mono);
		overflow-wrap: anywhere;
	}
	.pieces > span:nth-child(3n + 2) {
		border-color: color-mix(in srgb, var(--layer-1) 32%, var(--line));
	}
	.pieces > span:nth-child(3n + 3) {
		border-color: color-mix(in srgb, var(--layer-2) 30%, var(--line));
	}
	.pieces > span.special {
		color: var(--accent);
		border-color: color-mix(in srgb, var(--accent) 50%, var(--line));
	}
	.pieces > span.outside {
		opacity: 0.4;
		text-decoration: line-through;
	}
	.coverage,
	.token-error {
		font: 12px/1.65 var(--mono);
		color: var(--muted);
		margin: 7px 0 0;
	}
	.token-error {
		color: var(--warning);
		margin: 0;
	}
	.compact {
		padding: 0 15px 12px;
		background: transparent;
		border-top: 0;
	}
</style>
