<script lang="ts">
	import StoryLab from './StoryLab.svelte';
	import TokenStoryLab from './TokenStoryLab.svelte';
	import Icon from './Icon.svelte';
	let {
		theme,
		active = true,
		disabled = false,
		onbusy
	}: {
		theme: 'dark' | 'light';
		active?: boolean;
		disabled?: boolean;
		onbusy?: (busy: boolean) => void;
	} = $props();
	let view = $state<'subword' | 'characters'>('subword');
	let charactersVisited = $state(false);
	let subwordBusy = $state(false);
	let charactersBusy = $state(false);
	function reportBusy(kind: 'subword' | 'characters', value: boolean) {
		if (kind === 'subword') subwordBusy = value;
		else charactersBusy = value;
		onbusy?.(subwordBusy || charactersBusy);
	}
	function selectView(next: 'subword' | 'characters') {
		view = next;
		if (next === 'characters') charactersVisited = true;
	}
</script>

{#snippet representation()}
	<div class="token-mode-bar">
		<div class="mode-options" aria-label="TinyStories language representation">
			<button
				class:chosen={view === 'subword'}
				aria-label="Subword"
				aria-pressed={view === 'subword'}
				onclick={() => selectView('subword')}
				><Icon name="layers" size={12} />Subword<span>4,096 tokens</span>{#if subwordBusy}<i
						aria-label="Subword worker active"
					></i>{/if}</button
			>
			<button
				class:chosen={view === 'characters'}
				aria-label="Characters"
				aria-pressed={view === 'characters'}
				onclick={() => selectView('characters')}
				><Icon name="grid" size={12} />Characters<span>96 tokens</span>{#if charactersBusy}<i
						aria-label="Character worker active"
					></i>{/if}</button
			>
		</div>
	</div>
{/snippet}
<div hidden={view !== 'subword'}>
	<TokenStoryLab
		{representation}
		{theme}
		active={active && view === 'subword'}
		disabled={disabled || charactersBusy}
		onbusy={(value) => reportBusy('subword', value)}
	/>
</div>
{#if charactersVisited}
	<div hidden={view !== 'characters'}>
		<StoryLab
			{representation}
			{theme}
			active={active && view === 'characters'}
			disabled={disabled || subwordBusy}
			onbusy={(value) => reportBusy('characters', value)}
		/>
	</div>
{/if}

<style>
	.token-mode-bar {
		display: flex;
		align-items: center;
		gap: 22px;
		min-height: 48px;
		padding: 8px 18px;
		border-bottom: 1px solid var(--line);
		background: var(--surface);
	}
	.mode-options {
		display: flex;
		gap: 3px;
		padding: 3px;
		border: 1px solid var(--line);
		border-radius: 6px;
	}
	.mode-options button {
		display: flex;
		align-items: center;
		gap: 7px;
		padding: 7px 10px;
		border: 1px solid transparent;
		border-radius: 3px;
		background: transparent;
		color: var(--muted);
		font-size: 12px;
		cursor: pointer;
	}
	.mode-options button.chosen {
		color: var(--ink);
		background: var(--surface-raised);
		border-color: var(--line);
	}
	.mode-options button span {
		color: var(--faint);
		font: 12px var(--mono);
	}
	.mode-options i {
		width: 4px;
		height: 4px;
		border-radius: 50%;
		background: var(--accent);
	}
	button:focus-visible {
		outline: 1px solid var(--accent);
		outline-offset: 3px;
	}
	@media (max-width: 800px) {
		.token-mode-bar {
			padding: 8px 12px;
		}
		.mode-options {
			width: 100%;
		}
		.mode-options button {
			flex: 1;
			justify-content: center;
		}
	}

	.token-mode-bar {
		min-height: 0;
		padding: 0;
		background: transparent;
		border: 0;
	}
	.mode-options {
		padding: 3px;
		border-radius: 8px;
	}
	.mode-options button {
		min-height: 36px;
		border-radius: 5px;
		font-size: 13px;
	}
	.mode-options button span {
		display: none;
	}
	@media (max-width: 600px) {
		.token-mode-bar {
			padding: 0;
		}
	}
</style>
