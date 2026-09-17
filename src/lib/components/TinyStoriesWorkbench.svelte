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

<div class="token-mode-bar">
	<div class="mode-label"><Icon name="book" size={13} /><span>Language representation</span></div>
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
	<p>Separate models, measurements and checkpoints</p>
</div>
<div hidden={view !== 'subword'}>
	<TokenStoryLab
		{theme}
		active={active && view === 'subword'}
		disabled={disabled || charactersBusy}
		onbusy={(value) => reportBusy('subword', value)}
	/>
</div>
{#if charactersVisited}
	<div hidden={view !== 'characters'}>
		<StoryLab
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
	.mode-label {
		display: flex;
		align-items: center;
		gap: 7px;
		color: var(--muted);
		font: 9px var(--mono);
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
		font-size: 10px;
		cursor: pointer;
	}
	.mode-options button.chosen {
		color: var(--ink);
		background: var(--surface-raised);
		border-color: var(--line);
	}
	.mode-options button span {
		color: var(--faint);
		font: 8px var(--mono);
	}
	.mode-options i {
		width: 4px;
		height: 4px;
		border-radius: 50%;
		background: var(--accent);
	}
	p {
		margin: 0 0 0 auto;
		color: var(--faint);
		font: 8px var(--mono);
	}
	button:focus-visible {
		outline: 1px solid var(--accent);
		outline-offset: 3px;
	}
	@media (max-width: 800px) {
		.mode-label,
		p {
			display: none;
		}
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
</style>
