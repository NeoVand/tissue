<script lang="ts">
	import { onMount, type Snippet } from 'svelte';
	import Icon from './Icon.svelte';
	let {
		name,
		children,
		model,
		inspector,
		panel = $bindable('model'),
		collapsed = $bindable(false)
	}: {
		name: string;
		children: Snippet;
		model: Snippet;
		inspector: Snippet;
		panel?: 'model' | 'inspector';
		collapsed?: boolean;
	} = $props();
	const id = $props.id();
	let width = $state(360);
	let dragging = $state(false);
	let containerWidth = $state(1200);
	let limit = $derived(Math.max(280, Math.min(560, containerWidth - 480)));
	let displayedWidth = $derived(Math.min(width, limit));
	let origin: { x: number; width: number } | null = null;
	function save() {
		try {
			localStorage.setItem(`tissue.workspace.${name}`, JSON.stringify({ width, collapsed }));
		} catch {
			/* Layout preferences are optional. */
		}
	}
	onMount(() => {
		try {
			const saved = JSON.parse(localStorage.getItem(`tissue.workspace.${name}`) ?? 'null');
			if (saved && Number.isFinite(saved.width)) width = Math.max(280, Math.min(560, saved.width));
			if (typeof saved?.collapsed === 'boolean') collapsed = saved.collapsed;
		} catch {
			/* Ignore unavailable or invalid preferences. */
		}
	});
	function resize(event: KeyboardEvent) {
		if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
		event.preventDefault();
		width =
			event.key === 'Home'
				? 280
				: event.key === 'End'
					? limit
					: Math.max(280, Math.min(limit, displayedWidth + (event.key === 'ArrowLeft' ? 24 : -24)));
		save();
	}
	function finish(event: PointerEvent) {
		if (!origin) return;
		origin = null;
		dragging = false;
		(event.currentTarget as HTMLElement).releasePointerCapture(event.pointerId);
		save();
	}
</script>

<div
	class="workspace"
	class:collapsed
	class:dragging
	bind:clientWidth={containerWidth}
	style:--panel-width={`${displayedWidth}px`}
>
	<div class="workspace-content">{@render children()}</div>
	<div class="tools">
		<div class="tool-switcher" role="group" aria-label={`${name} tools`}>
			<button
				class:chosen={panel === 'model' && !collapsed}
				aria-pressed={panel === 'model' && !collapsed}
				onclick={() => {
					panel = 'model';
					collapsed = false;
					save();
				}}><Icon name="layers" size={16} />Model & runs</button
			>
			<button
				class:chosen={panel === 'inspector' && !collapsed}
				aria-pressed={panel === 'inspector' && !collapsed}
				onclick={() => {
					panel = 'inspector';
					collapsed = false;
					save();
				}}><Icon name="target" size={16} />Inspect</button
			>
			<button
				class="collapse-button"
				aria-label={collapsed ? 'Expand tools' : 'Collapse tools'}
				aria-expanded={!collapsed}
				aria-controls={id}
				onclick={() => {
					collapsed = !collapsed;
					save();
				}}><Icon name={collapsed ? 'left' : 'right'} size={16} /></button
			>
		</div>
		<div {id} class="tool-content" hidden={collapsed}>
			<div hidden={panel !== 'model'}>{@render model()}</div>
			<div hidden={panel !== 'inspector'}>{@render inspector()}</div>
		</div>
		<div
			class="resize-handle"
			class:inactive={collapsed}
			role="slider"
			tabindex={collapsed ? -1 : 0}
			aria-label="Resize tools panel"
			aria-orientation="vertical"
			aria-controls={id}
			aria-valuemin={280}
			aria-valuemax={limit}
			aria-valuenow={Math.round(displayedWidth)}
			onkeydown={resize}
			onpointerdown={(event) => {
				if (event.button !== 0) return;
				origin = { x: event.clientX, width: displayedWidth };
				dragging = true;
				event.currentTarget.setPointerCapture(event.pointerId);
			}}
			onpointermove={(event) => {
				if (origin) width = Math.max(280, Math.min(limit, origin.width + origin.x - event.clientX));
			}}
			onpointerup={finish}
			onpointercancel={finish}
			ondblclick={() => {
				width = 360;
				save();
			}}
			title="Drag or use arrow keys to resize. Double-click to reset."
		></div>
	</div>
</div>

<style>
	.workspace {
		display: grid;
		grid-template-columns: minmax(0, 1fr) var(--panel-width);
		gap: 20px;
		align-items: start;
		padding: 0 24px 24px;
	}
	.workspace.collapsed {
		grid-template-columns: minmax(0, 1fr);
	}
	.workspace-content {
		min-width: 0;
		border: 1px solid var(--line);
		border-radius: 12px;
		overflow: hidden;
		background: var(--surface);
	}
	.tools {
		position: sticky;
		top: 20px;
		min-width: 0;
		border: 1px solid var(--line);
		border-radius: 12px;
		background: var(--surface);
	}
	.tool-switcher {
		display: flex;
		align-items: center;
		padding: 8px;
		gap: 4px;
		border-bottom: 1px solid var(--line);
	}
	.tool-switcher button {
		display: flex;
		gap: 7px;
		align-items: center;
		justify-content: center;
		border: 1px solid transparent;
		border-radius: 7px;
		background: transparent;
		color: var(--muted);
		padding: 8px;
		font-size: 13px;
		min-height: 36px;
		flex: 1;
		white-space: nowrap;
	}
	.tool-switcher .chosen {
		background: var(--surface-raised);
		border-color: var(--line);
		color: var(--ink);
	}
	.tool-switcher .collapse-button {
		flex: 0 0 32px;
	}
	.tool-content {
		max-height: calc(100dvh - 120px);
		overflow: auto;
		scrollbar-gutter: stable;
		border-radius: 0 0 12px 12px;
	}
	.resize-handle {
		position: absolute;
		left: -15px;
		top: 0;
		bottom: 0;
		width: 10px;
		cursor: col-resize;
		touch-action: none;
		border-radius: 6px;
	}
	.resize-handle::after {
		content: '';
		position: absolute;
		top: 40%;
		bottom: 40%;
		left: 4px;
		width: 2px;
		border-radius: 2px;
		background: var(--muted);
		opacity: 0.45;
	}
	.resize-handle:hover::after,
	.resize-handle:focus-visible::after,
	.dragging .resize-handle::after {
		top: 12px;
		bottom: 12px;
		opacity: 1;
		background: var(--accent);
	}
	.resize-handle:focus-visible {
		outline: 1px solid var(--accent);
	}
	.dragging {
		user-select: none;
	}
	.inactive {
		display: none;
	}
	.collapsed .tools {
		grid-row: 1;
		position: static;
		justify-self: end;
	}
	.collapsed .workspace-content {
		grid-row: 2;
	}
	.collapsed .tool-switcher {
		border-bottom: 0;
	}
	@media (max-width: 1000px) {
		.workspace {
			grid-template-columns: minmax(0, 1fr);
			padding: 0 16px 16px;
			gap: 16px;
		}
		.tools {
			position: static;
			grid-row: 1;
		}
		.tool-content {
			max-height: 420px;
		}
		.resize-handle {
			display: none;
		}
		.collapsed .tools {
			justify-self: stretch;
		}
	}
	@media (max-width: 600px) {
		.workspace {
			padding: 0 10px 12px;
		}
	}
</style>
