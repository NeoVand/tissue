<script lang="ts">
	import { HugeiconsIcon } from '@hugeicons/svelte';
	import {
		Activity01Icon,
		AiNetworkIcon,
		ArrowDown01Icon,
		ArrowLeft01Icon,
		ArrowRight01Icon,
		Atom01Icon,
		BookOpen01Icon,
		Cancel01Icon,
		ChartLineData01Icon,
		CheckmarkCircle01Icon,
		CubeIcon,
		Download01Icon,
		FilterHorizontalIcon,
		FlaskConicalIcon,
		Github01Icon,
		GridViewIcon,
		InformationCircleIcon,
		Layers01Icon,
		Maximize01Icon,
		Moon02Icon,
		Orbit01Icon,
		PauseIcon,
		PlayIcon,
		Refresh01Icon,
		Search01Icon,
		Settings02Icon,
		Sun03Icon,
		Target01Icon,
		Upload01Icon
	} from '@hugeicons/core-free-icons';
	const icons = {
		activity: Activity01Icon,
		network: AiNetworkIcon,
		down: ArrowDown01Icon,
		left: ArrowLeft01Icon,
		right: ArrowRight01Icon,
		atom: Atom01Icon,
		book: BookOpen01Icon,
		close: Cancel01Icon,
		chart: ChartLineData01Icon,
		check: CheckmarkCircle01Icon,
		cube: CubeIcon,
		download: Download01Icon,
		filter: FilterHorizontalIcon,
		flask: FlaskConicalIcon,
		github: Github01Icon,
		grid: GridViewIcon,
		info: InformationCircleIcon,
		layers: Layers01Icon,
		expand: Maximize01Icon,
		moon: Moon02Icon,
		orbit: Orbit01Icon,
		pause: PauseIcon,
		play: PlayIcon,
		reset: Refresh01Icon,
		search: Search01Icon,
		settings: Settings02Icon,
		sun: Sun03Icon,
		target: Target01Icon,
		upload: Upload01Icon
	};
	let { name, size = 16 }: { name: keyof typeof icons; size?: number } = $props();
</script>

<span class="icon" aria-hidden="true"
	>{#key name}<HugeiconsIcon icon={icons[name]} {size} strokeWidth={1.6} />{/key}</span
>

<style>
	.icon {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		flex-shrink: 0;
		vertical-align: middle;
		line-height: 0;
	}
</style>
