import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { edgeFragment, edgeVertex, nodeFragment, nodeVertex } from './field-shaders';

export interface FieldPoint {
	id: number;
	position: [number, number, number];
	layer: number;
	/** Channel within this layer; older binding views use their 128-channel layout. */
	channel?: number;
	activation: number;
	effect?: number;
}
export interface FieldData {
	points: FieldPoint[];
	edges: Array<[number, number]>;
	selected: number | null;
	mode: 'activation' | 'effect';
	theme?: 'dark' | 'light';
	layerFilter?: number | null;
	/** Playback focus only. Other layers remain faintly visible; values are not changed. */
	activeLayer?: number | null;
	/** Encode measured values relative to this frame's maximum; exact values stay in the inspector. */
	activityMode?: boolean;
}
export interface FieldStats {
	nodes: number;
	edges: number;
	drawCalls: number;
	/** CPU update/render submission time; not GPU execution time or an FPS estimate. */
	frameMs: number;
	transitioning: boolean;
}
interface Anchor {
	id: number;
	layer: number;
	channel: number;
	x: number;
	y: number;
}
interface FieldOptions {
	onselect: (id: number) => void;
	onhover: (point: FieldPoint | null, x: number, y: number) => void;
	onerror: (message: string) => void;
	onanchor?: (anchor: Anchor | null) => void;
	onstats?: (stats: FieldStats) => void;
}
type Triple = [number, number, number];
interface NodeState {
	point: FieldPoint;
	from: Triple;
	to: Triple;
	radiusFrom: number;
	radiusTo: number;
	alphaFrom: number;
	alphaTo: number;
	colorFrom: Triple;
	colorTo: Triple;
	selectionFrom: number;
	selectionTo: number;
}
interface EdgeState {
	a: number;
	b: number;
	from: number;
	to: number;
	selectedFrom: number;
	selectedTo: number;
}
const smooth = (t: number) => t * t * (3 - 2 * t);
const mix = (a: number, b: number, t: number) => a + (b - a) * t;
const mix3 = (a: Triple, b: Triple, t: number): Triple => [
	mix(a[0], b[0], t),
	mix(a[1], b[1], t),
	mix(a[2], b[2], t)
];
const PALETTES = {
	dark: [0x91aaa2, 0xa8a0b8, 0xb5ab95, 0x879eaf, 0xb58d88, 0x8daab0],
	light: [0x53776d, 0x7e708f, 0x8c7d5f, 0x5f798d, 0x995e58, 0x427b83]
};

/** All motion interpolates supplied measurements. Stable IDs survive reorder/filter/interruptions. */
export function createNeuralField(canvas: HTMLCanvasElement, options: FieldOptions) {
	const renderer = new THREE.WebGLRenderer({
		canvas,
		antialias: true,
		alpha: true,
		powerPreference: 'high-performance'
	});
	renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
	renderer.setClearColor(0x111615, 0);
	renderer.outputColorSpace = THREE.SRGBColorSpace;
	const scene = new THREE.Scene();
	const camera = new THREE.PerspectiveCamera(36, 1, 0.01, 1000);
	const controls = new OrbitControls(camera, canvas);
	const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
	controls.enableDamping = !motion.matches;
	controls.dampingFactor = 0.1;
	controls.autoRotateSpeed = 0.26;
	controls.zoomSpeed = 0.7;
	controls.rotateSpeed = 0.6;
	const progress = { value: 1 };
	const activeLayer = { value: -1 };
	const activityMode = { value: 0 };
	const nodeMaterial = new THREE.ShaderMaterial({
		vertexShader: nodeVertex,
		fragmentShader: nodeFragment,
		uniforms: {
			progress,
			activeLayer,
			activityMode,
			viewportHeight: { value: 600 },
			dark: { value: 1 }
		},
		transparent: true,
		depthWrite: true
	});
	const nodeGeometry = new THREE.InstancedBufferGeometry();
	nodeGeometry.setAttribute(
		'position',
		new THREE.Float32BufferAttribute([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0], 3)
	);
	nodeGeometry.setIndex([0, 1, 2, 0, 2, 3]);
	nodeGeometry.instanceCount = 0;
	const nodes = new THREE.Mesh(nodeGeometry, nodeMaterial);
	nodes.frustumCulled = false;
	nodes.renderOrder = 2;
	scene.add(nodes);
	const edgeMaterial = new THREE.ShaderMaterial({
		vertexShader: edgeVertex,
		fragmentShader: edgeFragment,
		uniforms: {
			progress,
			activeLayer,
			baseColor: { value: new THREE.Color(0x71857e) },
			selectedColor: { value: new THREE.Color(0xb6c8c0) }
		},
		transparent: true,
		depthWrite: false
	});
	const edgeGeometry = new THREE.BufferGeometry();
	const fibers = new THREE.LineSegments(edgeGeometry, edgeMaterial);
	fibers.frustumCulled = false;
	fibers.renderOrder = 1;
	scene.add(fibers);
	const datumGeometry = new THREE.BufferGeometry();
	const datumMaterial = new THREE.PointsMaterial({
		color: 0x52665d,
		size: 1.2,
		sizeAttenuation: false,
		transparent: true,
		opacity: 0.45,
		depthWrite: false
	});
	const datum = new THREE.Points(datumGeometry, datumMaterial);
	scene.add(datum);

	let data: FieldData = { points: [], edges: [], selected: null, mode: 'activation' };
	let nodeStates = new Map<number, NodeState>();
	let edgeStates = new Map<string, EdgeState>();
	let nodeCapacity = 0;
	let edgeCapacity = 0;
	let startTime = 0;
	let duration = 0;
	let previousFrame = 0;
	let frame = 0;
	let disposed = false;
	let visible = true;
	let hasFramed = false;
	let unit = 1;
	let lastStats = 0;
	let lastTransition = false;
	let pointerStart: [number, number] | null = null;
	let hoverFrame = 0;
	let hoverEvent: PointerEvent | null = null;
	let hovered: number | null = null;
	const center = new THREE.Vector3();
	const vector = new THREE.Vector3();
	const raycaster = new THREE.Raycaster();
	const hitSphere = new THREE.Sphere();
	const hitPoint = new THREE.Vector3();
	const color = new THREE.Color();

	function amount(now = performance.now()) {
		return duration ? smooth(Math.min(1, Math.max(0, (now - startTime) / duration))) : 1;
	}
	function schedule() {
		if (!frame && !disposed && visible) frame = requestAnimationFrame(render);
	}
	function render(now: number) {
		frame = 0;
		const began = performance.now();
		const moving = controls.update(Math.min((now - previousFrame) / 1000, 0.05));
		previousFrame = now;
		progress.value = amount(now);
		const transitioning = progress.value < 1;
		renderer.render(scene, camera);
		const selected = data.selected === null ? undefined : nodeStates.get(data.selected);
		if (selected && selected.alphaTo > 0) {
			vector.set(...mix3(selected.from, selected.to, progress.value)).project(camera);
			const width = canvas.clientWidth;
			const height = canvas.clientHeight;
			const x = ((vector.x + 1) * width) / 2;
			const y = ((1 - vector.y) * height) / 2;
			options.onanchor?.(
				vector.z > -1 && vector.z < 1 && x > 0 && x < width && y > 0 && y < height
					? {
							id: selected.point.id,
							layer: selected.point.layer,
							channel: selected.point.channel ?? selected.point.id % 128,
							x,
							y
						}
					: null
			);
		} else options.onanchor?.(null);
		const cpu = performance.now() - began;
		if (now - lastStats > 800 || transitioning !== lastTransition) {
			options.onstats?.({
				nodes: data.points.filter((p) => data.layerFilter == null || p.layer === data.layerFilter)
					.length,
				edges: [...edgeStates.values()].filter((e) => e.to > 0).length,
				drawCalls: renderer.info.render.calls,
				frameMs: cpu,
				transitioning
			});
			lastStats = now;
			lastTransition = transitioning;
		}
		if (transitioning || moving || controls.autoRotate) schedule();
	}
	function reset() {
		const bounds = new THREE.Box3();
		for (const point of data.points) bounds.expandByPoint(vector.set(...point.position));
		if (!bounds.isEmpty()) {
			bounds.getCenter(center);
			const size = bounds.getSize(vector);
			unit = Math.max(size.x, size.y, size.z, 0.5) / 2;
		} else {
			center.set(0, 0, 0);
			unit = 1;
		}
		const direction = new THREE.Vector3(0.54, 0.32, 2.8).normalize();
		const right = new THREE.Vector3().crossVectors(camera.up, direction).normalize();
		const up = new THREE.Vector3().crossVectors(direction, right).normalize();
		const focal = 1 / Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
		let distance = unit * 0.6;
		// Fit the actual measured coordinates to 78% of the vertical viewport, constrained
		// to 86% width. Only initialization and an explicit Frame action change the camera.
		for (const point of data.points) {
			vector.set(...point.position).sub(center);
			const depth = vector.dot(direction);
			distance = Math.max(
				distance,
				depth + (Math.abs(vector.dot(up)) * focal) / 0.78,
				depth + (Math.abs(vector.dot(right)) * focal) / (camera.aspect * 0.86)
			);
		}
		if (!data.points.length) distance = unit * 4;
		camera.position.copy(center).addScaledVector(direction, distance);
		controls.target.copy(center);
		controls.minDistance = unit * 0.3;
		controls.maxDistance = unit * 18;
		camera.near = unit * 0.003;
		camera.far = unit * 60;
		camera.updateProjectionMatrix();
		controls.update();
		const dots: number[] = [];
		const y = bounds.isEmpty() ? -1.15 : bounds.min.y - unit * 0.28;
		for (let x = -5; x <= 5; x++)
			for (let z = -5; z <= 5; z++)
				dots.push(center.x + x * unit * 0.23, y, center.z + z * unit * 0.23);
		datumGeometry.dispose();
		datumGeometry.setAttribute('position', new THREE.Float32BufferAttribute(dots, 3));
		schedule();
	}
	function allocateNodes(count: number) {
		if (count <= nodeCapacity) return;
		nodeCapacity = Math.max(256, 2 ** Math.ceil(Math.log2(count)));
		nodeGeometry.dispose();
		for (const [name, size] of [
			['fromPosition', 3],
			['toPosition', 3],
			['radiusAlpha', 4],
			['fromColor', 3],
			['toColor', 3],
			['emphasis', 2],
			['layerIndex', 1]
		] as const) {
			nodeGeometry.setAttribute(
				name,
				new THREE.InstancedBufferAttribute(new Float32Array(nodeCapacity * size), size).setUsage(
					THREE.DynamicDrawUsage
				)
			);
		}
	}
	function allocateEdges(count: number) {
		if (count <= edgeCapacity) return;
		edgeCapacity = Math.max(512, 2 ** Math.ceil(Math.log2(count)));
		edgeGeometry.dispose();
		// The base position attribute determines vertex capacity; positions are moved in the vertex shader.
		for (const [name, size] of [
			['position', 3],
			['fromPosition', 3],
			['toPosition', 3],
			['opacity', 2],
			['emphasis', 2],
			['endpointLayers', 2]
		] as const) {
			edgeGeometry.setAttribute(
				name,
				new THREE.BufferAttribute(new Float32Array(edgeCapacity * 2 * size), size).setUsage(
					THREE.DynamicDrawUsage
				)
			);
		}
	}
	function update(next: FieldData) {
		activeLayer.value = next.activeLayer ?? -1;
		activityMode.value = Number(Boolean(next.activityMode));
		// Faint context beads must not write opaque depth over measured activity behind them.
		// Retain ordinary depth occlusion for static structural snapshots.
		nodeMaterial.depthWrite = !next.activityMode;
		// Layer stepping changes only shader uniforms. It never restarts a coordinate
		// interpolation or uploads thousands of unchanged neuron attributes.
		if (
			next.points === data.points &&
			next.edges === data.edges &&
			next.selected === data.selected &&
			next.mode === data.mode &&
			next.theme === data.theme &&
			next.layerFilter === data.layerFilter &&
			next.activityMode === data.activityMode
		) {
			data = next;
			hovered = null;
			options.onhover(null, 0, 0);
			schedule();
			return;
		}
		const now = performance.now();
		const t = amount(now);
		const first = !hasFramed;
		data = next;
		if (!hasFramed && next.points.length) {
			hasFramed = true;
			reset();
		}
		const theme = next.theme ?? 'dark';
		nodeMaterial.uniforms.dark.value = theme === 'dark' ? 1 : 0;
		edgeMaterial.uniforms.baseColor.value.setHex(theme === 'dark' ? 0x71857e : 0x708178);
		edgeMaterial.uniforms.selectedColor.value.setHex(theme === 'dark' ? 0xc4d0c8 : 0x435e51);
		datumMaterial.color.setHex(theme === 'dark' ? 0x52645c : 0x9ba69e);
		const neighbors = new Set<number>();
		for (const [a, b] of next.edges) {
			if (a === next.selected) neighbors.add(b);
			if (b === next.selected) neighbors.add(a);
		}
		const nextStates = new Map<number, NodeState>();
		let maximum = 0;
		if (next.activityMode)
			for (const point of next.points)
				if (next.layerFilter == null || next.layerFilter === point.layer)
					maximum = Math.max(
						maximum,
						Math.abs(next.mode === 'effect' ? (point.effect ?? 0) : point.activation)
					);
		for (const point of next.points) {
			const old = nodeStates.get(point.id);
			const selected = point.id === next.selected;
			const magnitude = Math.abs(next.mode === 'effect' ? (point.effect ?? 0) : point.activation);
			const signal = maximum > 0 ? Math.sqrt(magnitude / maximum) : 0;
			const radius = next.activityMode
				? unit * (0.005 + 0.031 * signal)
				: unit * (0.011 + 0.023 * Math.tanh(magnitude / (next.mode === 'effect' ? 0.12 : 2)));
			const active = next.layerFilter == null || next.layerFilter === point.layer;
			const opacity = !active
				? 0
				: next.activityMode
					? 0.07 + 0.93 * signal
					: next.selected !== null && !selected && !neighbors.has(point.id)
						? 0.7
						: 1;
			color.setHex(PALETTES[theme][point.layer % PALETTES[theme].length]);
			if (selected) color.setHex(theme === 'dark' ? 0xdee7d3 : 0x5f7856);
			const rgb: Triple = [color.r, color.g, color.b];
			nextStates.set(point.id, {
				point,
				from: old ? mix3(old.from, old.to, t) : [...point.position],
				to: [...point.position],
				radiusFrom: old ? mix(old.radiusFrom, old.radiusTo, t) : radius,
				radiusTo: radius,
				alphaFrom: old ? mix(old.alphaFrom, old.alphaTo, t) : first ? opacity : 0,
				alphaTo: opacity,
				colorFrom: old ? mix3(old.colorFrom, old.colorTo, t) : rgb,
				colorTo: rgb,
				selectionFrom: old ? mix(old.selectionFrom, old.selectionTo, t) : Number(selected),
				selectionTo: Number(selected)
			});
		}
		for (const [id, old] of nodeStates)
			if (!nextStates.has(id)) {
				const alpha = mix(old.alphaFrom, old.alphaTo, t);
				if (alpha > 0.001)
					nextStates.set(id, {
						...old,
						from: mix3(old.from, old.to, t),
						radiusFrom: mix(old.radiusFrom, old.radiusTo, t),
						alphaFrom: alpha,
						alphaTo: 0,
						colorFrom: mix3(old.colorFrom, old.colorTo, t),
						selectionFrom: mix(old.selectionFrom, old.selectionTo, t),
						selectionTo: 0
					});
			}
		nodeStates = nextStates;
		allocateNodes(nodeStates.size || 1);
		let i = 0;
		for (const node of nodeStates.values()) {
			const put = (name: string, values: number[]) =>
				(nodeGeometry.getAttribute(name).array as Float32Array).set(values, i * values.length);
			put('fromPosition', node.from);
			put('toPosition', node.to);
			put('radiusAlpha', [node.radiusFrom, node.radiusTo, node.alphaFrom, node.alphaTo]);
			put('fromColor', node.colorFrom);
			put('toColor', node.colorTo);
			put('emphasis', [node.selectionFrom, node.selectionTo]);
			put('layerIndex', [node.point.layer]);
			i++;
		}
		nodeGeometry.instanceCount = nodeStates.size;
		for (const [name, attribute] of Object.entries(nodeGeometry.attributes))
			if (name !== 'position') attribute.needsUpdate = true;
		const nextEdges = new Map<string, EdgeState>();
		for (const pair of next.edges) {
			const a = Math.min(...pair);
			const b = Math.max(...pair);
			if (a === b) continue;
			const start = nodeStates.get(a);
			const end = nodeStates.get(b);
			if (!start || !end) continue;
			const isSelected = a === next.selected || b === next.selected;
			const allowed =
				start.alphaTo > 0 &&
				end.alphaTo > 0 &&
				(next.edges.length <= Math.max(500, next.points.length * 5) || isSelected);
			const opacity = !allowed ? 0 : isSelected ? 0.82 : next.selected === null ? 0.17 : 0.08;
			const key = `${a}:${b}`;
			const old = edgeStates.get(key);
			nextEdges.set(key, {
				a,
				b,
				from: old ? mix(old.from, old.to, t) : first ? opacity : 0,
				to: opacity,
				selectedFrom: old ? mix(old.selectedFrom, old.selectedTo, t) : Number(isSelected),
				selectedTo: Number(isSelected)
			});
		}
		for (const [key, old] of edgeStates)
			if (!nextEdges.has(key) && nodeStates.has(old.a) && nodeStates.has(old.b)) {
				const opacity = mix(old.from, old.to, t);
				if (opacity > 0.001)
					nextEdges.set(key, {
						...old,
						from: opacity,
						to: 0,
						selectedFrom: mix(old.selectedFrom, old.selectedTo, t),
						selectedTo: 0
					});
			}
		edgeStates = nextEdges;
		allocateEdges(edgeStates.size || 1);
		i = 0;
		for (const edge of edgeStates.values())
			for (const id of [edge.a, edge.b]) {
				const node = nodeStates.get(id)!;
				(edgeGeometry.getAttribute('fromPosition').array as Float32Array).set(node.from, i * 3);
				(edgeGeometry.getAttribute('toPosition').array as Float32Array).set(node.to, i * 3);
				(edgeGeometry.getAttribute('opacity').array as Float32Array).set(
					[edge.from, edge.to],
					i * 2
				);
				(edgeGeometry.getAttribute('emphasis').array as Float32Array).set(
					[edge.selectedFrom, edge.selectedTo],
					i * 2
				);
				(edgeGeometry.getAttribute('endpointLayers').array as Float32Array).set(
					[nodeStates.get(edge.a)!.point.layer, nodeStates.get(edge.b)!.point.layer],
					i * 2
				);
				i++;
			}
		edgeGeometry.setDrawRange(0, edgeStates.size * 2);
		for (const [name, attribute] of Object.entries(edgeGeometry.attributes))
			if (name !== 'position') attribute.needsUpdate = true;
		startTime = now;
		// A live token's values must be visible during even the shortest layer dwell.
		// Do not blend in the previous token's activity and present it as this frame.
		duration = motion.matches || first || next.activityMode ? 0 : 480;
		progress.value = duration ? 0 : 1;
		hovered = null;
		options.onhover(null, 0, 0);
		schedule();
	}
	function hit(event: PointerEvent) {
		const rect = canvas.getBoundingClientRect();
		raycaster.setFromCamera(
			new THREE.Vector2(
				((event.clientX - rect.left) / rect.width) * 2 - 1,
				(-(event.clientY - rect.top) / rect.height) * 2 + 1
			),
			camera
		);
		const t = progress.value; // Same interpolation state as the last displayed GPU frame.
		let closest: FieldPoint | null = null;
		let distance = Infinity;
		for (const node of nodeStates.values()) {
			if (data.activeLayer != null && node.point.layer !== data.activeLayer) continue;
			if (node.alphaTo <= 0 || mix(node.alphaFrom, node.alphaTo, t) < 0.15) continue;
			hitSphere.center.set(...mix3(node.from, node.to, t));
			const viewDistance = camera.position.distanceTo(hitSphere.center);
			const targetPixels = event.pointerType === 'touch' ? 13 : 7;
			const targetWorld =
				(targetPixels * 2 * viewDistance) / (rect.height * camera.projectionMatrix.elements[5]);
			hitSphere.radius = Math.max(mix(node.radiusFrom, node.radiusTo, t), targetWorld);
			if (raycaster.ray.intersectSphere(hitSphere, hitPoint)) {
				const currentDistance = camera.position.distanceToSquared(hitPoint);
				if (currentDistance < distance) {
					closest = node.point;
					distance = currentDistance;
				}
			}
		}
		return closest;
	}
	function pointerDown(event: PointerEvent) {
		cancelAnimationFrame(hoverFrame);
		hoverFrame = 0;
		hoverEvent = null;
		pointerStart = [event.clientX, event.clientY];
		options.onhover(null, 0, 0);
	}
	function pointerUp(event: PointerEvent) {
		if (
			pointerStart &&
			Math.hypot(event.clientX - pointerStart[0], event.clientY - pointerStart[1]) < 5
		) {
			const point = hit(event);
			if (point) options.onselect(point.id);
		}
		pointerStart = null;
	}
	function pointerMove(event: PointerEvent) {
		if (pointerStart) return;
		hoverEvent = event;
		// Exact ray tests remain O(N), but a high-rate pointer never triggers several scans per frame.
		if (hoverFrame) return;
		hoverFrame = requestAnimationFrame(() => {
			hoverFrame = 0;
			const latest = hoverEvent;
			hoverEvent = null;
			if (!disposed && !pointerStart && latest) updateHover(latest);
		});
	}
	function updateHover(event: PointerEvent) {
		const point = hit(event);
		canvas.style.cursor = point ? 'pointer' : 'grab';
		if ((point?.id ?? null) !== hovered) {
			hovered = point?.id ?? null;
			const rect = canvas.getBoundingClientRect();
			options.onhover(
				point,
				Math.max(76, Math.min(rect.width - 76, event.clientX - rect.left)),
				Math.max(70, event.clientY - rect.top)
			);
		}
	}
	function pointerLeave() {
		cancelAnimationFrame(hoverFrame);
		hoverFrame = 0;
		hoverEvent = null;
		pointerStart = null;
		hovered = null;
		options.onhover(null, 0, 0);
	}
	function contextLost(event: Event) {
		event.preventDefault();
		options.onerror(
			'Graphics interrupted. Reload to restore the view; your measurements remain in the journal.'
		);
	}
	function motionChange() {
		controls.enableDamping = !motion.matches;
		if (motion.matches) {
			duration = 0;
			controls.autoRotate = false;
		}
		schedule();
	}
	function resize() {
		const { width, height } = canvas.getBoundingClientRect();
		if (width < 1 || height < 1) return;
		camera.aspect = width / height;
		camera.updateProjectionMatrix();
		renderer.setSize(width, height, false);
		nodeMaterial.uniforms.viewportHeight.value = height;
		schedule();
	}
	const observer = new ResizeObserver(resize);
	observer.observe(canvas);
	const visibilityObserver = new IntersectionObserver(([entry]) => {
		visible = entry.isIntersecting;
		if (visible) schedule();
		else {
			cancelAnimationFrame(frame);
			frame = 0;
		}
	});
	visibilityObserver.observe(canvas);
	controls.addEventListener('change', schedule);
	motion.addEventListener('change', motionChange);
	canvas.addEventListener('pointerdown', pointerDown);
	canvas.addEventListener('pointerup', pointerUp);
	canvas.addEventListener('pointermove', pointerMove);
	canvas.addEventListener('pointerleave', pointerLeave);
	canvas.addEventListener('webglcontextlost', contextLost);
	resize();
	reset();
	return {
		update,
		reset() {
			reset();
			// Recompute radii after the explicit camera/coordinate-scale reset.
			update({ ...data, points: [...data.points] });
		},
		rotate(enabled: boolean) {
			controls.autoRotate = enabled;
			schedule();
		},
		/** A measured view snapshot, useful for verification without inventing motion. */
		inspect() {
			return {
				activeLayer: data.activeLayer ?? null,
				activityMode: Boolean(data.activityMode),
				progress: progress.value,
				camera: camera.position.toArray(),
				nodes: [...nodeStates].map(([id, node]) => ({
					id,
					position: mix3(node.from, node.to, progress.value),
					target: node.to,
					radius: mix(node.radiusFrom, node.radiusTo, progress.value),
					alpha: mix(node.alphaFrom, node.alphaTo, progress.value)
				}))
			};
		},
		destroy() {
			disposed = true;
			cancelAnimationFrame(frame);
			cancelAnimationFrame(hoverFrame);
			observer.disconnect();
			visibilityObserver.disconnect();
			controls.removeEventListener('change', schedule);
			controls.dispose();
			motion.removeEventListener('change', motionChange);
			canvas.removeEventListener('pointerdown', pointerDown);
			canvas.removeEventListener('pointerup', pointerUp);
			canvas.removeEventListener('pointermove', pointerMove);
			canvas.removeEventListener('pointerleave', pointerLeave);
			canvas.removeEventListener('webglcontextlost', contextLost);
			nodeGeometry.dispose();
			nodeMaterial.dispose();
			edgeGeometry.dispose();
			edgeMaterial.dispose();
			datumGeometry.dispose();
			datumMaterial.dispose();
			renderer.dispose();
		}
	};
}
