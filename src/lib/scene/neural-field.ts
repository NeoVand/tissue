import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export interface FieldPoint {
	id: number;
	position: [number, number, number];
	layer: number;
	activation: number;
	effect?: number;
}

export interface FieldData {
	points: FieldPoint[];
	edges: Array<[number, number]>;
	selected: number | null;
	mode: 'activation' | 'effect';
}

interface FieldOptions {
	onselect: (id: number) => void;
	onhover: (point: FieldPoint | null, x: number, y: number) => void;
	onerror: (message: string) => void;
}

const LAYER_COLORS = [0xb75a3c, 0x3b7468, 0x858346, 0x746086];

/** A view of supplied measurements. This renderer never creates or relaxes node positions. */
export function createNeuralField(canvas: HTMLCanvasElement, options: FieldOptions) {
	const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
	renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
	renderer.setClearColor(0xf5f3eb, 0);
	renderer.outputColorSpace = THREE.SRGBColorSpace;
	const scene = new THREE.Scene();
	const camera = new THREE.PerspectiveCamera(37, 1, 0.01, 1000);
	const controls = new OrbitControls(camera, canvas);
	const motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)');
	controls.enableDamping = !motionPreference.matches;
	controls.dampingFactor = 0.09;
	controls.autoRotateSpeed = 0.32;
	controls.enablePan = true;
	controls.zoomSpeed = 0.7;
	controls.rotateSpeed = 0.6;
	controls.minDistance = 0.3;
	controls.maxDistance = 100;
	scene.add(new THREE.HemisphereLight(0xffffff, 0xb2a995, 2.4));
	const light = new THREE.DirectionalLight(0xfffbf1, 2.5);
	light.position.set(-3, 6, 5);
	scene.add(light);

	const sphere = new THREE.SphereGeometry(1, 16, 12);
	const beadMaterial = new THREE.MeshStandardMaterial({
		color: 0xffffff,
		roughness: 0.32,
		metalness: 0.08
	});
	let beads = new THREE.InstancedMesh(sphere, beadMaterial, 1);
	beads.count = 0;
	beads.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
	scene.add(beads);
	const fibers = new THREE.LineSegments(
		new THREE.BufferGeometry(),
		new THREE.LineBasicMaterial({
			color: 0x6f796b,
			transparent: true,
			opacity: 0.16,
			depthWrite: false
		})
	);
	const selectedFibers = new THREE.LineSegments(
		new THREE.BufferGeometry(),
		new THREE.LineBasicMaterial({ color: 0x485b43, transparent: true, opacity: 0.64 })
	);
	scene.add(fibers, selectedFibers);

	const ring = new THREE.Mesh(
		new THREE.RingGeometry(1, 1.075, 64),
		new THREE.MeshBasicMaterial({ color: 0x42503c, side: THREE.DoubleSide })
	);
	ring.visible = false;
	scene.add(ring);
	const projection = new THREE.Line(
		new THREE.BufferGeometry(),
		new THREE.LineDashedMaterial({
			color: 0x7d8772,
			dashSize: 0.025,
			gapSize: 0.025,
			transparent: true,
			opacity: 0.48
		})
	);
	scene.add(projection);
	const datum = new THREE.Group();
	scene.add(datum);

	let data: FieldData = { points: [], edges: [], selected: null, mode: 'activation' };
	let frame = 0;
	let disposed = false;
	let visible = true;
	let capacity = 1;
	let unit = 1;
	let floorY = -1.2;
	let hasFramed = false;
	let previousTime = 0;
	const center = new THREE.Vector3();
	const dummy = new THREE.Object3D();
	const color = new THREE.Color();
	const pale = new THREE.Color(0xcecfc1);
	const highlight = new THREE.Color(0xd4e66d);
	const pointer = new THREE.Vector2();
	const raycaster = new THREE.Raycaster();
	let pointerStart: [number, number] | null = null;
	let hoverId: number | null = null;

	function schedule() {
		if (!frame && !disposed && visible) frame = requestAnimationFrame(render);
	}

	function render(time: number) {
		frame = 0;
		const dt = Math.min((time - previousTime) / 1000, 0.05);
		previousTime = time;
		const moving = controls.update(dt);
		ring.quaternion.copy(camera.quaternion);
		renderer.render(scene, camera);
		if (controls.autoRotate || moving) schedule();
	}

	function clearDatum() {
		for (const child of [...datum.children]) {
			if (child instanceof THREE.LineSegments || child instanceof THREE.Points) {
				child.geometry.dispose();
				(child.material as THREE.Material).dispose();
			}
			datum.remove(child);
		}
	}

	function buildDatum() {
		clearDatum();
		const coordinates: number[] = [];
		const ticks: number[] = [];
		const radius = unit * 1.24;
		const step = radius / 6;
		for (let x = -6; x <= 6; x++) {
			for (let z = -6; z <= 6; z++) {
				coordinates.push(center.x + x * step, floorY, center.z + z * step);
			}
			const value = x * step;
			ticks.push(
				center.x + value,
				floorY,
				center.z - radius - step * 0.1,
				center.x + value,
				floorY,
				center.z - radius + step * 0.1,
				center.x - radius - step * 0.1,
				floorY,
				center.z + value,
				center.x - radius + step * 0.1,
				floorY,
				center.z + value
			);
		}
		const dots = new THREE.Points(
			new THREE.BufferGeometry().setAttribute(
				'position',
				new THREE.Float32BufferAttribute(coordinates, 3)
			),
			new THREE.PointsMaterial({
				color: 0x929988,
				size: 1.7,
				sizeAttenuation: false,
				transparent: true,
				opacity: 0.4
			})
		);
		const marks = new THREE.LineSegments(
			new THREE.BufferGeometry().setAttribute(
				'position',
				new THREE.Float32BufferAttribute(ticks, 3)
			),
			new THREE.LineBasicMaterial({ color: 0x999c8c, transparent: true, opacity: 0.4 })
		);
		datum.add(dots, marks);
	}

	function reset() {
		if (data.points.length) {
			const bounds = new THREE.Box3();
			for (const point of data.points) bounds.expandByPoint(new THREE.Vector3(...point.position));
			bounds.getCenter(center);
			const size = bounds.getSize(new THREE.Vector3());
			unit = Math.max(size.x, size.y, size.z, 0.5) * 0.5;
			floorY = bounds.min.y - unit * 0.22;
		} else {
			center.set(0, 0, 0);
			unit = 1;
			floorY = -1.2;
		}
		const fit = Math.max(1, 1 / camera.aspect);
		camera.position
			.copy(center)
			.add(new THREE.Vector3(0.6, 0.38, 2.8).multiplyScalar(unit * 1.75 * fit));
		controls.target.copy(center);
		controls.minDistance = unit * 0.35;
		controls.maxDistance = unit * 18;
		camera.near = unit * 0.005;
		camera.far = unit * 50;
		camera.updateProjectionMatrix();
		controls.update();
		buildDatum();
		schedule();
	}

	/** Fixed monotone mapping: token changes do not silently renormalize bead magnitudes. */
	function beadRadius(point: FieldPoint) {
		const magnitude = Math.abs(data.mode === 'effect' ? (point.effect ?? 0) : point.activation);
		return unit * (0.016 + 0.022 * Math.tanh(magnitude / (data.mode === 'effect' ? 0.12 : 2)));
	}

	function update(next: FieldData) {
		data = next;
		if (!hasFramed && next.points.length) {
			hasFramed = true;
			reset();
		}
		if (next.points.length > capacity) {
			scene.remove(beads);
			beads.dispose();
			capacity = Math.max(next.points.length, capacity * 2);
			beads = new THREE.InstancedMesh(sphere, beadMaterial, capacity);
			beads.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
			scene.add(beads);
		}
		const byId = new Map(next.points.map((point) => [point.id, point]));
		const neighbors = new Set<number>();
		for (const [a, b] of next.edges) {
			if (a === next.selected) neighbors.add(b);
			if (b === next.selected) neighbors.add(a);
		}
		beads.count = next.points.length;
		for (let i = 0; i < next.points.length; i++) {
			const point = next.points[i];
			dummy.position.set(...point.position);
			dummy.scale.setScalar(beadRadius(point) * (point.id === next.selected ? 1.2 : 1));
			dummy.updateMatrix();
			beads.setMatrixAt(i, dummy.matrix);
			color.setHex(
				LAYER_COLORS[
					((point.layer % LAYER_COLORS.length) + LAYER_COLORS.length) % LAYER_COLORS.length
				]
			);
			if (next.selected !== null && point.id !== next.selected && !neighbors.has(point.id))
				color.lerp(pale, 0.5);
			if (point.id === next.selected) color.copy(highlight);
			beads.setColorAt(i, color);
		}
		beads.instanceMatrix.needsUpdate = true;
		if (beads.instanceColor) beads.instanceColor.needsUpdate = true;
		beads.computeBoundingSphere();
		const backgroundSegments: number[] = [];
		const foregroundSegments: number[] = [];
		for (const [a, b] of next.edges) {
			const start = byId.get(a);
			const end = byId.get(b);
			if (!start || !end) continue;
			if (a === next.selected || b === next.selected) {
				foregroundSegments.push(...start.position, ...end.position);
			} else if (next.edges.length <= Math.max(500, next.points.length * 5)) {
				// Dense inputs show only the selected neighborhood, never a complete hairball.
				backgroundSegments.push(...start.position, ...end.position);
			}
		}
		setLinePositions(fibers.geometry, backgroundSegments);
		setLinePositions(selectedFibers.geometry, foregroundSegments);
		fibers.material.opacity = next.selected === null ? 0.19 : 0.08;
		const chosen = next.selected === null ? undefined : byId.get(next.selected);
		ring.visible = Boolean(chosen);
		projection.visible = Boolean(chosen);
		if (chosen) {
			ring.position.set(...chosen.position);
			ring.scale.setScalar(beadRadius(chosen) * 2.2);
			setLinePositions(projection.geometry, [
				...chosen.position,
				chosen.position[0],
				floorY,
				chosen.position[2]
			]);
			projection.material.dashSize = unit * 0.018;
			projection.material.gapSize = unit * 0.02;
			projection.computeLineDistances();
		}
		schedule();
	}

	function setLinePositions(geometry: THREE.BufferGeometry, positions: number[]) {
		const current = geometry.getAttribute('position');
		if (current && current.count * 3 === positions.length) {
			(current.array as Float32Array).set(positions);
			current.needsUpdate = true;
		} else {
			// Release the old GPU attribute before changing its capacity.
			if (current) geometry.dispose();
			geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
		}
		geometry.computeBoundingSphere();
	}

	function hit(event: PointerEvent) {
		const rect = canvas.getBoundingClientRect();
		pointer.set(
			((event.clientX - rect.left) / rect.width) * 2 - 1,
			-((event.clientY - rect.top) / rect.height) * 2 + 1
		);
		raycaster.setFromCamera(pointer, camera);
		const intersection = raycaster.intersectObject(beads, false)[0];
		return intersection?.instanceId === undefined ? null : data.points[intersection.instanceId];
	}

	function pointerDown(event: PointerEvent) {
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
		const point = hit(event);
		canvas.style.cursor = point ? 'pointer' : 'grab';
		if (point?.id !== hoverId) {
			hoverId = point?.id ?? null;
			const rect = canvas.getBoundingClientRect();
			options.onhover(point, event.clientX - rect.left, event.clientY - rect.top);
		}
	}

	function pointerLeave() {
		pointerStart = null;
		hoverId = null;
		options.onhover(null, 0, 0);
	}

	function contextLost(event: Event) {
		event.preventDefault();
		options.onerror(
			'The graphics context was interrupted. Reload this page to restore the 3D view.'
		);
	}

	function resize() {
		const { width, height } = canvas.getBoundingClientRect();
		if (width < 1 || height < 1) return;
		camera.aspect = width / height;
		camera.updateProjectionMatrix();
		renderer.setSize(width, height, false);
		schedule();
	}

	const observer = new ResizeObserver(resize);
	observer.observe(canvas);
	const visibilityObserver = new IntersectionObserver(([entry]) => {
		visible = entry.isIntersecting;
		if (visible) schedule();
		else if (frame) {
			cancelAnimationFrame(frame);
			frame = 0;
		}
	});
	visibilityObserver.observe(canvas);
	controls.addEventListener('change', schedule);
	canvas.addEventListener('pointerdown', pointerDown);
	canvas.addEventListener('pointerup', pointerUp);
	canvas.addEventListener('pointermove', pointerMove);
	canvas.addEventListener('pointerleave', pointerLeave);
	canvas.addEventListener('webglcontextlost', contextLost);
	resize();
	reset();

	return {
		update,
		reset,
		rotate(enabled: boolean) {
			controls.autoRotate = enabled;
			schedule();
		},
		destroy() {
			disposed = true;
			cancelAnimationFrame(frame);
			observer.disconnect();
			visibilityObserver.disconnect();
			controls.removeEventListener('change', schedule);
			controls.dispose();
			canvas.removeEventListener('pointerdown', pointerDown);
			canvas.removeEventListener('pointerup', pointerUp);
			canvas.removeEventListener('pointermove', pointerMove);
			canvas.removeEventListener('pointerleave', pointerLeave);
			canvas.removeEventListener('webglcontextlost', contextLost);
			clearDatum();
			beads.dispose();
			sphere.dispose();
			beadMaterial.dispose();
			for (const object of [fibers, selectedFibers, projection, ring]) {
				object.geometry.dispose();
				object.material.dispose();
			}
			renderer.dispose();
		}
	};
}
