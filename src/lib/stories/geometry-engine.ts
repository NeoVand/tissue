import type { StoryAtlas } from './protocol';
import type { StoryGeometryResult, StoryNeighbor } from './geometry';

type Operation =
	| { op: 'build'; atlas: StoryAtlas }
	| { op: 'setAtlas'; atlas: StoryAtlas }
	| { op: 'neighbors'; unit: number; k: number; layer?: number };
export type StoryGeometryRequest = Operation & { id: number; generation: number };
export type StoryGeometryResponse =
	| {
			id: number;
			generation: number;
			ok: true;
			result: StoryGeometryResult | StoryNeighbor[] | null;
	  }
	| { id: number; generation: number; ok: false; error: string };
interface Pending {
	resolve: (result: StoryGeometryResult | StoryNeighbor[] | null) => void;
	reject: (error: Error) => void;
}

/** Browser-only O(NP) geometry worker. Termination immediately cancels pending fits. */
export class StoryGeometryEngine {
	private worker: Worker;
	private pending = new Map<number, Pending>();
	private nextId = 1;
	private generation = 0;
	private closed = false;

	constructor() {
		this.worker = this.createWorker();
	}
	private createWorker(): Worker {
		const worker = new Worker(new URL('./geometry-worker.ts', import.meta.url), { type: 'module' });
		const generation = this.generation;
		worker.onmessage = (event: MessageEvent<StoryGeometryResponse>) => {
			const response = event.data;
			if (this.closed || generation !== this.generation || response.generation !== generation)
				return;
			const pending = this.pending.get(response.id);
			if (!pending) return;
			this.pending.delete(response.id);
			if (response.ok) pending.resolve(response.result);
			else pending.reject(new Error(response.error));
		};
		worker.onerror = (event) => {
			if (!this.closed && generation === this.generation)
				this.shutdown(new Error(event.message || 'Story geometry worker failed'));
		};
		worker.onmessageerror = () => {
			if (!this.closed && generation === this.generation)
				this.shutdown(new Error('Unable to decode story geometry response'));
		};
		return worker;
	}
	private rejectPending(error: Error) {
		for (const pending of this.pending.values()) pending.reject(error);
		this.pending.clear();
	}
	private shutdown(error: Error) {
		this.closed = true;
		this.rejectPending(error);
		this.worker.terminate();
	}
	private call(operation: Operation): Promise<StoryGeometryResult | StoryNeighbor[] | null> {
		if (this.closed) return Promise.reject(new Error('Story geometry engine is disposed'));
		const id = this.nextId++;
		return new Promise((resolve, reject) => {
			this.pending.set(id, { resolve, reject });
			try {
				// Retain the caller's raw atlas for archiving/probes; transfer only this private copy.
				if ('atlas' in operation) {
					const fingerprints = operation.atlas.fingerprints.slice();
					this.worker.postMessage(
						{
							...operation,
							atlas: { ...operation.atlas, fingerprints },
							id,
							generation: this.generation
						} satisfies StoryGeometryRequest,
						[fingerprints.buffer]
					);
				} else
					this.worker.postMessage({
						...operation,
						id,
						generation: this.generation
					} satisfies StoryGeometryRequest);
			} catch (error) {
				this.pending.delete(id);
				reject(error instanceof Error ? error : new Error(String(error)));
			}
		});
	}
	async build(atlas: StoryAtlas): Promise<StoryGeometryResult> {
		return (await this.call({ op: 'build', atlas })) as StoryGeometryResult;
	}
	/** Load original-space fingerprints for a saved map without recomputing its coordinates. */
	async setAtlas(atlas: StoryAtlas): Promise<void> {
		await this.call({ op: 'setAtlas', atlas });
	}
	async neighbors(unit: number, k = 6, layer?: number): Promise<StoryNeighbor[]> {
		return (await this.call({ op: 'neighbors', unit, k, layer })) as StoryNeighbor[];
	}
	reset(): void {
		if (this.closed) throw new Error('Story geometry engine is disposed');
		this.generation++;
		this.rejectPending(new Error('Story geometry computation canceled by reset'));
		this.worker.terminate();
		try {
			this.worker = this.createWorker();
		} catch (error) {
			this.closed = true;
			throw error;
		}
	}
	destroy(): void {
		if (this.closed) return;
		this.generation++;
		this.shutdown(new Error('Story geometry engine disposed'));
	}
	dispose(): void {
		this.destroy();
	}
}
