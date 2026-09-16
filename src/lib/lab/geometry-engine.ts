import type { GeometryKind, GeometryResult } from './geometry';

export type GeometryRequest =
	| {
			id: number;
			generation: number;
			op: 'build';
			fingerprints: number[][];
			kind: GeometryKind;
			reference?: { positions: GeometryResult['positions']; valid: boolean[] };
	  }
	| { id: number; generation: number; op: 'reset' };

export type GeometryResponse =
	| { id: number; generation: number; ok: true; result: GeometryResult | null }
	| { id: number; generation: number; ok: false; error: string };

interface Pending {
	generation: number;
	resolve: (result: GeometryResult | null) => void;
	reject: (error: Error) => void;
}

/** Browser-only numeric worker. Construct during mount and dispose during teardown. */
export class GeometryEngine {
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
		worker.onmessage = (event: MessageEvent<GeometryResponse>) => {
			const response = event.data;
			if (
				this.closed ||
				generation !== this.generation ||
				response.generation !== this.generation
			) {
				return;
			}
			const pending = this.pending.get(response.id);
			if (!pending || pending.generation !== response.generation) return;
			this.pending.delete(response.id);
			if (response.ok) pending.resolve(response.result);
			else pending.reject(new Error(response.error));
		};
		worker.onerror = (event) => {
			if (generation === this.generation && !this.closed) {
				this.shutdown(new Error(event.message || 'Geometry worker failed'));
			}
		};
		worker.onmessageerror = () => {
			if (generation === this.generation && !this.closed) {
				this.shutdown(new Error('Unable to decode geometry worker response'));
			}
		};
		return worker;
	}

	private rejectPending(error: Error): void {
		for (const pending of this.pending.values()) pending.reject(error);
		this.pending.clear();
	}

	private shutdown(error: Error): void {
		this.closed = true;
		this.rejectPending(error);
		this.worker.terminate();
	}

	private call(
		operation:
			| {
					op: 'build';
					fingerprints: number[][];
					kind: GeometryKind;
					reference?: { positions: GeometryResult['positions']; valid: boolean[] };
			  }
			| { op: 'reset' }
	): Promise<GeometryResult | null> {
		if (this.closed) return Promise.reject(new Error('Geometry engine is disposed'));
		const id = this.nextId++;
		return new Promise((resolve, reject) => {
			this.pending.set(id, { generation: this.generation, resolve, reject });
			try {
				const request: GeometryRequest = { id, generation: this.generation, ...operation };
				this.worker.postMessage(request);
			} catch (error) {
				this.pending.delete(id);
				reject(error instanceof Error ? error : new Error(String(error)));
			}
		});
	}

	async build(
		fingerprints: number[][],
		kind: GeometryKind,
		reference?: { positions: GeometryResult['positions']; valid: boolean[] }
	): Promise<GeometryResult> {
		const result = await this.call({ op: 'build', fingerprints, kind, reference });
		if (!result) throw new Error('Geometry worker returned an empty map');
		return result;
	}

	/** Start a new model/probe identity sequence; old frames must not influence alignment. */
	async reset(): Promise<void> {
		if (this.closed) throw new Error('Geometry engine is disposed');
		this.generation++;
		this.rejectPending(new Error('Geometry computation canceled by reset'));
		// PCA is synchronous within its worker. Termination preempts an old heavy fit,
		// rather than making a new run wait behind stale queued measurements.
		this.worker.terminate();
		try {
			this.worker = this.createWorker();
		} catch (error) {
			this.closed = true;
			throw error;
		}
		await this.call({ op: 'reset' });
	}

	dispose(): void {
		if (this.closed) return;
		this.generation++;
		this.shutdown(new Error('Geometry engine disposed'));
	}
}
