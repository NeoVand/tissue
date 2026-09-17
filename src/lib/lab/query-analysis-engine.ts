import type { QueryAnalysis } from './query-analysis';
import type { QueryMeasurement } from './query-protocol';

/** Keep CPU geometry fitting and neighborhood analysis off the rendering thread. */
export class QueryAnalysisEngine {
	private worker: Worker;
	private nextId = 0;
	private disposed = false;
	private pending = new Map<
		number,
		{ resolve: (result: QueryAnalysis) => void; reject: (error: Error) => void }
	>();

	constructor() {
		this.worker = new Worker(new URL('./query-analysis-worker.ts', import.meta.url), {
			type: 'module'
		});
		this.worker.onmessage = (
			event: MessageEvent<{ id: number; result?: QueryAnalysis; error?: string }>
		) => {
			const response = event.data;
			const request = this.pending.get(response.id);
			if (!request) return;
			this.pending.delete(response.id);
			if (response.result) request.resolve(response.result);
			else request.reject(new Error(response.error ?? 'Query analysis returned no result'));
		};
		this.worker.onerror = (event) => {
			this.rejectAll(new Error(event.message || 'Query analysis worker failed'));
			this.dispose();
		};
		this.worker.onmessageerror = () => {
			this.rejectAll(new Error('Could not read the query analysis worker response'));
			this.dispose();
		};
	}

	analyze(measurement: QueryMeasurement): Promise<QueryAnalysis> {
		if (this.disposed) return Promise.reject(new Error('Query analysis engine is disposed'));
		return new Promise((resolve, reject) => {
			const id = this.nextId++;
			this.pending.set(id, { resolve, reject });
			try {
				this.worker.postMessage({ id, measurement });
			} catch (error) {
				this.pending.delete(id);
				reject(error instanceof Error ? error : new Error(String(error)));
			}
		});
	}

	private rejectAll(error: Error) {
		for (const request of this.pending.values()) request.reject(error);
		this.pending.clear();
	}

	dispose(): void {
		if (this.disposed) return;
		this.disposed = true;
		this.worker.terminate();
		this.rejectAll(new Error('Query analysis was cancelled'));
	}
}
