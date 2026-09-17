import type {
	Atlas,
	Backend,
	Checkpoint,
	EngineEvent,
	Initialization,
	Metrics,
	Probe,
	RepairOptions,
	RepairResult,
	RpcResponse
} from './protocol';
import type { QueryMeasurement } from './query-protocol';
interface Pending {
	resolve: (value: unknown) => void;
	reject: (error: Error) => void;
	onMetrics?: (metrics: Metrics) => void;
}
/** Browser-only client. Construct during mount; dispose during teardown/reset. */
export class Engine {
	private worker: Worker;
	private pending = new Map<number, Pending>();
	private nextId = 1;
	private closed = false;
	private disposal: Promise<void> | null = null;
	constructor(private onEvent?: (event: EngineEvent) => void) {
		this.worker = new Worker(new URL('./model/worker.ts', import.meta.url), { type: 'module' });
		this.worker.onmessage = (event: MessageEvent<RpcResponse>) => {
			const message = event.data;
			const pending = this.pending.get(message.id);
			if (!pending) return;
			if (message.event) {
				this.onEvent?.(message.event);
				if (message.event.type === 'metrics') pending.onMetrics?.(message.event.metrics);
				return;
			}
			this.pending.delete(message.id);
			if (message.ok) pending.resolve(message.result);
			else pending.reject(new Error(message.error ?? 'Model operation failed'));
		};
		this.worker.onerror = (event) =>
			this.shutdown(new Error(event.message || 'Model worker failed'));
		this.worker.onmessageerror = () => this.shutdown(new Error('Unable to decode model response'));
	}
	private shutdown(error: Error) {
		this.closed = true;
		for (const pending of this.pending.values()) pending.reject(error);
		this.pending.clear();
		this.worker.terminate();
	}
	private call<T>(
		op: string,
		payload: Record<string, unknown> = {},
		onMetrics?: (metrics: Metrics) => void
	): Promise<T> {
		if (this.closed) return Promise.reject(new Error('Model engine is disposed'));
		const id = this.nextId++;
		return new Promise((resolve, reject) => {
			this.pending.set(id, { resolve: resolve as (value: unknown) => void, reject, onMetrics });
			try {
				this.worker.postMessage({ id, op, ...payload });
			} catch (error) {
				this.pending.delete(id);
				reject(error);
			}
		});
	}
	async initialize(seed = 42, backend: Backend | 'auto' = 'auto'): Promise<Initialization> {
		let timer: ReturnType<typeof setTimeout> | undefined;
		try {
			return await Promise.race([
				this.call<Initialization>('initialize', { seed, backend }),
				new Promise<never>((_, reject) => {
					timer = setTimeout(() => {
						const error = new Error(
							'Model initialization timed out. Retry, or choose the WASM backend.'
						);
						this.shutdown(error);
						reject(error);
					}, 60000);
				})
			]);
		} finally {
			clearTimeout(timer);
		}
	}
	train(steps: number, onMetrics?: (metrics: Metrics) => void): Promise<Metrics> {
		return this.call('train', { steps }, onMetrics);
	}
	pause(): Promise<void> {
		return this.call('pause');
	}
	probe(exampleIndex = 0, lesionNeuron?: number | null): Promise<Probe> {
		return this.call('probe', { exampleIndex, lesionNeuron });
	}
	captureAtlas(includeEffects = false): Promise<Atlas> {
		return this.call('atlas', { includeEffects });
	}
	measureQueryShifts(): Promise<QueryMeasurement> {
		return this.call('query-shifts');
	}
	controlledRepair(options: RepairOptions): Promise<RepairResult> {
		return this.call('repair', { options });
	}
	evaluate(): Promise<Metrics> {
		return this.call('evaluate');
	}
	exportCheckpoint(): Promise<Checkpoint> {
		return this.call('export');
	}
	loadCheckpoint(checkpoint: Checkpoint): Promise<Metrics> {
		return this.call('load', { checkpoint });
	}
	dispose(): Promise<void> {
		if (this.disposal) return this.disposal;
		if (this.closed) return Promise.resolve();
		const disposeId = this.nextId;
		const reply = this.call('dispose');
		this.closed = true;
		for (const [id, pending] of this.pending)
			if (id !== disposeId) {
				pending.reject(new Error('Model engine disposed'));
				this.pending.delete(id);
			}
		this.disposal = (async () => {
			let timer: ReturnType<typeof setTimeout> | undefined;
			try {
				await Promise.race([
					reply,
					new Promise<void>((resolve) => {
						timer = setTimeout(resolve, 500);
					})
				]);
			} catch {
				/* Worker termination below is unconditional. */
			} finally {
				clearTimeout(timer);
				this.shutdown(new Error('Model engine disposed'));
			}
		})();
		return this.disposal;
	}
}
