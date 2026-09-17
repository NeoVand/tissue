import type {
	StoryAtlas,
	StoryBackend,
	StoryCheckpoint,
	StoryEvent,
	StoryGeneration,
	StoryGenerationOptions,
	StoryInitialization,
	StoryMetrics,
	StoryPreset,
	StoryProbe
} from './protocol';
interface Pending {
	resolve: (value: unknown) => void;
	reject: (error: Error) => void;
	timer: ReturnType<typeof setTimeout>;
	onMetrics?: (metrics: StoryMetrics) => void;
}
/** Browser-only worker owner. Every operation has a deadline; timeout ends the worker. */
export class StoryEngine {
	private worker: Worker;
	private pending = new Map<number, Pending>();
	private nextId = 1;
	private closed = false;
	private disposal: Promise<void> | null = null;
	constructor(private onEvent?: (event: StoryEvent) => void) {
		this.worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
		this.worker.onmessage = (
			event: MessageEvent<{
				id: number;
				event?: StoryEvent;
				ok?: boolean;
				result?: unknown;
				error?: string;
			}>
		) => {
			const message = event.data,
				pending = this.pending.get(message.id);
			if (!pending) return;
			if (message.event) {
				this.onEvent?.(message.event);
				if (message.event.type === 'metrics') pending.onMetrics?.(message.event.metrics);
				return;
			}
			clearTimeout(pending.timer);
			this.pending.delete(message.id);
			if (message.ok) pending.resolve(message.result);
			else pending.reject(new Error(message.error ?? 'Story operation failed'));
		};
		this.worker.onerror = (event) =>
			this.shutdown(new Error(event.message || 'Story worker failed'));
		this.worker.onmessageerror = () => this.shutdown(new Error('Unable to decode story response'));
	}
	private shutdown(error: Error) {
		this.closed = true;
		for (const p of this.pending.values()) {
			clearTimeout(p.timer);
			p.reject(error);
		}
		this.pending.clear();
		this.worker.terminate();
	}
	private call<T>(
		op: string,
		payload: Record<string, unknown> = {},
		onMetrics?: (metrics: StoryMetrics) => void,
		timeoutMs = 180000
	): Promise<T> {
		if (this.closed) return Promise.reject(new Error('Story engine is disposed'));
		const id = this.nextId++;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(
				() =>
					this.shutdown(
						new Error(
							`Story ${op} timed out after ${timeoutMs / 1000}s. The worker was stopped; restore the last saved checkpoint.`
						)
					),
				timeoutMs
			);
			this.pending.set(id, {
				resolve: resolve as (value: unknown) => void,
				reject,
				timer,
				onMetrics
			});
			try {
				this.worker.postMessage({ id, op, ...payload });
			} catch (error) {
				clearTimeout(timer);
				this.pending.delete(id);
				reject(error);
			}
		});
	}
	initialize(
		preset: StoryPreset,
		seed = 42,
		backend: StoryBackend | 'auto' = 'auto'
	): Promise<StoryInitialization> {
		return this.call('initialize', { preset, seed, backend });
	}
	train(steps: number, onMetrics?: (metrics: StoryMetrics) => void): Promise<StoryMetrics> {
		return this.call('train', { steps }, onMetrics, 300000);
	}
	pause(): Promise<void> {
		return this.call('pause', {}, undefined, 30000);
	}
	captureAtlas(): Promise<StoryAtlas> {
		return this.call('atlas');
	}
	probe(prompt: string, lesionNeuron?: number | null): Promise<StoryProbe> {
		return this.call('probe', { prompt, lesionNeuron });
	}
	generate(prompt: string, options: StoryGenerationOptions = {}): Promise<StoryGeneration> {
		return this.call('generate', { prompt, options });
	}
	evaluate(): Promise<StoryMetrics> {
		return this.call('evaluate');
	}
	exportCheckpoint(): Promise<StoryCheckpoint> {
		return this.call('export');
	}
	/** Structured clone intentionally preserves every caller-owned typed checkpoint buffer. */
	loadCheckpoint(
		checkpoint: StoryCheckpoint,
		backend: StoryBackend | 'auto' = 'auto'
	): Promise<StoryInitialization> {
		return this.call('load', { checkpoint, backend });
	}
	dispose(): Promise<void> {
		if (this.disposal) return this.disposal;
		if (this.closed) return Promise.resolve();
		const disposeId = this.nextId,
			reply = this.call('dispose');
		this.closed = true;
		for (const [id, p] of this.pending)
			if (id !== disposeId) {
				clearTimeout(p.timer);
				p.reject(new Error('Story engine disposed'));
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
				/* Unconditional termination below. */
			} finally {
				clearTimeout(timer);
				this.shutdown(new Error('Story engine disposed'));
			}
		})();
		return this.disposal;
	}
}
