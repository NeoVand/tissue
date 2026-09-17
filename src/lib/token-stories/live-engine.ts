import type {
	TokenStoryAtlas,
	TokenStoryBackend,
	TokenStoryCheckpoint,
	TokenStoryEvent,
	TokenStoryGeneration,
	TokenStoryGenerationOptions,
	TokenStoryInitialization,
	TokenStoryMetrics,
	TokenStoryPreset,
	TokenStoryProbe
} from './protocol';
import type { LiveTokenStoryFrame, LiveTokenStoryFrameCallback } from './live-protocol';
interface Pending {
	resolve: (value: unknown) => void;
	reject: (error: Error) => void;
	timer?: ReturnType<typeof setTimeout>;
	timeoutMs: number;
	onMetrics?: (metrics: TokenStoryMetrics) => void;
	onFrame?: LiveTokenStoryFrameCallback;
	nextFrame: number;
	waitingIndex?: number;
	cancelled: boolean;
}
/** Compatible engine with frame ACKs. Computation has a deadline; user-controlled animation dwell does not. */
export class LiveTokenStoryEngine {
	private worker: Worker;
	private pending = new Map<number, Pending>();
	private nextId = 1;
	private closed = false;
	private disposal: Promise<void> | null = null;
	constructor(private onEvent?: (event: TokenStoryEvent) => void) {
		this.worker = new Worker(new URL('./live-worker.ts', import.meta.url), { type: 'module' });
		this.worker.onmessage = (
			event: MessageEvent<{
				id: number;
				event?: TokenStoryEvent;
				frame?: LiveTokenStoryFrame;
				ok?: boolean;
				result?: unknown;
				error?: string;
			}>
		) => {
			const message = event.data,
				pending = this.pending.get(message.id);
			if (!pending) return;
			if (message.frame) {
				this.receiveFrame(message.id, pending, message.frame);
				return;
			}
			if (message.event) {
				this.onEvent?.(message.event);
				if (message.event.type === 'metrics') pending.onMetrics?.(message.event.metrics);
				return;
			}
			clearTimeout(pending.timer);
			this.pending.delete(message.id);
			if (message.ok) pending.resolve(message.result);
			else pending.reject(new Error(message.error ?? 'TokenStory operation failed'));
		};
		this.worker.onerror = (event) =>
			this.shutdown(new Error(event.message || 'TokenStory worker failed'));
		this.worker.onmessageerror = () =>
			this.shutdown(new Error('Unable to decode tokenStory response'));
	}
	private arm(id: number, pending: Pending) {
		clearTimeout(pending.timer);
		pending.timer = setTimeout(
			() =>
				this.shutdown(
					new Error(
						`Live TokenStory operation ${id} timed out after ${pending.timeoutMs / 1000}s of waiting for the worker. Restore the last saved checkpoint.`
					)
				),
			pending.timeoutMs
		);
	}
	private receiveFrame(id: number, pending: Pending, frame: LiveTokenStoryFrame) {
		if (pending.cancelled) return;
		if (
			!pending.onFrame ||
			frame.index !== pending.nextFrame ||
			pending.waitingIndex !== undefined
		) {
			this.shutdown(new Error('Invalid live token frame sequence'));
			return;
		}
		clearTimeout(pending.timer);
		pending.timer = undefined;
		pending.waitingIndex = frame.index;
		pending.nextFrame++;
		Promise.resolve()
			.then(() => {
				if (!pending.cancelled && this.pending.get(id) === pending) return pending.onFrame!(frame);
			})
			.then(
				() => {
					if (pending.cancelled || this.pending.get(id) !== pending) return;
					pending.waitingIndex = undefined;
					this.arm(id, pending);
					this.worker.postMessage({ id, op: 'live-ack', index: frame.index });
				},
				(error) => {
					if (pending.cancelled || this.pending.get(id) !== pending) return;
					pending.waitingIndex = undefined;
					this.arm(id, pending);
					this.worker.postMessage({
						id,
						op: 'live-error',
						index: frame.index,
						error: error instanceof Error ? error.message : String(error)
					});
				}
			);
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
		onMetrics?: (metrics: TokenStoryMetrics) => void,
		timeoutMs = 180000,
		onFrame?: LiveTokenStoryFrameCallback
	): Promise<T> {
		if (this.closed) return Promise.reject(new Error('TokenStory engine is disposed'));
		const id = this.nextId++;
		return new Promise((resolve, reject) => {
			const pending: Pending = {
				resolve: resolve as (value: unknown) => void,
				reject,
				timeoutMs,
				onMetrics,
				onFrame,
				nextFrame: 0,
				cancelled: false
			};
			this.pending.set(id, pending);
			this.arm(id, pending);
			try {
				this.worker.postMessage({ id, op, ...payload });
			} catch (error) {
				clearTimeout(pending.timer);
				this.pending.delete(id);
				reject(error);
			}
		});
	}
	initialize(
		preset: TokenStoryPreset,
		seed = 42,
		backend: TokenStoryBackend | 'auto' = 'auto'
	): Promise<TokenStoryInitialization> {
		return this.call('initialize', { preset, seed, backend });
	}
	train(
		steps: number,
		onMetrics?: (metrics: TokenStoryMetrics) => void
	): Promise<TokenStoryMetrics> {
		return this.call('train', { steps }, onMetrics, 300000);
	}
	pause(): Promise<void> {
		for (const [id, pending] of this.pending)
			if (pending.onFrame) {
				pending.cancelled = true;
				this.arm(id, pending);
			}
		return this.call('pause', {}, undefined, 30000);
	}
	captureAtlas(): Promise<TokenStoryAtlas> {
		return this.call('atlas');
	}
	probe(prompt: string, lesionNeuron?: number | null): Promise<TokenStoryProbe> {
		return this.call('probe', { prompt, lesionNeuron });
	}
	generate(
		prompt: string,
		options: TokenStoryGenerationOptions = {}
	): Promise<TokenStoryGeneration> {
		return this.call('generate', { prompt, options });
	}
	/** Resolve onFrame only after playback; pause cancels a pending frame without committing its sample. */
	generateLive(
		prompt: string,
		options: TokenStoryGenerationOptions,
		onFrame: LiveTokenStoryFrameCallback
	): Promise<TokenStoryGeneration> {
		if (typeof onFrame !== 'function')
			return Promise.reject(new Error('A live frame callback is required'));
		return this.call('generate-live', { prompt, options }, undefined, 180000, onFrame);
	}
	evaluate(): Promise<TokenStoryMetrics> {
		return this.call('evaluate');
	}
	exportCheckpoint(): Promise<TokenStoryCheckpoint> {
		return this.call('export');
	}
	/** Structured clone intentionally preserves every caller-owned typed checkpoint buffer. */
	loadCheckpoint(
		checkpoint: TokenStoryCheckpoint,
		backend: TokenStoryBackend | 'auto' = 'auto'
	): Promise<TokenStoryInitialization> {
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
				p.reject(new Error('TokenStory engine disposed'));
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
				this.shutdown(new Error('TokenStory engine disposed'));
			}
		})();
		return this.disposal;
	}
}
