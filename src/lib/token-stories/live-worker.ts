/// <reference lib="webworker" />
import { LiveTokenStoryRuntime } from './live-runtime';
import type {
	TokenStoryBackend,
	TokenStoryCheckpoint,
	TokenStoryEvent,
	TokenStoryGenerationOptions,
	TokenStoryPreset
} from './protocol';

interface Request {
	id: number;
	op: string;
	[key: string]: unknown;
}
const runtime = new LiveTokenStoryRuntime();
const liveRequests = new Set<number>();
const cancelledLiveRequests = new Set<number>();
let waitingFrame: {
	id: number;
	index: number;
	resolve: () => void;
	reject: (error: Error) => void;
} | null = null;
function cancelFrame() {
	for (const id of liveRequests) cancelledLiveRequests.add(id);
	runtime.pause();
	const waiting = waitingFrame;
	waitingFrame = null;
	waiting?.resolve();
}
function buffers(value: unknown, found = new Set<ArrayBuffer>()): ArrayBuffer[] {
	if (ArrayBuffer.isView(value) && value.buffer instanceof ArrayBuffer) found.add(value.buffer);
	else if (Array.isArray(value)) for (const child of value) buffers(child, found);
	else if (value && typeof value === 'object')
		for (const child of Object.values(value)) buffers(child, found);
	return [...found];
}
async function dispatch(request: Request) {
	const emit = (event: TokenStoryEvent) => self.postMessage({ id: request.id, event });
	try {
		let result: unknown;
		switch (request.op) {
			case 'initialize':
				result = await runtime.initialize(
					request.preset as TokenStoryPreset,
					request.seed as number,
					request.backend as TokenStoryBackend | 'auto',
					undefined,
					emit
				);
				break;
			case 'train':
				result = await runtime.train(request.steps as number, emit);
				break;
			case 'atlas':
				result = await runtime.captureAtlas(emit);
				break;
			case 'probe':
				result = await runtime.probe(
					request.prompt as string,
					request.lesionNeuron as number | null
				);
				break;
			case 'generate':
				result = await runtime.generate(
					request.prompt as string,
					request.options as TokenStoryGenerationOptions
				);
				break;
			case 'generate-live':
				result = await runtime.generateLive(
					request.prompt as string,
					request.options as TokenStoryGenerationOptions,
					(frame) =>
						new Promise<void>((resolve, reject) => {
							waitingFrame = { id: request.id, index: frame.index, resolve, reject };
							self.postMessage({ id: request.id, frame }, buffers(frame));
						}),
					cancelledLiveRequests.has(request.id)
				);
				break;
			case 'evaluate':
				result = await runtime.evaluate();
				break;
			case 'export':
				result = await runtime.exportCheckpoint();
				break;
			case 'load':
				result = await runtime.loadCheckpoint(
					request.checkpoint as TokenStoryCheckpoint,
					request.backend as TokenStoryBackend | 'auto',
					emit
				);
				break;
			case 'dispose':
				runtime.dispose();
				result = null;
				break;
			default:
				throw new Error(`Unknown tokenStory operation: ${request.op}`);
		}
		self.postMessage({ id: request.id, ok: true, result }, buffers(result));
	} catch (error) {
		self.postMessage({
			id: request.id,
			ok: false,
			error: error instanceof Error ? `${error.name}: ${error.message}` : String(error)
		});
	} finally {
		liveRequests.delete(request.id);
		cancelledLiveRequests.delete(request.id);
		if (waitingFrame?.id === request.id) {
			const waiting = waitingFrame;
			waitingFrame = null;
			waiting.resolve();
		}
	}
}
let queue = Promise.resolve();
self.onmessage = (event: MessageEvent<Request>) => {
	const request = event.data;
	if (request.op === 'generate-live') liveRequests.add(request.id);
	if (request.op === 'live-ack' || request.op === 'live-error') {
		if (waitingFrame?.id === request.id && waitingFrame.index === request.index) {
			const waiting = waitingFrame;
			waitingFrame = null;
			if (request.op === 'live-error') waiting.reject(new Error(String(request.error)));
			else waiting.resolve();
		}
		return;
	}
	if (request.op === 'pause') {
		cancelFrame();
		self.postMessage({ id: request.id, ok: true, result: null });
		return;
	}
	if (request.op === 'dispose' || request.op === 'initialize' || request.op === 'load')
		cancelFrame();
	queue = queue.then(() => dispatch(request));
};
