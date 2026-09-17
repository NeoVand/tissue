/// <reference lib="webworker" />
import { TokenStoryRuntime } from './runtime';
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
const runtime = new TokenStoryRuntime();
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
	}
}
let queue = Promise.resolve();
self.onmessage = (event: MessageEvent<Request>) => {
	const request = event.data;
	if (request.op === 'pause') {
		runtime.pause();
		self.postMessage({ id: request.id, ok: true, result: null });
		return;
	}
	if (request.op === 'dispose' || request.op === 'initialize' || request.op === 'load')
		runtime.pause();
	queue = queue.then(() => dispatch(request));
};
