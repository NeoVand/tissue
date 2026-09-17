/// <reference lib="webworker" />
import { ModelRuntime } from './runtime';
import type {
	Backend,
	Checkpoint,
	EngineEvent,
	RepairOptions,
	RpcRequest,
	RpcResponse
} from '../protocol';
const runtime = new ModelRuntime();
const post = (message: RpcResponse) => self.postMessage(message);
async function dispatch(request: RpcRequest) {
	const emit = (event: EngineEvent) => post({ id: request.id, event });
	try {
		let result: unknown;
		switch (request.op) {
			case 'initialize':
				result = await runtime.initialize(
					request.seed as number,
					request.backend as Backend | 'auto',
					emit
				);
				break;
			case 'train':
				result = await runtime.train(request.steps as number, emit);
				break;
			case 'probe':
				result = await runtime.probe(
					request.exampleIndex as number,
					request.lesionNeuron as number | undefined
				);
				break;
			case 'atlas':
				result = await runtime.captureAtlas(request.includeEffects === true, emit);
				break;
			case 'query-shifts':
				result = await runtime.measureQueryShifts(emit);
				break;
			case 'repair':
				result = await runtime.controlledRepair(request.options as RepairOptions, emit);
				break;
			case 'export':
				result = await runtime.exportCheckpoint();
				break;
			case 'load':
				result = await runtime.loadCheckpoint(request.checkpoint as Checkpoint);
				break;
			case 'evaluate':
				result = await runtime.evaluate();
				break;
			case 'dispose':
				runtime.dispose();
				result = null;
				break;
			default:
				throw new Error(`Unknown model operation: ${request.op}`);
		}
		post({ id: request.id, ok: true, result });
	} catch (error) {
		post({
			id: request.id,
			ok: false,
			error: error instanceof Error ? `${error.name}: ${error.message}` : String(error)
		});
	}
}
let queue = Promise.resolve();
self.onmessage = (event: MessageEvent<RpcRequest>) => {
	const request = event.data;
	if (request.op === 'pause') {
		runtime.pause();
		post({ id: request.id, ok: true, result: null });
		return;
	}
	if (request.op === 'dispose' || request.op === 'initialize') runtime.pause();
	queue = queue.then(() => dispatch(request));
};
