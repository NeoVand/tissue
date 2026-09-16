/// <reference lib="webworker" />
import { buildGeometry, type GeometryKind, type GeometryResult } from './geometry';
import type { GeometryRequest, GeometryResponse } from './geometry-engine';

const previous = new Map<GeometryKind, GeometryResult>();
let generation: number | null = null;

function dispatch(request: GeometryRequest): void {
	const post = (response: GeometryResponse) => self.postMessage(response);
	try {
		if (generation !== request.generation || request.op === 'reset') {
			previous.clear();
			generation = request.generation;
		}
		if (request.op === 'reset') {
			post({ id: request.id, generation: request.generation, ok: true, result: null });
			return;
		}
		const result = buildGeometry(request.fingerprints, {
			kind: request.kind,
			previous: request.reference
				? { ...request.reference, kind: request.kind }
				: previous.get(request.kind)
		});
		previous.set(request.kind, result);
		post({ id: request.id, generation: request.generation, ok: true, result });
	} catch (error) {
		post({
			id: request.id,
			generation: request.generation,
			ok: false,
			error: error instanceof Error ? error.message : String(error)
		});
	}
}

// Explicit serialization preserves alignment order even if dispatch later becomes asynchronous.
let queue = Promise.resolve();
self.onmessage = (event: MessageEvent<GeometryRequest>) => {
	queue = queue.then(() => dispatch(event.data));
};
