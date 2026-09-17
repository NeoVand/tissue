/// <reference lib="webworker" />
import {
	buildStoryGeometry,
	prepareStoryGeometryIndex,
	storyNearestNeighbors,
	type StoryGeometryIndex,
	type StoryGeometryResult
} from './geometry';
import type { StoryGeometryRequest, StoryGeometryResponse } from './geometry-engine';

let previous: StoryGeometryResult | undefined;
let index: StoryGeometryIndex | undefined;

function dispatch(request: StoryGeometryRequest) {
	const post = (response: StoryGeometryResponse) => self.postMessage(response);
	try {
		let result: StoryGeometryResult | ReturnType<typeof storyNearestNeighbors> | null = null;
		if (request.op === 'build') {
			const fit = buildStoryGeometry(request.atlas, { previous });
			previous = fit.geometry;
			index = fit.index;
			result = fit.geometry;
		} else if (request.op === 'setAtlas') {
			index = prepareStoryGeometryIndex(request.atlas).index;
			previous = undefined;
		} else {
			if (!index) throw new Error('Measure or load an atlas before inspecting its neighbors');
			result = storyNearestNeighbors(index, request.unit, request.k, request.layer);
		}
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

// Keep selection queries behind their atlas load, even if implementation becomes asynchronous.
let queue = Promise.resolve();
self.onmessage = (event: MessageEvent<StoryGeometryRequest>) => {
	queue = queue.then(() => dispatch(event.data));
};
