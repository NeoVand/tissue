/// <reference lib="webworker" />
import { analyzeQueryShifts } from './query-analysis';
import type { QueryMeasurement } from './query-protocol';

const worker = self as DedicatedWorkerGlobalScope;
worker.onmessage = (event: MessageEvent<{ id: number; measurement: QueryMeasurement }>) => {
	const { id, measurement } = event.data;
	try {
		worker.postMessage({ id, result: analyzeQueryShifts(measurement) });
	} catch (error) {
		worker.postMessage({ id, error: error instanceof Error ? error.message : String(error) });
	}
};
