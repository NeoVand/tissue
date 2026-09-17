import { MODEL_CONFIG as cfg, NEURON_COUNT, type Checkpoint, type Probe } from './protocol';

export interface ParameterDescriptor {
	leafIndex: number;
	name: string;
	shape: readonly [number, number];
	role: 'embedding' | 'readout' | 'attention' | 'mlp';
	layer?: number;
}
/** Checkpoint v1 preserves Params object insertion order, then layer-array order.
 * This mapping is checked against the real JaxJS serializer in model-inspection.spec.ts. */
export const PARAMETER_DESCRIPTORS: readonly ParameterDescriptor[] = [
	{ leafIndex: 0, name: 'wte', shape: [cfg.vocabulary.length, cfg.width], role: 'embedding' },
	{ leafIndex: 1, name: 'wpe', shape: [cfg.sequenceLength, cfg.width], role: 'embedding' },
	{ leafIndex: 2, name: 'lmHead', shape: [cfg.width, cfg.vocabulary.length], role: 'readout' },
	...Array.from({ length: cfg.layers }, (_, layer) =>
		(['wq', 'wk', 'wv', 'wo', 'mlpFc1', 'mlpFc2'] as const).map((name, offset) => ({
			leafIndex: 3 + layer * 6 + offset,
			name: `layers[${layer}].${name}`,
			shape: (name === 'mlpFc1'
				? [cfg.width, cfg.hidden]
				: name === 'mlpFc2'
					? [cfg.hidden, cfg.width]
					: [cfg.width, cfg.width]) as [number, number],
			role: (name.startsWith('mlp') ? 'mlp' : 'attention') as ParameterDescriptor['role'],
			layer
		}))
	).flat()
];
export const MODEL_PARAMETER_COUNT = PARAMETER_DESCRIPTORS.reduce(
	(sum, item) => sum + item.shape[0] * item.shape[1],
	0
);

export interface VectorStatistics {
	min: number;
	max: number;
	mean: number;
	l1: number;
	l2: number;
	rms: number;
	nonzero: number;
}
export function vectorStatistics(values: readonly number[]): VectorStatistics {
	if (!values.length || values.some((value) => !Number.isFinite(value)))
		throw new Error('A weight or activation profile must contain finite values');
	let sum = 0,
		squares = 0,
		l1 = 0,
		min = Infinity,
		max = -Infinity,
		nonzero = 0;
	for (const value of values) {
		sum += value;
		squares += value * value;
		l1 += Math.abs(value);
		min = Math.min(min, value);
		max = Math.max(max, value);
		if (value !== 0) nonzero++;
	}
	return {
		min,
		max,
		mean: sum / values.length,
		l1,
		l2: Math.sqrt(squares),
		rms: Math.sqrt(squares / values.length),
		nonzero
	};
}
export function neuronAddress(id: number) {
	if (!Number.isInteger(id) || id < 0 || id >= NEURON_COUNT)
		throw new Error('Neuron index is outside the model');
	return { id, layer: Math.floor(id / cfg.hidden), channel: id % cfg.hidden };
}
export interface WeightProfile {
	descriptor: ParameterDescriptor;
	selection: string;
	values: number[];
	statistics: VectorStatistics;
}
export interface NeuronInspection {
	id: number;
	layer: number;
	channel: number;
	step: number;
	incoming: WeightProfile;
	outgoing: WeightProfile;
}
export function inspectNeuron(checkpoint: Checkpoint, id: number): NeuronInspection {
	const address = neuronAddress(id);
	if (checkpoint.version !== 1 || checkpoint.architecture !== 'binding-transformer-v1')
		throw new Error('Unsupported checkpoint architecture');
	if (checkpoint.parameters.length !== PARAMETER_DESCRIPTORS.length)
		throw new Error('Checkpoint is missing parameter tensors');
	const profile = (offset: number, incoming: boolean): WeightProfile => {
		const descriptor = PARAMETER_DESCRIPTORS[3 + address.layer * 6 + offset];
		const leaf = checkpoint.parameters[descriptor.leafIndex];
		if (
			!leaf ||
			leaf.shape.join(',') !== descriptor.shape.join(',') ||
			leaf.values.length !== descriptor.shape[0] * descriptor.shape[1]
		)
			throw new Error(`Invalid tensor shape for ${descriptor.name}`);
		const values = Array.from(
			{ length: cfg.width },
			(_, dimension) =>
				leaf.values[
					incoming
						? dimension * cfg.hidden + address.channel
						: address.channel * cfg.width + dimension
				]
		);
		return {
			descriptor,
			selection: incoming ? `[:, ${address.channel}]` : `[${address.channel}, :]`,
			values,
			statistics: vectorStatistics(values)
		};
	};
	return {
		...address,
		step: checkpoint.step,
		incoming: profile(4, true),
		outgoing: profile(5, false)
	};
}
export interface ActivationProfile {
	step: number;
	values: number[];
	tokens: string[];
	statistics: VectorStatistics;
	lesionNeuron: number | null;
}
export function inspectActivations(probe: Probe, id: number): ActivationProfile {
	neuronAddress(id);
	if (
		probe.activations.length !== cfg.sequenceLength ||
		probe.example.tokens.length !== cfg.sequenceLength
	)
		throw new Error('Probe sequence does not match the model');
	const values = probe.activations.map((row) => row[id]);
	const statistics = vectorStatistics(values);
	if (statistics.min < 0) throw new Error('Expected nonnegative post-ReLU activations');
	return {
		step: probe.step,
		values,
		tokens: [...probe.example.tokens],
		statistics,
		lesionNeuron: probe.lesionNeuron
	};
}
