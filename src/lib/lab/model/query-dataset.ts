import type { QueryGroup } from '../query-protocol';
import { assignmentSplits, makeExample, RandomStream } from './dataset';

export const QUERY_GROUP_COUNT = 16;
export const QUERIES_PER_GROUP = 3;
/** Eight complete triples per batch: no padding, and every numerical batch has shape 24. */
export const QUERY_BATCH_SIZE = 24;

/** No model seed, checkpoint, training stream, or measured outcome enters this design. */
export function pairedQueryGroups(split: 'calibration' | 'test'): QueryGroup[] {
	const random = new RandomStream(split === 'calibration' ? 0x70616972 : 0x71756572);
	const assignments = assignmentSplits[split].map((assignment) => [...assignment]);
	for (let i = assignments.length - 1; i > 0; i--) {
		const j = random.integer(i + 1);
		[assignments[i], assignments[j]] = [assignments[j], assignments[i]];
	}
	return assignments.slice(0, QUERY_GROUP_COUNT).map((assignment) => {
		const order = [0, 1, 2];
		for (let i = order.length - 1; i > 0; i--) {
			const j = random.integer(i + 1);
			[order[i], order[j]] = [order[j], order[i]];
		}
		return {
			id: `${split}:${assignment.join('')}:${order.join('')}`,
			assignment,
			order,
			examples: [0, 1, 2].map((query) => makeExample(assignment, query, order, split))
		};
	});
}
export const QUERY_GROUPS = {
	calibration: pairedQueryGroups('calibration'),
	test: pairedQueryGroups('test')
};
