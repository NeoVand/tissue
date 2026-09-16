import { MODEL_CONFIG, type BindingExample } from '../protocol';

/** Serializable mulberry32: probe and evaluation code never receive the training stream. */
export class RandomStream {
	state: number;
	constructor(seed: number) {
		this.state = seed >>> 0;
	}
	next() {
		this.state = (this.state + 0x6d2b79f5) >>> 0;
		let t = this.state;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	}
	integer(n: number) {
		return Math.floor(this.next() * n);
	}
}
function shuffle<T>(array: T[], random: RandomStream) {
	for (let i = array.length - 1; i > 0; i--) {
		const j = random.integer(i + 1);
		[array[i], array[j]] = [array[j], array[i]];
	}
	return array;
}
export type Split = BindingExample['split'];
/** Splits are fixed across model seeds. Every ordering and query of an assignment
 * remains in the same split. Distinct values make random input-copy accuracy 1/3. */
export const assignmentSplits = (() => {
	const all: number[][] = [];
	for (let a = 0; a < 8; a++)
		for (let b = 0; b < 8; b++)
			for (let c = 0; c < 8; c++) {
				if (a !== b && a !== c && b !== c) all.push([a, b, c]);
			}
	shuffle(all, new RandomStream(0x74535355));
	return { train: all.slice(0, 236), calibration: all.slice(236, 286), test: all.slice(286) };
})();
export function makeExample(
	assignment: number[],
	query: number,
	order: number[],
	split: Split
): BindingExample {
	const tokenIds: number[] = [];
	for (const key of order) tokenIds.push(key, 3, assignment[key] + 6, 4);
	tokenIds.push(5, query);
	return {
		id: `${assignment.join('')}:${order.join('')}:${query}`,
		tokens: tokenIds.map((id) => MODEL_CONFIG.vocabulary[id]),
		tokenIds,
		answer: String(assignment[query]),
		answerId: assignment[query] + 6,
		assignment: [...assignment],
		query: MODEL_CONFIG.vocabulary[query],
		split
	};
}
export function sampleExample(split: Split, random: RandomStream): BindingExample {
	const choices = assignmentSplits[split];
	return makeExample(
		choices[random.integer(choices.length)],
		random.integer(3),
		shuffle([0, 1, 2], random),
		split
	);
}
/** Balanced answers and unique sequences, independently sampled once and fixed. */
export function fixedExamples(split: 'calibration' | 'test', count: number): BindingExample[] {
	const random = new RandomStream(split === 'calibration' ? 0xca11b : 0x7e57);
	const examples: BindingExample[] = [];
	const seen = new Set<string>();
	for (let i = 0; i < count; i++) {
		const answer = i % 8;
		let example: BindingExample;
		do {
			const choices = assignmentSplits[split].filter((assignment) => assignment.includes(answer));
			const assignment = choices[random.integer(choices.length)];
			example = makeExample(
				assignment,
				assignment.indexOf(answer),
				shuffle([0, 1, 2], random),
				split
			);
		} while (seen.has(example.id));
		seen.add(example.id);
		examples.push(example);
	}
	return examples;
}
export const CALIBRATION_EXAMPLES = fixedExamples('calibration', 16);
export const TEST_EXAMPLES = fixedExamples('test', 96);
