/** Catalog entries describe captured measurements, not promised performance. */
export const references = [
	{
		seed: 42,
		path: '/experiments/binding-seed-42.json',
		title: 'Seed 42 · a late learning transition',
		description:
			'32.3% at step 500 → 97.9% at step 2000. Two captured anatomies and one repair pilot with a mild lesion.',
		accuracy: 94 / 96,
		step: 2000
	},
	{
		seed: 7,
		path: '/experiments/binding-seed-7.json',
		title: 'Seed 7 · a second late transition',
		description:
			'29.2% at step 500, 32.3% at step 1500 → 100% at step 2000. Three captured anatomies; no repair experiment.',
		accuracy: 1,
		step: 2000
	}
] satisfies Array<{
	seed: number;
	path: string;
	title: string;
	description: string;
	accuracy: number;
	step: number;
}>;
