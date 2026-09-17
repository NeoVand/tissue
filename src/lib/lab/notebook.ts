export interface NotebookEntry {
	id: string;
	date: string;
	kind: 'hypothesis' | 'decision' | 'observation';
	title: string;
	text: string;
	lesson: string;
	links?: { title: string; href: string }[];
}

/** Curated research record. Measurements live alongside this in exported runs. */
export const notebook: NotebookEntry[] = [
	{
		id: '009',
		date: '2026-09-16',
		kind: 'decision',
		title: 'Give every point an address in the model.',
		text: 'The first interface gave the map more prominence than its connection to computation. The workbench now links each MLP point to its layer, channel, incoming column, outgoing row, and token activation trace. An architectural layout provides a second view of those same units.',
		lesson:
			'Dark and light themes, dense linked instruments, and shader interpolation improve inspection. Interpolated display frames are not new measurements; similarity links are not computational edges.'
	},
	{
		id: '010',
		date: '2026-09-16',
		kind: 'observation',
		title: 'Much of the activation map separates layers.',
		text: 'At step 2,000, 95.0% and 89.5% of directed six-neighbor selections in activation space stay within a layer for seeds 42 and 7. The corresponding effect-space shares are about 69%. Activation and effect neighborhoods overlap by 18.5% and 20.1%, above an independent-uniform baseline of about 2.35%.',
		lesson:
			'Layer membership is a strong confound. The two maps describe different but related organizations; neither establishes semantic specialization or repair value. The next comparison must include within-layer structure and matched controls. The Findings panel and saved comparison report expose the measurements and null assumptions.'
	},
	{
		id: '008',
		date: '2026-09-16',
		kind: 'decision',
		title: 'Missing evidence stays missing.',
		text: 'A checkpoint capture can fail after weights have already changed. That interrupted run keeps its measured metrics, while recovery starts a separate specimen from the last complete checkpoint. Similarly, a zero-variance activation or zero-effect fingerprint cannot define a directional neighborhood.',
		lesson:
			'Training and intervention are blocked after failures until recovery. Undefined fingerprints are excluded from directional repair neighborhoods. Browser tests deliberately inject a capture failure to verify that the record and restored weights agree.'
	},
	{
		id: '005',
		date: '2026-09-16',
		kind: 'observation',
		title: 'A zero was a measurement failure.',
		text: 'The first headless WebGPU run returned all-zero synchronous readbacks, even for simple known-answer arithmetic. That would have produced a false zero loss. Asynchronous GPU readback returned the actual values.',
		lesson:
			'The engine now checks known-answer arithmetic and probability mass before accepting a backend. Production observations use asynchronous readbacks. Numerical validity comes before the chart.'
	},
	{
		id: '006',
		date: '2026-09-16',
		kind: 'observation',
		title: 'A plateau, then a rapid gain.',
		text: 'With seed 42, held-out accuracy was 32.29% at 500 updates, near the 33.33% random input-copy baseline. It reached 97.92% at 2,000 updates. Seed 7 independently went from 29.17% at 500 to 100% of the 96 fixed evaluation prompts at 2,000. Both showed a long plateau and a late rapid gain.',
		lesson:
			'Two seeds support investigating this transition. They do not establish a universal training schedule or prove a particular internal mechanism. The small fixed evaluation set is not a population-wide accuracy estimate.'
	},
	{
		id: '007',
		date: '2026-09-16',
		kind: 'observation',
		title: 'The first repair pilot did not favor our maps.',
		text: 'At seed 42, step 2,000, a neuron selected by its calibration intervention magnitude caused only a small held-out loss increase: 0.08027 to 0.08479. After 50 updates of eight same-layer units, outgoing-weight neighbors reached loss 0.06122, random units 0.06688, activation neighbors 0.07016, and intervention neighbors 0.07142.',
		lesson:
			'This was a weak lesion and one seed. Improvement below the intact loss also requires an unlesioned fine-tuning control. Next: stronger calibration-selected lesions, strength-matched random units, a repair-gradient baseline, and repeated seeds. No evidence of a superior functional geometry yet.'
	},
	{
		id: '001',
		date: '2026-09-16',
		kind: 'hypothesis',
		title: 'Can proximity predict repair?',
		text: 'A functional neighborhood may identify units that can compensate for a damaged neuron. Compare neighborhoods derived from activations and interventions against outgoing weights, matched random units, and repair gradients.',
		lesson:
			'Open question. Similar immediate effects do not establish redundancy or recoverability. Select neighborhoods before observing repair outcomes.'
	},
	{
		id: '002',
		date: '2026-09-16',
		kind: 'decision',
		title: 'Begin with a language we can check.',
		text: 'The first specimen learns variable binding: a=3; b=7; c=2; ?b → 7. A causal transformer predicts the answer token. Assignment mappings are separated between training, calibration, and evaluation. Answer accuracy is measured independently of the training loss.',
		lesson:
			'This is a controlled next-token task, not a test of general language ability. The atlas is built on calibration prompts; evaluation prompts assess learning.'
	},
	{
		id: '003',
		date: '2026-09-16',
		kind: 'decision',
		title: 'Let the evidence determine the shape.',
		text: 'Each point is an MLP neuron. Its fingerprint contains measured responses across a fixed collection of prompts and token positions. PCA produces three coordinates. Neighbor retention records how much local structure survives projection.',
		lesson:
			'A region in the picture is a candidate relationship. Projection can create false neighbors. Links indicate similarity, not synapses or causal influence.',
		links: [
			{ title: 'NeuroCartography', href: 'https://arxiv.org/abs/2108.12931' },
			{ title: 'TopoLM', href: 'https://topolm.epfl.ch/' }
		]
	},
	{
		id: '004',
		date: '2026-09-16',
		kind: 'observation',
		title: 'Motion can come from the measuring instrument.',
		text: 'Jaxverse’s earlier PCA view mirrored when eigenvectors changed sign or order. Interpolation made those coordinate changes look like a restructuring network. Tissue aligns successive maps and retains the unprojected measurements.',
		lesson:
			'Compare checkpoint metrics and neighborhood structure before interpreting visual movement as learning.'
	}
];
