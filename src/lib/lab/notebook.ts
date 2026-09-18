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
		id: '023',
		date: '2026-09-18',
		kind: 'decision',
		title: 'Keep the next action within reach.',
		text: 'The first interface pass left evidence below the full sidebar height, creating an empty gap beneath shorter visualizations. It also buried weight loading and intervention measurement behind tools. Evidence now follows its own canvas column. TinyStories can load trained weights and begin live generation from one action; the intervention map offers its prerequisite measurement directly.',
		lesson:
			'A disabled control needs an explanation and a useful next step. Saved activations are not a resident model, and an activation map is not an intervention map. These distinctions belong beside the action, where researchers can act on them.',
		links: [
			{
				title: 'Interface audit and corrections',
				href: 'https://github.com/NeoVand/tissue/blob/main/docs/interface-audit.md'
			}
		]
	},
	{
		id: '022',
		date: '2026-09-17',
		kind: 'decision',
		title: 'Give the instrument a clear hierarchy.',
		text: 'The interface audit found competing toolbars, fixed narrow sidebars, and labels as small as 7–9 pixels. The lab now puts the measured network first, with one resizable Model/Inspect panel, readable typography, and named disclosures for setup, training, projection diagnostics, and evidence. The journal presents a readable sequence of decisions rather than three crowded columns.',
		lesson:
			'Progressive disclosure must preserve the information needed to interpret a view. Model identity, measured checkpoint, generated text, raw selected activity, and projection caveats remain accessible. This redesign changes the instrument, not the recorded experiment or the strength of its evidence.',
		links: [
			{
				title: 'Interface audit and interaction design',
				href: 'https://github.com/NeoVand/tissue/blob/main/docs/interface-audit.md'
			}
		]
	},
	{
		id: '021',
		date: '2026-09-17',
		kind: 'decision',
		title: 'Make the laboratory shareable.',
		text: 'GitHub Actions builds and publishes the lab to GitHub Pages, with trained reference specimens available for inspection. The README opens with a real high-resolution capture of the 1.85M-parameter model at step 4,096, paused during measured replay. Dataset and checkpoint requests resolve beneath the repository hosting path in both the interface and model workers.',
		lesson:
			'Publishing the instrument is not a new model experiment. Original data and provenance stay unchanged; the predeployment checkout preserves the source hashes for strict historical audits. Training and personal archives remain local to each browser.',
		links: [
			{ title: 'Public laboratory', href: 'https://neovand.github.io/tissue/' },
			{
				title: 'Deployment, screenshot, and historical source verification',
				href: 'https://github.com/NeoVand/tissue/blob/main/docs/github-pages.md'
			}
		]
	},
	{
		id: '020',
		date: '2026-09-16',
		kind: 'decision',
		title: 'Keep the quiet neurons in view.',
		text: 'The subword viewer now defaults to fixed-size neuron cores with activity-driven brightness and shader glow. Size remains a separate encoding. Starting live generation or recorded replay preserves the chosen layout and layer filter, and the primary action becomes Pause or Resume in place.',
		lesson:
			'A visible baseline describes the model structure, not positive activity. Brightness remains relative to the visible maximum in each live frame; raw values stay in the inspector and archive. These are improvements to the viewing instrument, not new model measurements or evidence of a discovered circuit.',
		links: [
			{
				title: 'Encoding and playback method',
				href: 'https://github.com/NeoVand/tissue/blob/main/docs/live-activations.md'
			}
		]
	},
	{
		id: '019',
		date: '2026-09-16',
		kind: 'decision',
		title: 'Follow each token through measured layers.',
		text: 'Live subword generation now waits for the viewer after every forward pass. The lab presents actual post-ReLU MLP values layer by layer, then reveals the sampled token and advances the exact token-ID context. Pause, layer stepping, token stepping and recorded replay share the same measurements. A 24-token WebGPU trace from the saved 4,096-update model preserves all 2,048 channel values and 4,096 raw output probabilities per frame; weights, Adam state and training randomness remained unchanged.',
		lesson:
			'This is an instrument, not evidence of a discovered circuit. The final input position predicts the next token; paced layer playback does not measure GPU timing or show attention and residual-stream internals. Brightness is normalized within each frame; the inspector retains raw values. The replay still changes a little girl called Tom into “He”. Its training evidence is inherited from Study 004, not a new training run. Continuous training now saves maps and weights every 100 updates and on pause. The spatial map remains 3D PCA; token progression supplies time.',
		links: [
			{
				title: 'Live measurement and playback method',
				href: 'https://github.com/NeoVand/tissue/blob/main/docs/live-activations.md'
			},
			{
				title: 'Measured replay and checkpoint-preservation checks',
				href: 'https://github.com/NeoVand/tissue/blob/main/static/experiments/token-stories-live-demo-d474f204-report.json'
			}
		]
	},
	{
		id: '018',
		date: '2026-09-16',
		kind: 'observation',
		title: 'Story-like phrases emerge; the shape still loses most neighbors.',
		text: 'The 1.85M-parameter subword model completed the declared 4,096 updates on 1,868,552 nonpadding targets. Held-out loss reached 3.560 nats/token and accuracy 31.1%, versus 6.059 and 6.1% for the train-only unigram baseline on the same 2,040 fixed targets. All four measured maps and all five samples are retained. The continuations contain recognizable story phrasing but inconsistent pronouns, grammar and events.',
		lesson:
			'The final 3D map retains 6.12% of audited six-neighbor memberships, with 11.56% explained variance. Retention was 3.65%, 9.24% and 5.47% at the earlier captures: predictive improvement does not imply a more faithful projection. Inspect exact token responses and original-space neighbors. One seed and sixteen evaluation windows do not establish narrative competence; tokenization and training budget changed together. Next: matched story facts, disjoint templates and repeated intact controls before interpreting causal neighborhoods.',
		links: [
			{
				title: 'Results, complete samples and limitations',
				href: 'https://github.com/NeoVand/tissue/blob/main/docs/token-stories-results.md'
			},
			{
				title: 'Independent binary evidence audit',
				href: 'https://github.com/NeoVand/tissue/blob/main/static/experiments/token-stories-audit.json'
			}
		]
	},
	{
		id: '017',
		date: '2026-09-16',
		kind: 'decision',
		title: 'Give the context a larger unit of language.',
		text: 'The previous TinyStories specimens predict characters and saw only 51,200 training characters. Study 004 adds a deterministic 4,096-piece vocabulary fitted only on the 2,097 training stories. Its 443,892 training tokens include explicit story boundaries. The same 32 calibration and 152 evaluation stories stay separate. On training text, one BPE text token averages 4.02 characters; 148 of the 152 evaluation stories fit a 256-token context including boundaries.',
		lesson:
			'Subword units extend effective context and make token-by-token probes easier to read. They do not guarantee coherent stories. This first run also increases training, so it cannot isolate tokenization as the cause of any improvement. Loss is now nats per subword token, and should not be numerically compared with the character loss. Keep both instruments and preserve every generated sample.',
		links: [
			{
				title: 'Study 004 protocol recorded before measurement',
				href: 'https://github.com/NeoVand/tissue/blob/1e1e7b5/docs/token-stories-design.md'
			}
		]
	},
	{
		id: '016',
		date: '2026-09-16',
		kind: 'observation',
		title: 'A tiny computed effect can overlap numerical variation.',
		text: 'Restoring the full 10.74M checkpoint in a separate browser worker preserved its step and held-out accuracy; held-out loss differed by about 7e-8 nats. For the same prompt, the maximum intact probability difference across workers was 2.68e-7. The prespecified unit-4608 ablation changed a probability by at most 3.87e-7 in the original worker and 5.22e-7 after restoration.',
		lesson:
			'This single cross-worker comparison does not calibrate a numerical noise floor. It does show why this tiny ablation cannot support a substantive causal interpretation by itself. Raw intact/ablated distributions are retained; small displayed effects use scientific notation. Future causal studies need repeated intact controls and effects large enough to distinguish from numerical variation.',
		links: [
			{
				title: 'Raw restoration comparison',
				href: 'https://github.com/NeoVand/tissue/blob/main/static/experiments/stories-large-restoration.json'
			}
		]
	},
	{
		id: '015',
		date: '2026-09-16',
		kind: 'observation',
		title: 'Larger models run; attractive geometry still loses neighbors.',
		text: 'At the declared 51,200-character budget, the 827k, 3.23M and 10.74M TinyStories models reached held-out losses of 2.453, 2.454 and 2.434 nats per character, versus a 3.048 training-unigram baseline. All three trained on WebGPU and retained four measured maps, samples and raw selected-unit interventions. Their generated text remains fragmented.',
		lesson:
			'The final 3D maps retain 25.8%, 16.8% and 22.5% of audited six-neighbor memberships, over 119, 113 and 111 scorable focal units respectively. Every channel was captured, but unresolved directions are omitted from functional placement. These shapes are navigable measurements, not established semantic circuits. Next: train longer, repeat seeds, then test whether neighborhoods predict held-out intervention responses.',
		links: [
			{
				title: 'Results, raw samples, and limitations',
				href: 'https://github.com/NeoVand/tissue/blob/main/docs/stories-results.md'
			}
		]
	},
	{
		id: '014',
		date: '2026-09-16',
		kind: 'decision',
		title: 'Scale the specimen; keep the measurements accountable.',
		text: 'Study 003 moves from the 25,920-parameter binding model to TinyStories character models with 827,392, 3,227,648 and 10,739,712 parameters. Every MLP channel is measured: 2,048, 4,096 and 9,216 points with real layer/channel addresses. The corpus is attributed to its source and split by story before fixed calibration and evaluation windows are selected.',
		lesson:
			'The first comparison fixes 51,200 training characters and seed 42. Architecture, learning rate, and batch size differ, so this is not a causal scaling law. PCA uses every resolved fingerprint, while projection retention audits 128 preselected focal IDs. The TinyStories workspace records maps, loss curves, generated samples, exact selected-unit ablations, and binary archives.',
		links: [
			{
				title: 'Design recorded before the scale comparison',
				href: 'https://github.com/NeoVand/tissue/blob/a73098c/docs/stories-design.md'
			}
		]
	},
	{
		id: '013',
		date: '2026-09-16',
		kind: 'observation',
		title: 'Zero final-token activity can coexist with query-dependent effects.',
		text: 'Seven first-layer units in seed 42 and 47 in seed 7 have exactly zero final-token activation on all 48 paired calibration prompts, yet nonzero query-dependent effects when silenced at every position. They are excluded from the locked common cohort because the query-activation comparator has no direction. Unit 88 in seed 7 is the strongest such unit by calibration query-effect RMS: 0.0181; its held-out RMS is 0.00151.',
		lesson:
			'The activation probe measures one position; the intervention changes all positions. These observations are compatible with effects through earlier positions and later processing, but do not isolate the route. The next mechanism test is a location-specific lesion, with units chosen on calibration evidence only. Query shifts lets us inspect excluded units and their raw paired responses.',
		links: [
			{
				title: 'Evidence and cohort audit',
				href: 'https://github.com/NeoVand/tissue/blob/main/docs/query-shifts-results.md'
			}
		]
	},
	{
		id: '012',
		date: '2026-09-16',
		kind: 'observation',
		title: 'Neighborhoods transfer; query contrasts do not beat full effects.',
		text: 'The locked paired-query study was measured on both trained checkpoints. Query-effect neighbors scored mean held-out cosine 0.324 and 0.279 for seeds 42 and 7, versus exact matched-pool random expectations 0.066 and 0.062. Full-effect neighbors scored slightly higher: 0.334 and 0.292. The primary advantage over random fell to 0.008 and 0.003 after the descriptive identity shuffle.',
		lesson:
			'There is cross-assignment neighborhood structure beyond this same-layer, approximately strength-matched control. The fixed 32-candidate pool can contain large strength ratios. The new contrast representation has not improved on full effects. The common eligible cohorts contain 248 and 204 of 256 units, and outgoing weights remain a strong baseline. These are two checkpoints, dependent neuron pairs, and measured held-out effects—not predictions of unmeasured outputs.',
		links: [
			{
				title: 'Results and limitations',
				href: 'https://github.com/NeoVand/tissue/blob/main/docs/query-shifts-results.md'
			}
		]
	},
	{
		id: '011',
		date: '2026-09-16',
		kind: 'hypothesis',
		title: 'Change the question; hold the assignments fixed.',
		text: 'Study 002 tests whether similarity of query-dependent ablation effects on calibration assignments transfers to disjoint held-out assignments. The protocol was committed as ca45f91 before reading the new measurements: 16 paired groups per split, six neighbors, a shared same-layer pool of 32 units matched by full-effect RMS, and fixed orthonormal query contrasts.',
		lesson:
			'All methods use the same eligible focal units and the same candidate pool. The random comparator is its exact expected mean. Fixed numerical floors and a single identity shuffle are declared controls, not significance tests. Raw measurements, checkpoint hashes, and study history live in Query shifts.',
		links: [
			{
				title: 'Protocol recorded before measurement',
				href: 'https://github.com/NeoVand/tissue/blob/ca45f91/docs/query-shifts-design.md'
			}
		]
	},

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
