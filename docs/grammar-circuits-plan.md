# How grammar develops: circuit research plan

**Status: proposed study, 18 September 2026.** This is a research and implementation plan, not a report of discovered grammar circuits. No new training or grammar evaluation was performed while preparing it. The code audit uses Tissue at `b42525f`; the checkpoint audit below verifies stored bytes, not new model execution.

The next question should be concrete: **as a transformer learns to choose the right verb despite a distracting noun, what changes first—the representation of the subject, the routing of information through attention, or its use in the output?** Build an instrument that connects that question to sentences, checkpoints, interventions, and reproducible evidence.

## What we can build on

| Existing evidence or capability                                                                                           | Consequence for the next study                                                                                                                                                                                                                                                                                                                                         |
| ------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A 1.85M-parameter BPE model trained for 4,096 updates; held-out loss 3.560 nats/token versus 6.059 for a unigram baseline | There is useful language learning, but neither this metric nor fluent fragments establish grammatical competence.                                                                                                                                                                                                                                                      |
| Full local checkpoints at updates 0, 256, 1,024, and 4,096                                                                | We can start with a retrospective pilot. On 18 September, all four files matched the byte counts and SHA-256 hashes in the [local-copy manifest](../static/experiments/token-stories-local-copies.json). They still need restore and inference checks in the new runner. Only the final full model is currently public; earlier public maps are not resumable weights. |
| Four attention layers, four heads per layer, and 2,048 MLP channels in the compact model                                  | Sixteen heads make direct intervention sweeps practical to investigate before introducing gradient approximations. Throughput still needs measurement.                                                                                                                                                                                                                 |
| Live frames contain final-input-position MLP activations; probes can capture MLP activity across a prompt                 | We have neither attention measurements nor residual-stream traces. Layer playback is display pacing, not a recording of execution timing.                                                                                                                                                                                                                              |
| Current neuron masks apply across every position                                                                          | Existing lesions cannot tell us where a channel matters. Position-specific interventions are a prerequisite for circuits.                                                                                                                                                                                                                                              |
| Earlier binding experiments found channels with zero final-token activation but nonzero effects from all-position lesions | Token location is already a concrete missing variable. Repeat those cases with prefix-only and final-token-only lesions to validate the new instrument.                                                                                                                                                                                                                |

The last finding comes from the [paired-query results](query-shifts-results.md). It suggests testing downstream routing, but does not identify an attention path. Likewise, the [TinyStories results](token-stories-results.md) show that the final 3D projection retains only 6.12% of audited neighbors. Circuit discovery must operate on original tensors and interventions, independent of the map.

## What to borrow from current research

Anthropic's [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) offers a useful workflow: propose a computational explanation, expose its approximation errors, and test it with interventions. Its feature graphs use replacement models; attribution and successful reconstruction alone do not establish mechanistic faithfulness. We should begin with our model's native heads, MLP channels, and residual streams before adding another learned model.

Their subsequent [attention tracing work](https://transformer-circuits.pub/2025/attention-qk/index.html) addresses why attention selects positions through query–key feature interactions. This matters: an attention map shows a routing pattern, while explaining that pattern and explaining the content transmitted are separate tasks. The lab should distinguish **QK routing** from **OV content**, rather than treating a bright attention cell as a causal explanation.

Grammar circuits and their development are already research topics. [Tigges et al.](https://arxiv.org/html/2407.10827v2) explicitly study subject–verb agreement across training and model scales. [Ferrando and Costa-jussà](https://aclanthology.org/2024.findings-emnlp.591/) investigate agreement mechanisms across languages. Our initial agreement study is an instrument-building replication and extension, not a claim to be first.

Recent Anthropic work on [model differences](https://www.anthropic.com/research/diff-tool), and earlier [crosscoders](https://transformer-circuits.pub/2024/crosscoders/index.html), also caution against claiming that feature comparison across checkpoints is new. Tissue's potential contribution is a tightly controlled, interactive account of how causal roles develop—and eventually a spatial description that makes useful, held-out predictions.

## First experiment: agreement under distraction

Begin with a balanced subject-number × distractor-number design:

| Prefix                     | Preferred continuation | Competitor |
| -------------------------- | ---------------------- | ---------- |
| The key near the cabinet   | is                     | are        |
| The key near the cabinets  | is                     | are        |
| The keys near the cabinet  | are                    | is         |
| The keys near the cabinets | are                    | is         |

These are illustrative examples, not a finalized evaluation set. Use familiar TinyStories vocabulary, audit naturalness and tokenizer boundaries, and add no-distractor controls. Start with short prepositional phrases; introduce longer distances and relative clauses as separately labeled difficulty groups.

Generate matched sentence families with separate lexical and template holdouts. Keep all variants of a family together when splitting or estimating uncertainty. An adapted agreement subset of [BLiMP](https://github.com/alexwarstadt/blimp) can provide an external check; label any adaptation explicitly and preserve attribution and licensing. Do not call an adapted subset a full BLiMP score.

Score both alternatives with the model's **raw conditional log probabilities**, without sampling temperature or top-k filtering. Record the signed margin:

`margin = log P(grammatical continuation | prefix) − log P(ungrammatical continuation | prefix)`

For multi-piece continuations, teacher-force and sum all continuation log probabilities. Preserve exact token IDs, spaces, word boundaries, and context truncation rules. Verify that the shared text prefix really has a shared token prefix; do not silently compare different tokenization contexts. Start causal comparisons with position-aligned pairs, and represent word spans explicitly rather than assuming one word equals one token.

Show balanced accuracy, margin distributions, and results for each distractor condition alongside held-out story loss. Compare against unigram frequency and a nearest-noun-number heuristic; add a corpus-trained local-context baseline to test whether shallow continuation statistics explain the result. Report ties and exclusions. Free samples remain useful examples, not the grammar metric.

Use the existing checkpoints for an exploratory pilot. Before confirmation, lock the generator, exclusions, discovery/evaluation split, intervention sites, and primary outcome in a versioned protocol. A provisional budget is 128 discovery and 256 held-out sentence families, followed by three training seeds if the pilot supports a tractable study; these counts are planning estimates, not a power analysis. Never select a favorable seed after inspecting held-out results.

If the trained model does not reliably handle the task, that is the first result. Expand training or model capacity as an explicitly new condition. A synthetic grammar curriculum could answer a useful controlled question, but it must be a separate experiment from grammar learned through TinyStories.

## The new instruments

### 1. A faithful trace of the forward pass

Add an opt-in diagnostic path to [the BPE model](../src/lib/token-stories/model.ts), capturing selected positions and layers:

- Residual stream before and after each sublayer, including normalized inputs.
- Q, K, V, causal attention probabilities, and each head's output contribution.
- MLP preactivations, post-ReLU channels, output writes, and final raw logits.

Every observation needs a checkpoint identity, prompt and token IDs, tensor site, layer/head/channel, backend, and capture version. Keep the normal training path fast. If attention probabilities require an explicit QK/softmax/AV implementation, compare it against the existing attention operation on both WebGPU and WASM before accepting its traces. Check causal masking, padded positions, head reshaping, unchanged weights, and numerical error against repeated forwards.

Budget storage before adding visual complexity. At 128 positions, all 16 float32 attention matrices occupy 1 MiB per prompt; all MLP channel activations occupy another 1 MiB, before residuals and other tensors. Capturing only the latest attention row takes 8 KiB. Use selective live capture, bounded buffers, and deeper captures at chosen checkpoints. Display measured latency and bytes. Old live archives remain readable and explicitly lack the new fields.

### 2. An attention inspector with an output ledger

Select a token and head to see its attention row or matrix, the tokens supplying values, and the head's signed contribution to the chosen output contrast. Expand to compare heads when needed. This should work beside live generation, with a fixed-prompt mode for controlled comparisons.

The current architecture gives us a useful exact accounting identity. It applies no final normalization before the linear language-model head. For two single-token alternatives, let `d` be the difference between their output-weight columns. At the prediction position:

`logit margin = embedding residual · d + Σ(head output write · d) + Σ(MLP output write · d)`

All terms are measured in the same forward pass; the sum should reconstruct its margin within numerical tolerance. This is ordinary residual-stream accounting, not a new theorem. It describes direct contributions, not total causal effects: changing an early write can change later attention and MLP computations. Multi-token continuation scores require separate forward steps and cannot be treated as this single identity.

### 3. A counterfactual workbench

Present an intact prompt, its number-swapped counterpart, and the intervened run together. Let the researcher restore a selected residual, head output, or MLP activation from one run into the other at an explicit token position. Recompute downstream normally. Show raw before/after margins and probabilities, including both directions of the swap.

Keep a fixed output contrast across a counterfactual pair. For example, measure `log P(is) − log P(are)` throughout, even when the plural prompt changes which alternative is grammatical. If displaying normalized recovery, expose its denominator and mark near-zero baseline gaps undefined rather than manufacturing a large percentage.

Controls include identity patches, repeated forwards, matched random sites, position controls, and clean-to-corrupt as well as corrupt-to-clean patches. Compare natural counterfactual replacement with zero ablation; they answer different questions. [Activation-patching methodology](https://arxiv.org/abs/2309.16042) shows why corruption and metric choices need explicit treatment.

Start with exact head/block sweeps. Later, use [EAP-IG](https://arxiv.org/abs/2403.17806) or [AtP*](https://arxiv.org/abs/2403.00745) to shortlist large search spaces, and validate selections with actual interventions. Estimated importance must remain visibly distinct from measured effects.

### 4. A circuit view whose edges have a meaning

A circuit node is a **component at a token position**, not simply a neuron. Use token position and layer as stable axes, with heads and MLP sites inside each layer. Offer a readable 2D view and a linked 3D view of the same evidence. Keep 3D activation geometry as an optional comparison lens.

A node's ablation effect does not identify a direct edge. Start with a site-effect heatmap; add candidate paths, then edge/path interventions with a declared background before labeling a path causally tested. Attention weights, direct logit contributions, gradient estimates, and patch effects need separate legends and units. Include negative contributions and interactions; do not imply that positive paths alone explain a prediction.

For any proposed circuit, measure behavior when damaging it, restoring it, and retaining it against a declared counterfactual background. Evaluate on untouched sentence families and compare matched random circuits. Record unexplained output, failed rescues, and sensitivity to the background. A faithful subgraph need not be unique or minimal; redundant mechanisms can hide individual necessity.

Learned sparse features become appropriate if native components resist interpretation. Start with a per-layer SAE or transcoder, and measure reconstruction, output preservation, and intervention agreement separately. Retain reconstruction-error terms in graphs. Do not jump directly to cross-layer transcoders or assign semantic labels from a handful of attractive examples. Decoding subject number from an activation also does not prove the model uses it; [probed information can exceed causally relevant information](https://aclanthology.org/2023.eacl-main.58/).

## Make the research visible in the interface

Preserve the current action bar and restrained visual design. Promote the buried instruments into workspace destinations: **Network, Training, Samples, Circuits, Evidence**. These select the main content; Generate, Train, Probe, and Intervene remain actions. Avoid another stack of nested toolbars. Attention belongs inside Network and Circuits as a lens.

All views share the selected run, checkpoint, prompt, and evaluation cohort. Let a researcher pin one secondary view beside the main one with a resizable divider:

- **Training:** readable loss and grammar curves; choose a retained checkpoint to inspect its actual weights. Metrics-only checkpoints must say so. Compare exposure using trained tokens, not just updates.
- **Samples:** fixed prompts and sampling seeds across checkpoints, plus live generation; keep failures visible.
- **Circuits:** the sentence, output contrast, site effects, and selected causal paths. A checkpoint comparison shows how those effects change.
- **Evidence:** hypotheses, protocols, raw artifacts, controls, exclusions, and conclusions attached to each experiment.

The visually compelling scene is a small number of inspectable paths through the actual sentence, coordinated with changing predictions—not thousands of decorative edges. Use desaturated signed colors, readable labels, restrained glow, and exact values on selection. Smooth camera and geometric transitions; never interpolate unmeasured activations or present interpolated checkpoints as executed models.

## What could become a research contribution

Test whether **causal routing changes predict generalization beyond what training loss and activation probes predict**. At each checkpoint, compare subject-number decodability, routing sensitivity, and downstream readout effects. Does a model already contain number information but fail to use it across distractors? Does a successful route remain stable while the responsible heads change? These are hypotheses, not expected discoveries.

Use continuous trajectories and interval estimates. With four checkpoints, we can bound an observed change between measurements; we cannot identify its exact birth or establish an abrupt phase transition. In new runs, sample early training more densely and repeat across seeds. Any checkpoints added after seeing a transition are exploratory until confirmed under a locked schedule.

A later spatial experiment can embed components by their **signed intervention effects across sentence families**. Test whether that geometry predicts unseen rescue or transfer better than activation similarity, output weights, and layer/strength-matched random controls. This reconnects to Tissue's original idea with a falsifiable purpose. The earlier negative neighborhood results remain relevant; a different-looking map is not evidence of improvement. A broader literature review would still be needed before claiming novelty.

## Delivery sequence and stopping rules

| Milestone                       | Reviewable deliverable                                                                                                   | Evidence required to continue                                                                                                                                                               |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Bring the evidence forward   | Training, Samples, and Evidence workspace views; frozen agreement pilot and scorecards for the four existing checkpoints | Restored checkpoints reproduce reference inference; tokenizer and scoring checks pass; report failures and whether this model has measurable agreement ability.                             |
| 2. Observe attention faithfully | Selectable head matrices, residual/output ledger, selective capture, and position-specific lesions                       | Diagnostic and normal forwards agree within declared tolerances; ledger reconstructs the margin; binding-task prefix/final-position controls behave as predicted by the intervention scope. |
| 3. Test a candidate mechanism   | Paired-prompt workbench, exact head/block patches, then a small tested path graph                                        | Effects exceed repeatability error, survive held-out families, and improve over matched controls. If no compact circuit is faithful, show that result.                                      |
| 4. Study development            | Linked checkpoint/circuit comparison, denser training traces, repeated seeds                                             | Separate changes in representation, routing, and readout; report generalization and uncertainty without cherry-picking successful heads or seeds.                                           |
| 5. Extend only when justified   | Sparse features or intervention-derived geometry; larger-model comparison                                                | The added method improves a declared held-out prediction or explanation criterion enough to justify its cost.                                                                               |

Each milestone should produce an inspectable artifact, a journal entry, and a committed protocol or result. Record source and checkpoint hashes, dataset splits, exact interventions, sampling settings, precision, timing, and errors. Keep new studies separate from the historical source-hash-locked experiments. First measure capture and patch-sweep cost, then choose browser batch sizes and checkpoint frequency; no unmeasured performance promises.

The first complete demonstration should let someone choose a sentence, compare **is/are**, inspect a head, replace its measured output with a counterfactual activation, observe the changed prediction, and revisit the same test at another checkpoint. That is a useful instrument even if the first hypothesis fails.
