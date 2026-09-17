# Larger TinyStories models: first capability study

Recorded before running the new language models. This iteration tests whether the lab can train, inspect, and preserve substantially larger models on natural-language data. It is an exploratory capability study, not a claim of a new interpretability method or a controlled scaling law.

## Corpus and prediction task

Use the attributed TinyStories subset already prepared by Pattern from Ronen Eldan and Yuanzhi Li's original release. The bundled token stream is pinned by SHA-256 `67fb35357ec45562152b012c60d4a2d1aec5a95ad9d9a4fc6d534bc97fb4c6db`. Its metadata, original license, byte counts, modifications, and split audit are retained in `static/data/tinystories`.

This is a 96-symbol character tokenizer: newline plus printable ASCII. The original preparation normalized punctuation. The 2,097 original training stories contain 1,773,785 characters. Three stories include internal blank lines, yielding 2,100 blank-line segments; do not relabel these as 2,100 original stories. Training samples fixed-length windows from the training stream, which can cross training-story boundaries.

The original validation subset contains 184 complete, distinct stories. Reserve its first 32 stories for calibration and the remaining 152 for evaluation. No whole validation story occurs verbatim within the training stream. Eight fixed calibration windows define the geometry; 16 fixed evaluation windows measure next-character loss and accuracy. Neither held-out partition enters gradient updates. This small, fixed evaluation panel is not the full TinyStories benchmark.

Predict every next character in a 128-character context. Display mean cross-entropy in nats per character, character accuracy, and a unigram baseline fitted on training characters only. Perplexities or losses here are not comparable to word-token or BPE-token benchmarks. Generation shows actual model output; it must not be presented as evidence of fluent storytelling merely because loss decreases.

## Configurations and initial training budget

All presets use a causal transformer with RMS normalization, attention, and a ReLU MLP of four times the residual width. Token embeddings and output weights are separate; the context is 128 characters. Every displayed node is one actual post-ReLU MLP channel, with an exact layer and channel address. Attention heads and residual dimensions are not additional plotted neurons.

| Preset | Layers | Width | Heads | MLP width | Parameters | Plotted units | Batch | Adam learning rate |
| ------ | -----: | ----: | ----: | --------: | ---------: | ------------: | ----: | -----------------: |
| Small  |      4 |   128 |     4 |       512 |    827,392 |         2,048 |     4 |              0.001 |
| Medium |      4 |   256 |     8 |     1,024 |  3,227,648 |         4,096 |     4 |             0.0006 |
| Large  |      6 |   384 |     8 |     1,536 | 10,739,712 |         9,216 |     2 |             0.0004 |

Start with seed 42 and **51,200 supervised training characters per preset**: 100 updates for Small and Medium, 200 for Large. Retain the untrained map and observations at 12,800, 25,600, and 51,200 trained characters. Separate initial compilation from completed training time. Preserve failures or resource limits rather than silently replacing a model with a smaller one. Any later training extension is recorded as such.

These configurations also differ in depth, heads, batch size, and learning rate. Matched token counts do not isolate the causal effect of parameter count or establish a scaling law. The goal is to measure practical execution and reveal the actual larger model structures for inspection.

## Geometry and interventions

Capture every MLP unit on eight fixed calibration windows at 16 declared positions per window: 128 raw activation coordinates per unit. The raw values remain in the archive. Center each unit's fingerprint and normalize its nonzero direction. A unit with no resolved variation has no directional placement; display its exclusion rather than inventing a location.

Fit the three-dimensional view through feature-space covariance and a deterministic iterative eigensolver. Report convergence, residual error, variance, and ambiguity near the third/fourth component boundary. Align successive measured frames without changing pairwise distances. Display interpolation between recorded frames is not an intermediate measurement. Architectural layout uses the same unit identities but derives positions from architecture instead of activation similarity.

Do not allocate an all-pairs distance matrix for thousands of units. Selected-unit nearest neighbors are exact scans of the original activation vectors. Projection-neighborhood retention is audited on a fixed, activation-independent sample of 128 focal IDs, including the denominator and unresolved cases. Sampled retention is not a full-population diagnostic. Similarity links are not computational or causal edges.

For a selected unit, permit a real all-position zero ablation on the user's prompt, alongside the intact prediction. Label the checkpoint, prompt truncation, token, layer, and channel. Whole-model activation maps and individual causal interventions are different evidence; no exhaustive intervention map is claimed for these larger models.

## Durable evidence

Each run retains configuration, corpus identity, metrics, raw activation snapshots, projected coordinates and diagnostics, generation settings/output, and observations. A resumable checkpoint contains parameters, Adam moments/counter, and training RNG. Probe and sampling operations must not advance that RNG. Binary `.tissue` archives store float32 arrays directly rather than converting millions of tensor elements to JSON numbers.

The latest complete checkpoint may precede the latest recorded measurement after an interruption; show both steps explicitly. Restoring earlier weights must preserve the interrupted evidence and start a separate continuation record. Reference archives may omit weights to keep a public artifact bounded, but must then be labeled inspection-only and offer no checkpoint resume. The 10.74M model's parameters and two Adam moment arrays alone require approximately 129 MB of float32 storage; browser GPU working memory is additional.

The source dataset is licensed CDLA-Sharing-1.0. Source: [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories), [dataset paper](https://arxiv.org/abs/2305.07759), and the retained license and preparation manifest. Architecture and training details in this document describe Tissue's implementation, not the paper's trained checkpoints.
