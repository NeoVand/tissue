# TinyStories model engine validation

These are engineering smoke checks, separate from the matched-token-budget preset comparison. The corpus, held-out windows, and model objective are specified in [stories-design.md](./stories-design.md).

## Numerical contracts

`src/lib/stories/model.spec.ts` verifies five groups of contracts on a small diagnostic architecture, using the real checksum-verified corpus and WASM:

- Preset parameter totals and bounded architecture validation.
- Whole validation-story exclusion from training; calibration and evaluation windows drawn from different held-out stories; unigram frequencies computed exclusively from training characters.
- Explicit rejection of unsupported characters and recorded prompt truncation.
- Exact checkpoint tensor order, identical intact inference with activation capture, unchanged prefix activations when only a future token changes, and exact zeroing of the selected MLP channel at every position.
- Held-out loss improvement, correct unit-major atlas versus token-major probe indexing, and exact restored parameter/Adam/RNG trajectory despite intervening atlas capture, probing, and generation.

The suite passed with 5 tests. Model sources also passed ESLint and TypeScript checking; the concurrent application check reported only in-progress UI integration errors at the time of the engine freeze.

## Actual browser worker checks

Chromium 153 headless on this development Mac, using actual WebGPU and the Small preset (827,392 parameters, 2,048 MLP channels):

| Observation                                                           | Result                                      |
| --------------------------------------------------------------------- | ------------------------------------------- |
| Initial fixed held-out next-character CE                              | 4.563250                                    |
| After 5 training updates                                              | CE 3.211337; 19.78% next-character accuracy |
| Training-only unigram baseline on the same held-out windows           | CE 3.047754; 18.60% accuracy                |
| Initial five-update training interval, including gradient compilation | 1,972 ms                                    |
| Step 5                                                                | 184.1 ms                                    |
| Atlas capture                                                         | 2,048 × 128 floats in 936.3 ms              |
| Probe probability mass                                                | 1.000000057                                 |
| Selected-channel intervention                                         | Exactly zero at every real prompt position  |

A second, independent lifecycle check requested 100 updates and paused when the step-25 metrics event arrived. The worker acknowledged the pause at step 26, with CE 2.714868 and 28.37% accuracy. Its training interval was 12,235 ms including evaluation. This timing differs from the shorter smoke check and is not a controlled hardware benchmark. The sample `Once upon a time` → ` cto in sore` demonstrates why a better character loss is not a claim of coherent story generation.

The lifecycle check additionally verified:

- Probing, sampling, and lesion inference preserve every parameter, both Adam moments, and the training RNG exactly.
- A fresh worker can load a checkpoint without first initializing a random model; predictions and model identity are preserved exactly.
- Loading leaves the caller's typed checkpoint buffers attached.
- Cancelling an atlas measurement rejects it rather than returning a partial atlas.
- Worker disposal completes after both training and standalone restoration.

The raw local lifecycle receipt was saved to `/tmp/tissue-stories-engine-validation.json`. The reproducible research runner records its own separate provenance and artifacts; temporary smoke receipts are not substituted for recorded study checkpoints.

## Scope and limits

The graph represents post-ReLU MLP channels, and fingerprints capture activation on eight fixed calibration windows at sixteen positions each. Selected-unit intervention zeros that channel at every token position. It measures a direct intervention on this trained parameterization; geometric proximity alone does not establish shared causal function.

Generation uses its own seeded stream. Training uses packed training characters and the all-position next-character objective. Fixed evaluation windows cover 2,048 target characters; they are useful for controlled comparisons, not an exhaustive assessment of the 152 evaluation stories. Long-running coherence, generalization across seeds, and device memory limits require further experiments.
