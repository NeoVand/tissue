# Paired-query measurement validation

The `paired-query-v1` instrumentation was checked independently of the neighborhood analysis. These checks establish numerical and provenance properties of the measurement, not the usefulness of the proposed representation.

## Implementation contract

- Each split uses 16 distinct assignment groups from its existing assignment split. None occurs in training or in the other measurement split. Assignment selection and presentation order use fixed local RNG seeds and do not depend on model seed, parameters, measured outcomes, or the training RNG.
- A group contains queries `a`, `b`, and `c`, in that order. All 13 preceding tokens are identical. Two batches of 24 examples per split preserve complete triples and keep JIT shapes fixed without padding.
- Activations are raw, final-token post-ReLU MLP values, shaped `256 × 48`. Effects are all-token zero-ablation minus intact probabilities for answer values 0–7, shaped `256 × 384` in group/query/answer order. These answer probabilities are not renormalized over digits. Accuracy uses argmax over the entire 14-token vocabulary.
- The operation compares capture against ordinary inference over all 14 output probabilities. It checks all earlier MLP activations within each triple. Either maximum difference above `1e-6` rejects the measurement. Nonfinite or negative activations, nonfinite log probabilities, and probability mass error above `1e-3` also reject it.
- `checkpointHash` is SHA-256 over UTF-8 `JSON.stringify(exportCheckpoint())`, including model parameters, Adam moments and counter, training RNG state, elapsed training time, and training loss. The complete serialized checkpoint must match exactly before and after a successful measurement.
- Progress counts 512 neuron/split measurements; each comprises two inference batches. Pause is cooperative, returns a cancellation error, and never returns a partially successful measurement. No measurement path writes to model parameters, optimizer, or training RNG.

## Automated checks

`src/lib/lab/model/query-model.spec.ts` checks assignment disjointness, fixed query ordering, identical prefixes, correct targets, independent metadata, matrix axes, probability mass, and the full operation on WASM after a real training update. An independent forward pass uses a different batch size and verifies the stored effects for global neuron 135, including exact zero activation at every token of its layer-2 channel 7. This catches layer/channel and group/query/answer indexing errors.

The test also verifies the checkpoint hash against Node's SHA-256 implementation and exact checkpoint preservation after success and cancellation. The query-only suite passed all three tests. Type checking reported no errors or warnings, and ESLint passed the changed measurement files.

## Actual browser check

An isolated headless Chromium session with WebGPU loaded the saved seed-42, step-2000 checkpoint and called the public worker API `Engine.measureQueryShifts()`. The completed sweep took **6,052.8 ms**.

| Check                                             | Calibration | Held-out test |
| ------------------------------------------------- | ----------: | ------------: |
| Correct full-vocabulary predictions               |     45 / 48 |       47 / 48 |
| Maximum earlier-prefix activation difference      |           0 |             0 |
| Maximum capture/prediction probability difference |           0 |             0 |
| Maximum probability mass error                    |    2.829e-7 |      3.174e-7 |

The resident checkpoint was unchanged after this complete sweep. A second request was paused after progress 8, rejected with `Paired-query measurement cancelled`, and again left the complete checkpoint unchanged. Worker disposal completed normally.

The browser checkpoint hash was:

```text
521061f9565bab8c6d14e386bd8713561884aff73b9dbc749510f836669c274a
```

The temporary validation artifact is `/tmp/tissue-query-seed42-step2000.json`; it is a local diagnostic artifact and is not checked in. The reproducible study runner, `scripts/measure-query-shifts.mjs`, separately records source hashes, the locked design hash, checkpoint-file provenance, raw measurements, and analysis. Its recorded study artifacts are the source for scientific interpretation. Timing above is one machine/browser measurement and not a performance guarantee.

## Review of the scoring boundary

An independent read-only review checked that the analysis uses the declared Helmert layout, performs no additional row centering, selects the same-layer strength-matched candidate pool solely from calibration measurements, and computes each method's neighbors before accessing held-out targets. The held-out score uses the declared zero-credit convention for unresolved neighbors and one common focal cohort. The random baseline is the exact expected mean for a uniform six-element sample without replacement: the candidate-pool mean. The shuffled control preserves each unit's resolved/unresolved status and leaves the calibration plan unchanged.

These checks found no selection leakage or inconsistency with the locked design. They do not establish population significance, semantic meaning, or that similar units can repair one another.

## Application and archive checks

The server suite passes 91 tests across seven files, including the measurement, neighborhood analysis, and raw-record validation contracts. The independent arithmetic audit also passes against both recorded studies and their implementation hashes. All four changed Svelte components pass the Svelte autofixer without issues or suggestions.

Production browser coverage exercises real model execution, reference inspection, linked raw probes, export/import, IndexedDB persistence, reload, cancellation, and checkpoint preservation. It also retains the earlier training, repair, mobile-layout, and injected-capture-failure checks. An import receives a fresh local archive ID while preserving its measurement and provenance, so it cannot replace prior evidence merely by reusing an external ID.

The final production run passed all five Playwright tests in 2.6 minutes. `pnpm check` reports zero errors and warnings; `pnpm lint` passes. The production build passes with Vite's existing advisory about chunks larger than 500 kB. Desktop dark/light and narrow-screen layouts were inspected, with no console errors in the final query-study view.

A separate boundary test holds a successful worker response after the real `512/512` progress event. Cancellation must keep model-changing controls locked until that response is handled; releasing it must leave the completed raw measurement both archived and exportable. This specifically guards the finalization interval after numerical progress has completed. The UI clears a previous analysis before assigning newly captured evidence, and background checkpoint restoration preserves the user's selected lab view.
