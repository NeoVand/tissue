# Binding: measured references

These files contain two actual browser training runs and a controlled repair pilot, captured on September 16, 2026. Both runs used JaxJS in headless Playwright Chromium with its **WebGPU** backend. The captures did not preserve a browser user-agent string, browser version, or graphics-adapter identity; none is inferred here. Available source provenance is described per seed.

## Seed 42

The seed-42 capture did not record a source commit or source hashes.

- `binding-seed-42.json` is an importable Tissue `RunRecord`: 81 measured evaluations (initialization and every 25 updates through step 2000), **only two** captured anatomy snapshots (500 and 2000), the latest raw activation/effect fingerprints, latest model and optimizer weights, a probe, and the actual four-arm repair result.
- `binding-seed-42-500-raw.json` preserves the earlier capture byte for byte, including its raw atlas, weights, optimizer state, trajectory, probe, and timings.
- `binding-seed-42-provenance.json` records capture times, original input checksums, packaging checksum, available timings, model configuration, and the recorded checkpoint-restoration/cancellation checks.

No anatomy was fabricated for uncaptured steps. The 3D maps are recomputed from raw fingerprints using the same `buildGeometry` code as the lab. Step-2000 maps are aligned to the respective step-500 maps; activation and intervention coordinate frames are separate. Invalid fingerprints remain explicitly invalid in the data. The package creation date uses the earliest retained atlas timestamp; it is not an estimate of training start time.

The model is `binding-transformer-v1`: two layers, width 32, four attention heads, 128 MLP units per layer, sequence length 14, vocabulary size 14, and 25,920 parameters. Training uses batches of 32 and Adam learning rate 0.002. Seed 42 selects the recorded sampling stream. Assignment sets are disjoint: 236 train, 50 calibration, and 50 test assignments. Evaluation uses 96 fixed test prompts; atlases use 16 fixed calibration prompts. See `src/lib/lab/protocol.ts`, `model/dataset.ts`, and `model/transformer.ts` for the executable definitions.

At step 500, held-out answer accuracy was **31/96 (32.3%)**, near the 33.3% random input-copy baseline. At step 2000 it reached **94/96 (97.9%)**. This is evidence of learning in one measured run; the two captured anatomies cannot establish when a geometric change occurred or whether it caused the behavioral improvement.

The repair pilot disabled neuron 172, chosen by the largest effect energy on calibration prompts. The injury was small: accuracy fell from 94/96 to 93/96 and answer loss rose from 0.0803 to 0.0848 nats. Each arm updated eight surviving units in the same layer (512 parameters) for 50 steps, from the same damaged checkpoint. Final answer loss was 0.0702 for activation neighbors, 0.0714 for intervention neighbors, 0.0612 for outgoing-weight neighbors, and 0.0669 for random neighbors. Random neighbors were not matched for activation strength. **This single-seed, single-lesion pilot supports no claim that either functional map predicts superior recovery.** Its practical result is that the next experiment needs a more informative injury and multiple seeds and lesions.

Repackage the original captures with Node 24 or newer:

```sh
node scripts/package-reference.mjs --input-dir /path/to/captures --output-dir static/experiments
```

The input directory must contain `tissue-reference-run.json` (step 500), `tissue-reference-2000.json`, and `tissue-repair-2000.json`. Individual paths may be overridden with `--capture500`, `--capture2000`, and `--repair`. The script checks shared initialization and the exact trajectory prefix before packaging, performs no training, and keeps the earlier raw capture unchanged. The latest raw atlas, checkpoint, and metrics are retained in the importable run; packaging-specific timings and original checksums are in the provenance file. Re-running training on another backend is a separate experiment and need not be bitwise identical.

## Seed 7 replication

`binding-seed-7.json` contains a second measured run: 81 evaluations, captured activation and intervention anatomies at steps **500, 1500, and 2000**, and the latest resumable checkpoint. Accuracy was 28/96 (29.2%) at step 500, 31/96 (32.3%) at step 1500, and 96/96 (100%) at step 2000. At step 1750 it was still 28/96 (29.2%). Final held-out answer loss was 0.0213 nats. No repair experiment was performed for seed 7.

All three original capture files are retained as `binding-seed-7-{step}-raw.json`. The original `provenance.json` and `summary.json` are preserved byte for byte as `binding-seed-7-provenance.json` and `binding-seed-7-summary.json`. Unlike the earlier seed-42 capture, seed 7 records source-file hashes and the base commit with `workingTreeDirty: true`; the base commit alone does not identify the executing source. Browser version and graphics-adapter identity remain unrecorded. `binding-seed-7-package.json` records input and output checksums without changing the original provenance.

The same architecture and fixed evaluation set were used. The second late accuracy improvement makes the behavior worth investigating, but two seeds and sparse anatomy checkpoints do not establish a universal transition, a causal geometric explanation, or a ranking of the two maps.

Package a complete `scripts/verify-model.mjs` output directory:

```sh
node scripts/package-reference.mjs --verify-dir /tmp/tissue-study-seed7 --output-dir static/experiments
```

This path reads the recorded seed and checkpoint list from `provenance.json`, checks every capture against it and the final summary, retains the original files, and builds only the anatomies actually captured. The script supports the capture structure produced by `verify-model.mjs`; it does not launch a browser or train a model.


## Paired-query neighborhood transfer

Study 002 uses the same trained checkpoints with a new fixed paired-query measurement design. It performs no training. The [protocol](../../docs/query-shifts-design.md) was committed before results, and the [results report](../../docs/query-shifts-results.md) explains the positive transfer signal, stronger full-effect baseline, and cohort limitations.

- `query-shifts-seed-42.json`, `query-shifts-seed-7.json`: full raw paired-query study packages, checkpoint hashes, and implementation provenance.
- `query-shifts-seed-42-analysis.json`, `query-shifts-seed-7-analysis.json`: deterministic analysis receipts with per-unit candidate pools, neighbors, scores, coverage, shuffle identities, and calibration geometry.
- `query-shifts-summary.json`: compact cross-seed and layer summaries.
- `query-shifts-audit.json`: independent hash and arithmetic checks, strength-matching distributions, and excluded-unit examples.

The app loads raw records and recomputes analysis locally. Reproduce with `pnpm research:query-shifts --seeds 42,7`; use `--analyze-only` to retain measurements and recompute only the analysis. The first measured runtime was roughly 7–10 seconds per checkpoint on the tested WebGPU browser; this is a device-specific observation.


## TinyStories scale study

`stories-index.json` lists three measured reference runs: 827,392, 3,227,648 and 10,739,712 parameters. Every run used seed 42 and 51,200 supervised training characters, with four raw activation maps. See [the full results](../../docs/stories-results.md) for metrics, actual generation samples, geometry coverage and limitations.

Each catalog entry pins an ordered binary archive (or its numbered parts) by SHA-256 and links its uniquely named report. Small and medium include exact latest weights, Adam state and training RNG. Large is explicitly inspection-only in the public archive: the complete 151 MB archive is preserved locally at `.local-experiments/stories-scale-v1/stories-large-complete.tissue` and can be imported through the lab. The public large archive still contains every raw captured activation, geometry diagnostics, metrics, sample and paired intervention probabilities.

The binary format starts with `TISSUE-STORY-V1\n`, a little-endian JSON-header byte count, JSON metadata with typed-array ordinals, then contiguous float32 payloads. Reference assets above 20 MiB are split into ordered parts; the client verifies the hash of their concatenation before decoding. `stories-audit.json` records an independent check of all published artifacts, reports, source hashes and the original local complete checkpoints. Re-run using `node scripts/audit-stories.mjs`; use `--full-archives` only on a machine retaining the report's original local capture paths.
