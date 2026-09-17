# Tissue

A browser laboratory for discovering useful spatial descriptions of small language models. Real training, measured geometry, interventions, and a persistent research journal share one interface.

## Run

```sh
pnpm install
pnpm dev
```

Open the local URL. WebGPU is preferred; initialization validates numerical readback and can fall back to WASM. The first compilation takes a few seconds. **Explore that specimen** loads a measured, trained reference without a training wait. The source model runs entirely in your browser.

## What is here

- A 25,920-parameter causal transformer: two layers, width 32, four heads, 256 MLP neurons.
- A controlled next-token task: `a=3;b=7;c=2;?b → 7`. Answer-only cross-entropy. Whole assignment mappings are disjoint across training, calibration, and evaluation.
- Activation and exact zero-ablation fingerprints, deterministic 3D PCA, temporal Procrustes alignment, original-space neighbor links, and projection-quality measurements.
- A dense dark-default workbench with persistent light mode and HugeIcons. Functional and architectural layouts share the same stable neuron identities.
- Linked model anatomy, token-by-neuron activation matrix, exact weight slices, token responses, original-space neighbor distances, and selected-neuron interventions.
- Instanced shader nodes and similarity edges interpolate measured checkpoint positions while preserving the camera. Intermediate frames are display interpolation, not measurements.
- Development checkpoints and four-arm constrained repair pilots.
- IndexedDB experiment history; JSON export/import includes model, Adam state, RNG, metrics, snapshots, notes, latest raw fingerprints, and repair receipts. Reload resumes the active specimen.
- A curated research notebook that records findings and limitations alongside measured reference runs.

The visualization does not impose a spatial training objective or a brain-shaped layout. Size encodes compressed activation (or intervention strength); color identifies the layer; links mean fingerprint similarity, not causal connections. Projection does lose information—inspect neighbor retention before interpreting a cluster.

## Evidence, including the unhelpful result

Seed 42 was near the 33.3% random input-copy baseline at 500 steps and reached 97.9% on 96 fixed held-out prompts at 2,000 steps. Seed 7 independently exhibited a similar late transition. These are two small-model runs, not a general training guarantee.

Across both final checkpoints, 89–95% of activation-neighbor selections stay within a layer, versus about 69% for intervention neighbors. The two maps share 19–20% of nearest-neighbor selections (the independent-uniform expectation is 2.35%). This descriptive comparison exposes a layer confound; it does not establish a new interpretability method. The in-app Findings tab and [reproducible comparison](docs/fingerprint-comparison.md) preserve the full counts, earlier checkpoints, null definitions, and source hashes.

The first four-arm repair pilot **did not favor functional neighborhoods**. The lesion caused little damage, and unlesioned fine-tuning controls were absent. The next experiments are recorded in the lab, not hidden behind the attractive geometry.

Measured artifacts and provenance live in [`static/experiments`](static/experiments/README.md). The method is described in [`docs/methods.md`](docs/methods.md).

## Paired-query study

Open **Query shifts** to measure the current checkpoint or inspect recorded results for both trained seeds. Three prompts share an identical assignment and order; only the queried variable changes. Every MLP unit is silenced at all positions. Calibration query-effect contrasts define the map and neighborhood selections; disjoint held-out assignments supply the evaluation target.

The [prospectively recorded protocol](docs/query-shifts-design.md) fixes six neighbors, a shared same-layer pool of 32 units matched by calibration full-effect strength, and four comparison methods. Query-effect neighborhoods transfer above the matched random expectation, but full-effect neighborhoods score slightly better in both seeds. See [the results](docs/query-shifts-results.md) for cohort exclusions, per-layer results, matching quality, and limitations.

Completed studies retain raw probes, exact checkpoint identity, and numerical checks in their own IndexedDB archive. The Field journal links to them. Study JSON export/import preserves raw evidence and provenance; analysis is recomputed in a worker when opened. The model, Adam state, and training RNG remain unchanged. Reference studies are clearly distinguished from the current resident model.

```sh
# Keep pnpm dev running; no retraining is required.
pnpm research:query-shifts --seeds 42,7
# Recompute analysis from preserved raw records without running a model.
pnpm research:query-shifts --seeds 42,7 --analyze-only
```

## Feedback loops

```sh
pnpm check                         # TypeScript, Svelte, and generated runtime types
pnpm exec vitest run --project server # numerical, data, and record contracts
pnpm lint                          # application formatting and lint
pnpm build                         # production worker and app build
pnpm test:e2e                      # production browser flows; install Chromium first if needed
```

Recompute the five-capture geometric comparison without retraining (Node 24+):

```sh
node scripts/compare-fingerprints.mjs
```

The slower research reproduction is separate from routine checks:

```sh
# Keep pnpm dev running in another terminal.
pnpm research:verify --output /tmp/tissue-study --seed 42 --checkpoints 500,2000
```

It saves every requested stage before assessing the final learning threshold, including unsuccessful runs. Results contain source hashes, numerical backend, complete checkpoints, calibration measurements, and evaluation metrics. `--seed 7`, `--backend wasm`, and `--skip-repair` support controlled comparisons.

The first browser run exposed silently incorrect synchronous WebGPU readbacks. Production uses asynchronous readback with known-answer arithmetic and probability-mass checks. Other numerical contracts check capture parity, actual ablation, exact continuation, frozen repair weights, PCA behavior, and corrupt-record rejection.

## Where to work

- `src/lib/lab/model/`: task, JaxJS transformer, optimizer, training, interventions, repair.
- `src/lib/lab/protocol.ts`: versioned worker and model data contracts.
- `src/lib/lab/geometry.ts`: measurable fingerprint geometry and diagnostics.
- `src/lib/lab/geometry-worker.ts`: analysis off the UI thread.
- `src/lib/lab/journal.ts`: validated durable experiment records.
- `src/lib/lab/notebook.ts`: the research narrative, including negative findings.
- `src/lib/lab/model-inspection.ts`: neuron addresses and exact checkpoint tensor slices.
- `src/lib/components/`: linked research views, findings, and journal.
- `src/lib/scene/neural-field.ts`: Three.js scene, stable selection, and camera controls.
- `src/lib/scene/field-shaders.ts`: instanced node and edge interpolation shaders.

The original Jaxverse and Pattern projects supplied the browser-training patterns; the JaxJS skill supplied verified array ownership and optimizer conventions. New experiments should preserve these contracts and add evidence to the lab.
