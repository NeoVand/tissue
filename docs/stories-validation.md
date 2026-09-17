# TinyStories lab validation

Validation performed September 16, 2026, with the real bundled corpus and recorded scale-study artifacts. Source code and public data are committed together; numerical study sources are independently pinned by their archive hashes.

- `pnpm check`: zero errors and warnings.
- `pnpm exec vitest run --project server`: **121 tests passed** across 11 files. Coverage includes the existing binding/query contracts and new model, geometry, geometry-report validation, and binary archive contracts.
- `pnpm exec prettier --check .` and `pnpm exec eslint .`: passed.
- Official Svelte autofixer: no issues or suggestions for every changed Svelte component.
- `pnpm build`: Cloudflare production app and all worker bundles compile. The existing application chunk-size warning remains; no individual research asset exceeds the adapter's 25 MiB limit.
- `pnpm test:e2e`: **9 production browser tests passed** in five minutes, using one worker and real browser computation.
- `node scripts/audit-query-shifts.mjs`: previous study artifacts remain consistent with their preserved numerical sources.
- `node scripts/audit-stories.mjs --full-archives`: all three published references, source hashes, reports, raw intervention summaries, and original complete local checkpoints pass. The public audit receipt records the limitations of this independent check.

The nine browser tests cover the five existing workbench/query flows plus:

1. Resuming the compact TinyStories reference, inspecting channel 511 in layer 4, recording an exact ablation, training 25 further updates, preventing simultaneous binding-model training, sampling 32 characters, switching historical maps, exporting/importing, and retaining two distinct local records after reload.
2. Opening the large public archive, addressing channel 1,535 in layer 6 (unit 9,215), architectural/functional view switching, all-channel raw capture in export, explicit inspection-only behavior, and a 390-pixel mobile viewport.
3. Training after clearing the probe context: the earlier intact baseline must be invalidated; a subsequent valid prompt and ablation must both belong to step 125.
4. The same checkpoint-isolation regression with an unsupported Unicode prompt. The error is visible, measurements and training remain preserved, and stale ablation stays disabled.

Review found and fixed three scientific/UI correctness problems before release: cross-checkpoint intact/ablated comparisons, stale neighbor edges while replacing an atlas, and an explicit save-success message after a storage failure. The first has the two real-training regressions above; neighbor acceptance now binds to the displayed atlas and request revision, and saving returns its actual outcome.

The binary round-trip contract initially exceeded Vitest's default five-second limit when its assertion recursively compared millions of numbers during parallel work. It now compares every float32 value directly, along with tensor shape and optimizer/RNG metadata, and the complete suite passes without raising the timeout.

Manual browser checks covered dark/light themes, functional/architectural layouts, exact unit addresses beyond 127, multipart medium and large reference downloads, and mobile overflow. Viewing an archived reference does not allocate a Story model worker. The saved-run list now scrolls within its panel so accumulated history does not stretch the research workspace.

See [model checks](stories-model-validation.md), [geometry checks](stories-geometry-validation.md), and [measured results](stories-results.md) for separate numerical and scientific evidence. Training-loss improvement does not establish story coherence or causal meaning of a spatial cluster.

The complete 151,025,218-byte large checkpoint was copied to durable local storage and its SHA-256 verified against the original report. Import, restoration, live token activations, and selected-unit ablation also passed through the actual interface. The resulting cross-worker numerical differences were retained in `stories-large-restoration.json` and interpreted explicitly in the results and Field journal; probabilities were close but not bit-identical across separate workers. Parameter/optimizer/RNG preservation within each recorded measurement sequence is a distinct contract.
