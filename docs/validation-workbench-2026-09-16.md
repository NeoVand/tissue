# Workbench revision validation

Recorded on 2026-09-16 for version 0.2.0. This revision connects the spatial view to model anatomy and exact checkpoint tensors, replaces the editorial layout with linked instruments, and adds a reproducible comparison of the recorded fingerprints. It does not establish a new interpretability method.

## Automated checks

- TypeScript and Svelte diagnostics: zero errors or warnings.
- Official Svelte autofixer: no outstanding issues or suggestions in changed components.
- Prettier and ESLint: passed.
- Numerical, geometry, journal, and model-inspection tests: 70 passed.
- Production browser tests: three passed, using real model operations.
- Production application and workers: built successfully. The existing bundle-size advisory remains; shader draw-call efficiency does not eliminate JavaScript download cost.

The four new model-inspection tests check every tensor descriptor against the actual serialized JaxJS parameter tree, extract incoming columns and outgoing rows at both layer boundaries, preserve activation checkpoint and lesion provenance, and reject invalid channels and nonfinite values.

The browser flows now also verify dark mode by default, persistent light mode, neuron 172 mapping to layer 1 / channel 44 in zero-based tensor indexing, keyboard heatmap selection, architectural layout switching, and the recorded Findings panel. Continuing training after selecting Intervention switches back to Activation when the next capture has no intervention fingerprints. The graph labels its atlas step independently of current training metrics.

Existing browser coverage still exercises real ablation, a four-arm repair, historical inspection, notes, JSON export, reload, training and pause, mobile navigation, and recovery from an injected capture failure. Historical snapshots expose their recorded probes without presenting the latest checkpoint's weights as historical weights.

## Renderer and visual inspection

The reference scene contains 256 MLP units and renders in three draw calls, including shader nodes, shader similarity edges, and the guide grid. This is a small-scene observation, not a scalability or frames-per-second benchmark.

Browser inspection exercised interrupted checkpoint transitions, reordered stable neuron IDs, selection during transitions, layer filtering, camera preservation, dark and light themes, and reduced motion. Display interpolation is explicitly separated from measured checkpoints. Similarity edges are labeled as noncausal.

Desktop inspection at 1512 × 982 fits the workbench within the viewport, including the complete activation matrix and its raw-value readout. The inspector scrolls independently. The architecture view shows the same channels in layer/index order. Mobile inspection at 390 pixels and the production mobile flow confirm no horizontal document overflow.

## Scientific checks

The [comparison report](fingerprint-comparison.md) was regenerated from five saved raw captures across two seeds. The script checks that recomputed PCA retention and variance match the saved diagnostics within 1e-10, and preserves hashes of its inputs and implementation. All eight hashes and five capture comparisons were verified.

Temporary activation-only copies of all five captures were also passed through the script: activation diagnostics were preserved and missing intervention/overlap measurements returned null. Contradictory intervention metadata was rejected.

The recorded independent-uniform expectations are descriptive baselines. They do not test statistical independence after shared layer or feature biases. Two trajectories, sparse checkpoints, changing valid pools, and unmatched activation/effect strengths limit inference. The initial repair pilot remains inconclusive.
