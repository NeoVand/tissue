# What the recorded neighborhoods contain

These descriptive comparisons use the latest raw atlases in the two recorded reference runs: seeds **42** and **7**, both at **2,000 updates**. They use the same 16 calibration prompts and six nearest neighbors in the original normalized fingerprint space. Their corresponding evaluation accuracy was 94/96 and 96/96 on the fixed held-out prompt set.

The report also includes the available earlier raw captures: step 500 for both seeds and step 1,500 for seed 7. These are observations from two training trajectories, not independent replications at each checkpoint.

The machine-readable result is [fingerprint-comparison.json](../static/experiments/fingerprint-comparison.json). It preserves source and implementation SHA-256 hashes, exact counts, unrounded fractions, valid-neuron pools, PCA diagnostics, and method definitions.

## The picture loses substantial neighborhood information

| Seed | Map          | Valid neurons | Numerical PCA rank | Variance in 3D | Six-neighbor retention |
| ---- | ------------ | ------------- | ------------------ | -------------- | ---------------------- |
| 42   | Activation   | 256           | 180                | 35.82%         | 38.35%                 |
| 42   | Intervention | 254           | 96                 | 60.97%         | 43.77%                 |
| 7    | Activation   | 256           | 180                | 30.21%         | 28.06%                 |
| 7    | Intervention | 250           | 104                | 54.63%         | 45.60%                 |

Fewer than half the projected nearest neighbors remain original-space neighbors in each of these four maps. The 3D view is therefore a lossy overview. Selection and neighborhood comparisons should retain access to the original measurements. Higher captured variance also does not guarantee high local-neighborhood retention.

Rank is the numerical rank under the tolerance in `geometry.ts`, not a count of concepts. A neuron with an unresolved effect direction is excluded from that map rather than placed into the neighbor pool as a zero vector.

## Activation neighborhoods strongly track layer membership

| Seed | Map          | Same-layer neighbor selections | Observed share | Uniform-neighbor expectation |
| ---- | ------------ | ------------------------------ | -------------- | ---------------------------- |
| 42   | Activation   | 1,459 / 1,536                  | 94.99%         | 49.80%                       |
| 42   | Intervention | 1,053 / 1,524                  | 69.09%         | 49.81%                       |
| 7    | Activation   | 1,374 / 1,536                  | 89.45%         | 49.80%                       |
| 7    | Intervention | 1,031 / 1,500                  | 68.73%         | 49.83%                       |

These are **directed neighbor selections**, before the renderer merges reciprocal links. For each focal neuron, the baseline holds its identity and valid pool fixed and draws neighbors uniformly from all other valid units. Its expected same-layer fraction is `(valid units in its layer − 1) / (all valid units − 1)`.

Much of the activation neighborhood structure follows the architecture's layer labels. Intervention neighborhoods mix layers more, although their same-layer shares also exceed this simple baseline. Visible separation should not, by itself, be described as a discovery of semantic regions.

## The two maps overlap, but organize most neighbors differently

| Seed | Focal units valid in both maps | Shared neighbor selections | Observed overlap | Independent-uniform expectation |
| ---- | ------------------------------ | -------------------------- | ---------------- | ------------------------------- |
| 42   | 254                            | 282 / 1,524                | 18.50%           | 2.35%                           |
| 7    | 250                            | 302 / 1,500                | 20.13%           | 2.35%                           |

For each common focal unit, intersect its activation and intervention top-six neighbor sets. Divide the total intersection count by the number of activation-neighbor selections across those focal units. Each map retains its own valid candidate pool. The null draws the two sets independently from those respective pools; the JSON records its exact formula and expected counts.

The organizations differ substantially, but their overlap also exceeds the independent-uniform expectation. The maps are not identical, and their shared neighborhoods are more numerous than this simple uniform reference predicts. This is not a test of statistical independence after accounting for shared layer or feature biases. It motivates inspecting which relationships agree and disagree, without establishing that either map predicts repair or identifies a human-interpretable concept.

## What changed at the available earlier checkpoints

| Seed | Updates | Held-out accuracy | Activation same-layer share | Intervention same-layer share | Activation/intervention overlap |
| ---- | ------- | ----------------- | --------------------------- | ----------------------------- | ------------------------------- |
| 42   | 500     | 32.29%            | 76.04%                      | 77.78%                        | 28.86%                          |
| 42   | 2,000   | 97.92%            | 94.99%                      | 69.09%                        | 18.50%                          |
| 7    | 500     | 29.17%            | 86.33%                      | 84.62%                        | 28.77%                          |
| 7    | 1,500   | 32.29%            | 85.48%                      | 74.56%                        | 26.70%                          |
| 7    | 2,000   | 100.00%           | 89.45%                      | 68.73%                        | 20.13%                          |

Layer association is already substantial at step 500. From step 500 to step 2,000, activation neighborhoods become more concentrated within layers in both seeds, intervention neighborhoods become less concentrated, and the two maps' overlap declines. This accompanies a rise in held-out accuracy. The sparse checkpoints do not locate the transition or establish that geometric changes cause improved performance. Valid intervention pools also change over time; the JSON records them and recomputes the uniform baselines for each capture.

## Reproduce

With Node 24 or newer, from the repository root:

```sh
node scripts/compare-fingerprints.mjs
```

Optional `--input-dir` and `--output` arguments choose another directory containing the two named reference packages and three earlier raw captures, or another output path. The script performs no training or inference. It explicitly requests `k = 6`, recomputes activation and effect maps using the lab's geometry implementation, and checks that projected-neighbor retention and explained variance match the corresponding saved maps within `1e-10`. Orthogonal temporal alignment is unnecessary for these distance-based comparisons. Missing effect fingerprints are represented as unavailable; all five checked-in captures currently contain them.

Activation rows are centered across probes and L2 normalized. Intervention rows contain lesion-minus-intact answer probabilities and are L2 normalized without centering. Original-space distances are Euclidean. Exact selection ties break by neuron ID. Projection retention accepts original-space distance ties using the lab's relative numerical tolerance; it is not simply an index-set overlap in tied cases.

These established geometric tools and uniform baselines provide descriptive controls. Selected checkpoints from two seeds, dependent neighbor relations, and unmatched activation strengths do not support a significance claim, a novelty claim, or a causal interpretation of the shape.
