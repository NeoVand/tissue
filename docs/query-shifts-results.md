# Paired-query study v1: recorded results

Calibration-selected neighborhoods retain some similarity of query-dependent intervention effects on held-out assignments. **The proposed query contrast does not improve selection over full effects in these two runs.** Full-effect neighbors have the highest seed-wide mean held-out query-contrast cosine in both seeds and outperform query contrasts in all four seed/layer cells. This is a useful constraint on what to pursue next, not evidence for a new interpretability method.

The [locked design](query-shifts-design.md) was committed as `ca45f918ebc0f26224b7f7ef8d2b361c0e0876bd` before these measurements. Both models were measured at step 2,000 on WebGPU, using the fixed 16 calibration and 16 held-out assignment triples. Results, controls, and exclusions below follow v1 unchanged. The matching-quality and excluded-unit inspections are **post-result descriptive audits**.

## Held-out neighborhood transfer

Values are mean signed cosine similarities, not accuracies. Every selection method uses the same calibration-only, within-layer pool of 32 candidates and selects six. Random is the exact expected mean for uniform selection from that pool. All methods score the same held-out query-effect target and the same focal units.

| Seed | Scored units | Matched random | Query effects, primary | Full effects | Query activations | Outgoing weights |
| ---- | ------------ | -------------- | ---------------------- | ------------ | ----------------- | ---------------- |
| 42   | 248          | 0.0658         | 0.3243                 | **0.3338**   | 0.2075            | 0.3112           |
| 7    | 204          | 0.0616         | 0.2792                 | **0.2916**   | 0.2111            | 0.2343           |

The primary improvement over matched random is +0.2584 and +0.2176. The corresponding deltas after the locked within-layer held-out identity shuffle are +0.0083 and +0.0031. A single shuffle is a descriptive check of identity correspondence, not a significance test. The full-effect method exceeds the primary score by 0.0096 and 0.0124.

| Seed | Layer | Scored units | Matched random | Query effects | Full effects | Query activations | Outgoing weights |
| ---- | ----- | ------------ | -------------- | ------------- | ------------ | ----------------- | ---------------- |
| 42   | 1     | 121          | 0.0260         | 0.2789        | **0.2879**   | 0.1406            | 0.2441           |
| 42   | 2     | 127          | 0.1038         | 0.3675        | **0.3775**   | 0.2713            | 0.3752           |
| 7    | 1     | 81           | 0.0462         | 0.1545        | 0.1611       | 0.1159            | **0.1854**       |
| 7    | 2     | 123          | 0.0717         | 0.3613        | **0.3775**   | 0.2737            | 0.2665           |

Full effects exceed query effects in each cell, but outgoing weights are strongest in seed 7, layer 1. The primary is also below outgoing weights in seed 42, layer 2. The seed means weight scored units equally; the very different first-layer cohort sizes matter when comparing seeds.

## Cohort and numerical coverage

Seed 42 excludes eight units: seven in layer 1 because their final-token query-activation contrast is zero, and one in layer 2 whose activation and effect contrasts are zero. Seed 7 excludes 52: 47 such layer-1 activation-zero units and five layer-2 zero-effect units. All remaining units have 32 candidates and a resolved held-out target. Selected-neighbor and random-pool test coverage are 100%, so no zero-credit neighbor imputation enters these particular scores.

These results therefore describe the **common-method eligible cohort**, not all 256 MLP units. In particular, the query-activation baseline excludes units that still have nonzero intervention effects. The smallest scored held-out query-effect RMS is `1.31e−5` for seed 42 and `4.75e−8` for seed 7. The latter is only about 4.75 times the heuristic `1e−8` floor; numerical robustness to backend precision or a different floor has not been established.

## How close was the strength matching?

The fixed pool contains the 32 eligible same-layer units nearest in log raw full-effect RMS. It imposes no maximum strength ratio. The audit measures larger/smaller calibration RMS over directed focal/candidate pairs:

| Seed | Pool median ratio | Pool 90th percentile | Pool maximum | Primary-neighbor median | Primary-neighbor 90th percentile | Primary-neighbor maximum |
| ---- | ----------------- | -------------------- | ------------ | ----------------------- | -------------------------------- | ------------------------ |
| 42   | 1.23×             | 1.85×                | 40.82×       | 1.23×                   | 1.86×                            | 34.24×                   |
| 7    | 1.51×             | 3.22×                | 212.85×      | 1.50×                   | 2.79×                            | 212.85×                  |

Matching is often reasonably close, but the tails are loose, especially in seed 7. All methods and random share the same pool, which makes their comparison well defined; it does not fully remove effect-strength confounding. Quantiles interpolate sorted ratios at `(n−1)p`. A future protocol should specify a maximum log-strength distance and how to handle insufficient candidates before measuring new outcomes.

## A more concrete follow-up than the shape alone

All seven and 47 excluded layer-1 units have **exactly zero raw final-token activation on all 48 calibration prompts**. They are not merely constant at a nonzero value. Nevertheless, their all-position lesions have nonzero query-effect contrasts. The following examples were chosen by the largest **calibration** query-effect RMS within this excluded first-layer subset; held-out values are shown descriptively afterward. IDs and channels are zero based.

| Seed | Unit / layer-1 channel | Calibration query-effect RMS | Held-out query-effect RMS |
| ---- | ---------------------- | ---------------------------- | ------------------------- |
| 42   | 118                    | 0.002349                     | 0.002713                  |
| 42   | 20                     | 0.001921                     | 0.001336                  |
| 42   | 43                     | 0.001309                     | 0.000646                  |
| 7    | 88                     | 0.018059                     | 0.001505                  |
| 7    | 6                      | 0.012366                     | 0.001352                  |
| 7    | 43                     | 0.008122                     | 0.001105                  |

An observation at the final token cannot characterize an intervention applied at every token position. This mismatch motivates measuring where a unit is active and performing separate prefix-only and final-token-only lesions, with the same paired queries. A downstream routing explanation is plausible, but this experiment does not isolate a particular attention path or prove that mechanism. The difference between activation and intervention neighborhoods should not automatically receive a semantic label.

## Projection and provenance

The calibration-only 3D maps retain 33.99% and 42.16% of six-neighbor relations, capturing 44.81% and 50.80% of variance. They contain 255 and 251 contrast-resolved units; this is a broader set than the 248 and 204 units eligible for the common-method comparison. The transfer scores use original vectors and restricted pools, never 3D distances. The attractive map remains a lossy inspection surface.

Both captures preserve the serialized checkpoint exactly. Identical-prefix activation checks and capture-versus-prediction differences are zero. Maximum probability-mass error is `3.18e−7` or smaller. Held-out full-vocabulary answer accuracy on these 48 prompts is 47/48 and 48/48; these are learning checks, separate from the neighborhood metric.

The [independent audit](../static/experiments/query-shifts-audit.json) confirms source, locked-design, checkpoint, and measurement hashes; independently recomputes candidate pools, top-six membership, held-out and shuffled scores, and layer/seed means; and records the strength distributions and excluded examples. A few calibration selection boundaries are numerically tied within `1e−12`: query activations in 3 and 9 focals, and full effects in one seed-7 focal. The declared deterministic selection remains unchanged; no tied boundary was found for the primary.

Raw evidence and full reports: [seed 42 measurement](../static/experiments/query-shifts-seed-42.json), [seed 42 analysis](../static/experiments/query-shifts-seed-42-analysis.json), [seed 7 measurement](../static/experiments/query-shifts-seed-7.json), [seed 7 analysis](../static/experiments/query-shifts-seed-7-analysis.json), [compact summary](../static/experiments/query-shifts-summary.json).

With the development server running, recompute v1 from saved evidence without model execution:

```sh
node scripts/measure-query-shifts.mjs --analyze-only --seeds 42,7
node scripts/audit-query-shifts.mjs
```

Two seeds, 16 held-out assignments, dependent neighbor pairs, a selected cohort, and approximate strength matching do not support a population significance claim. The next study should separate token location in the intervention and predefine tighter matching and numerical sensitivity checks. V1's negative result for the proposed contrast stays in the record.
