# Scalable story geometry validation

These checks validate the numerical pipeline and renderer. They do **not** demonstrate language learning, interpretable features, or throughput of the transformer. The recorded story-training experiments provide separate model evidence.

## Measurement and numerical contract

Each unit has a fixed, ordered calibration fingerprint: eight windows × sixteen token positions. A row is mean-centered and divided by its L2 norm. Its pre-normalization population standard deviation is retained as `magnitudes`. Rows at or below the declared RMS floor of `1e-8` are unresolved; this is a numerical heuristic, not a biological or statistical activity threshold. Negative correlation remains negative similarity.

The implementation centers these normalized rows across resolved units and forms a feature-space cross-product matrix. For 128 features, this matrix has 16,384 entries regardless of the number of units. A deterministic eight-vector block power iteration with a small Rayleigh–Ritz solve estimates the leading four directions; three become coordinates and the fourth diagnoses a nearly tied projection boundary. Maximum iterations are 256 and the relative eigenpair-residual tolerance is `1e-8`. This is approximate PCA, with convergence and residual reported, not an exact numerical rank estimate. `pca.totalVariance` is the trace of the **unnormalized** centered cross-product matrix: a sum of squared deviations, not variance per unit. Captured-variance fractions and relative residuals do not depend on that scale convention.

The worker retains an O(NP) normalized index. Selected-neuron neighbors use an exact scan of all eligible rows, optionally restricted to a layer; Euclidean distances are in the original normalized feature space and signed cosine is `1 − distance²/2`. No N × N distance matrix is allocated. Saved atlases can load that index without recomputing saved coordinates.

Projection-neighbor retention is audited at 128 unit IDs drawn without replacement by a fixed, value-independent schedule (seed `0x51a7c0de`). Unresolved sampled units are not replaced. The report carries planned, resolved, and scored counts, plus each focal's actual neighbor count. The score checks whether each exact original-space neighbor lies within the projected k-neighbor radius; boundary ties receive credit within `max(1e-12, radius × 1e-9)`. A cloud with no directional variance has no retention score. This is a tie-inclusive diagnostic over the reported sample, not exact whole-model retention or evidence of causal connections. Dense projection collisions can make tie-inclusive scores less discriminating.

Temporal alignment uses the existing orthogonal Procrustes implementation, including reflections and no dilation, only for matching model/corpus/probe identities. It removes arbitrary frame orientation; it does not prove that a changing 3D neighborhood reflects a changing function, particularly near a tied third/fourth component.

## Automated numerical checks

Run:

```sh
pnpm exec vitest run --project server src/lib/stories/geometry.spec.ts
```

All twelve tests passed locally. The suite checks:

- Agreement with the existing independent neuron-Gram PCA on a small cloud, including captured variance and projected pairwise distances.
- Mean/positive-scale invariance, explicit magnitude retention, signed anticorrelation, and constant-row exclusion.
- Exact nearest neighbors against a brute-force calculation, layer restrictions, and deterministic distance ties.
- A fixed audit schedule with twenty unresolved sampled rows; coverage is reported without replacements.
- Exact retention for a cloud contained in three dimensions; finite zero/identical-direction cases; tied third/fourth eigenvalues; visible non-convergence when stopped after one iteration.
- Determinism, permutation-equivariant distances for a separated PCA subspace, identity-gated temporal alignment, and malformed-input rejection.
- A 9,216 × 128 fit with a compact result and a 9,437,184-byte retained Float64 fingerprint index; at most 768 sampled edges.
- Raw-buffer ownership, pending-request cancellation, and stale worker-generation rejection.

The final run took about 0.68 seconds including Vitest startup; the tests themselves took about 0.52 seconds. These timings are local observations, not performance guarantees. ESLint passed for the geometry and shared renderer files. The Svelte autofixer reported no issues or suggestions for the modified `NeuralField.svelte`.

## Browser worker and renderer smoke check

A separate headless Chromium check used the real Vite-loaded worker/client and Three.js renderer. It generated 9,216 rows of 128 Float32 values using:

```text
x[i,d] = Float32(3 + sin((i + 0.3)(d + 1.7)) + cos(1.9i + d))
```

The fixture used six layers of 1,536 units and did not execute a language model. The check built geometry in the worker, queried unit 9,215 within layer six, rendered all units, changed the layer filter, verified camera preservation, then destroyed the scene and canceled a pending fit.

| Observed check                                                | Result                                    |
| ------------------------------------------------------------- | ----------------------------------------- |
| Worker fit, including startup and transfer                    | 408.7 ms                                  |
| Exact selected-neighbor query, restricted to 1,536-unit layer | 0.7 ms                                    |
| Resolved units / scored audit focals                          | 9,216 / 128                               |
| PCA iterations / relative residual                            | 161 / 9.49 × 10⁻⁹                         |
| Captured variance / sampled retention                         | 0.51406 / 0.17708                         |
| Instanced units in scene                                      | 9,216                                     |
| Renderer draw calls                                           | 3                                         |
| Selected address                                              | global unit 9,215; layer 6; channel 1,535 |
| Camera unchanged after update/filter                          | Yes                                       |
| Caller-owned raw fingerprint array intact after transfer      | Yes                                       |
| Pending fit rejected when worker destroyed                    | Yes                                       |
| Renderer errors                                               | None                                      |

The reported render submission measurement was 0.5 ms during the filtered scene's transition. It measures CPU submission work only; it is neither GPU execution time nor an FPS measurement. Pointer hover tests are coalesced to one exact O(N) ray scan per animation frame. Palette and channel labels support all six current layers. No conclusion about the quality of an actual trained model follows from these synthetic-fixture numbers.
