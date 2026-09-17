# BPE TinyStories engine validation

The Medium and Large BPE presets both completed real WebGPU allocation, one training update, activation capture, geometry fitting, generation, and selected-unit intervention. This is an engineering smoke check, separate from the Small 4,096-update language experiment. One update does not establish useful generation or a model-size advantage.

The checked-in [machine-readable receipt](../static/experiments/token-stories-engine-smoke.json) contains exact metrics, model identities, source hashes, calibration positions, checksums, and local artifact paths. The objective, tokenizer, corpus, and evaluation design are specified in [token-stories-design.md](./token-stories-design.md).

## Actual browser execution

Run on 2026-09-17 UTC in headless Chromium with WebGPU, seed 42:

```sh
node scripts/measure-token-stories.mjs --presets medium,large --steps 1 --output /tmp/tissue-token-scale-smoke
```

No archives were published by this command. The Small 4,096-update study was training concurrently on the same device. All times below include shared-device contention; the first update also includes gradient compilation. They are diagnostics, not isolated throughput benchmarks.

| Measurement                 |             Medium |              Large |
| --------------------------- | -----------------: | -----------------: |
| Parameters                  |          5,308,416 |         13,860,864 |
| Post-ReLU MLP units         |              4,096 |              9,216 |
| Context / batch size        |            256 / 2 |            256 / 1 |
| Completed optimizer updates |                  1 |                  1 |
| Actual supervised targets   | 455 / 512 capacity | 199 / 256 capacity |
| Initial held-out CE         |           8.318573 |           8.318293 |
| Step-1 held-out CE          |           8.281241 |           8.268112 |
| Training-only unigram CE    |           5.971826 |           5.971826 |
| Fixed held-out target count |              2,768 |              2,768 |
| Initialization              |            4.581 s |           11.954 s |
| First update                |            2.954 s |            3.547 s |
| Step-1 atlas capture        |            1.837 s |            8.251 s |
| Step-1 geometry fit         |            0.446 s |            0.700 s |

Both models remain worse than the unigram loss baseline. Sampling was requested for 128 new tokens at steps 0 and 1 and after the final probe, with sampling seed 71, temperature 0.8, and top-k 40. Actual lengths were 128/30/93 for Medium and 128/53/2 for Large. Every shortened sample ended on EOS; none was cancelled. The terminating EOS is included in the token count and omitted from decoded text. Early EOS and fragmented samples after one update are not evidence of coherent story generation.

## Masking and variable-length captures

Training partitions each story into context-sized chunks without dropping targets.
If chunk `j` has `m_j` valid targets and the corpus has `T` targets, its draw
probability is `m_j / T`. The objective averages each selected chunk's masked
mean equally. Therefore
`E[loss] = sum_j (m_j/T) * (sum_t CE_jt/m_j) = sum_j,t CE_jt/T`.
This gives the corpus token objective in expectation without overweighting short
story tails. Evaluation instead sums all scored token losses and divides by the
exact number of scored targets. Padding contributes to neither objective.

An independent reconstruction of the first training batch from the corpus boundaries and seeded RNG confirmed 199 and 256 supervised targets for Medium, and 199 for Large. These total 455 and 199, respectively, and match both checkpoint and metric counts. The reconstructed RNG state also matched each checkpoint. Padding targets were not counted as training tokens.

Both presets used eight calibration prefixes with lengths `[82, 145, 152, 169, 170, 211, 146, 166]`. All eight are shorter than the 256-token context. Each prefix contributes sixteen evenly spaced **real input positions**, giving 128 fingerprint dimensions per unit. Independent archive checks reproduced the exact token IDs, decoded text, and `tokenPositions`; no selected position falls in padding.

Both step-0 and step-1 geometry fits completed and passed the independent original-space neighbor and projection audit. All 4,096 Medium and 9,216 Large units had valid nonconstant activation fingerprints. The geometry adapter therefore exercised short windows at the full Large unit count. Successful fitting is not evidence that the 3D projection preserves causal relationships.

## Intervention and model-state checks

The prompt was `Once upon a time, there was a little`. The runtime zeroed Medium unit 2048 (zero-based layer 2, channel 0) and Large unit 4608 (layer 3, channel 0) at every actual prompt position. It checked the resulting captured channel was exactly zero.

The runtime then compared every parameter and Adam moment, the optimizer step, and training RNG before and after intact/lesioned probing and generation. All comparisons passed exactly. This is a recorded runtime assertion; the independent archive audit validates saved state and arithmetic without rerunning model inference.

Both intact and lesioned outputs contain 4,096 finite probabilities with mass within `5e-7` of one. The maximum probability changes were `4.202593117952347e-8` for Medium and `6.51925802230835e-9` for Large. These very small floating-point differences have no repeated-forward numerical-noise calibration and do not support a claim about the importance of either unit.

## Evidence and independent audits

Full weights, both Adam moments, raw activations, geometry, samples, and intervention probabilities remain in:

| Archive                                                              |       Bytes | SHA-256                                                            |
| -------------------------------------------------------------------- | ----------: | ------------------------------------------------------------------ |
| `/tmp/tissue-token-scale-smoke/token-stories-medium-complete.tissue` |  68,814,960 | `3bdefbd45aa65d306f94c12a09472c33f79c9c47eef57a693950ae7e64f0dfaa` |
| `/tmp/tissue-token-scale-smoke/token-stories-large-complete.tissue`  | 177,589,904 | `c6a0459b17975cc2e9a4723d91bcc0be9f49ffd99d16a2be02f38d4d0a865217` |

Step-0 and step-1 archives, raw reports, and independent audit outputs are in the same directory. Their exact hashes are in the checked-in receipt. These local temporary files are not downloadable public reference models.

Both of these commands passed:

```sh
node scripts/audit-token-stories.mjs --file /tmp/tissue-token-scale-smoke/token-stories-medium-complete.tissue --report /tmp/tissue-token-scale-smoke/token-stories-medium-report.json --output /tmp/tissue-token-scale-smoke/token-stories-medium-audit.json
node scripts/audit-token-stories.mjs --file /tmp/tissue-token-scale-smoke/token-stories-large-complete.tissue --report /tmp/tissue-token-scale-smoke/token-stories-large-report.json --output /tmp/tissue-token-scale-smoke/token-stories-large-audit.json
```

The audit independently parses the packed archives and checks corpus/tokenizer/source/report hashes, checkpoint tensor shapes, fixed held-out windows and unigram loss, raw activation normalization, original-space focal neighbors, tie-inclusive 3D retention, BPE prompt encoding, sample decoding, and intervention arithmetic. It does not repeat GPU forwards or PCA fitting. The runner also confirmed its hashed implementation sources remained unchanged throughout both runs.

## Numerical unit-test coverage

The five tests in `src/lib/token-stories/model.spec.ts` passed before the source freeze. They cover complete within-story target partitioning and target-weighted sampling; prompt truncation and unsupported text; masked per-window cross-entropy against manual arithmetic; capture/inference parity, causal prefix invariance, exact channel ablation and tensor order; and learning plus exact checkpoint/Adam/RNG resumption on a tiny diagnostic architecture. Controlled generation verifies BOS is never sampled and EOS terminates generation correctly. Those tests and these larger WebGPU smoke checks exercise different parts of the engine; neither establishes broad language competence or generalization across devices and seeds.

The complete final smoke archives, reports and independent audit receipts are also
retained under the ignored local directory `.local-experiments/token-stories-scale-smoke/`;
the public receipt lists their exact byte counts and SHA-256 hashes.
