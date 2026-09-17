# TinyStories: first larger-model observations

Measured on September 16, 2026 (local time), following the [design recorded at a73098c](https://github.com/NeoVand/tissue/blob/a73098c/docs/stories-design.md). All three models trained in browser workers using JaxJS WebGPU. The executing working tree was dirty; per-file SHA-256 hashes, rather than the base commit alone, identify the implementation. The [catalog](../static/experiments/stories-index.json), individual reports, binary archives, and [independent audit](../static/experiments/stories-audit.json) preserve the evidence.

## Learning at the declared budget

Seed 42; 51,200 supervised training characters for each preset. Loss is mean next-character cross-entropy on the same 2,048 fixed held-out target characters. The training-only unigram baseline is **3.047754 nats/character and 18.60% accuracy**.

| Parameters | MLP channels | Updates | Initial loss | Final loss | Final accuracy |
| ---------: | -----------: | ------: | -----------: | ---------: | -------------: |
|    827,392 |        2,048 |     100 |     4.563250 |   2.453397 |         29.35% |
|  3,227,648 |        4,096 |     100 |     4.564281 |   2.454017 |         29.39% |
| 10,739,712 |        9,216 |     200 |     4.565700 |   2.434085 |         28.42% |

All three beat this baseline. The medium model did not improve loss over the compact model at this budget. The large model has slightly lower loss but lower argmax character accuracy. These measures need not rank models identically. Different depth, width, head count, learning rate, and batch grouping prevent isolating a parameter-count effect. One seed and 2,048 evaluation characters do not establish generalization or a scaling law.

The first interval is intentionally short. These checkpoints learn character statistics; generated text does not demonstrate coherent storytelling. More training and repeated seeds are required before interpreting regions as language functions.

## What survives the three-dimensional projection

| Parameters | Resolved / captured channels | 3D variance retained | Audited neighbor retention | Scored / planned focal units |
| ---------: | ---------------------------: | -------------------: | -------------------------: | ---------------------------: |
|    827,392 |                1,915 / 2,048 |               46.72% |                     25.77% |                    119 / 128 |
|  3,227,648 |                3,819 / 4,096 |               42.84% |                     16.81% |                    113 / 128 |
| 10,739,712 |                7,707 / 9,216 |               29.05% |                     22.52% |                    111 / 128 |

Every captured channel retains its raw 128-coordinate activation fingerprint. A channel without resolved variation on these probes has no normalized direction and is omitted from the functional map; this does not establish that it is globally inactive. The architectural view still includes every channel.

Retention is the mean over scorable members of the fixed 128-ID audit, not an all-channel statistic. Original-space neighbors are exact six-neighbor scans; projected ties are included at the sixth-neighbor radius. The [geometry method](stories-geometry-validation.md) defines preprocessing, convergence, coverage, and ties. Every captured PCA fit converged to its declared residual tolerance, with no uncertain third/fourth-component boundary.

The attractive shapes lose substantial local information. Selected-neighbor links use the original fingerprints, not 3D proximity. No semantic-cluster or causal-circuit claim follows from these maps. A useful next study would ask whether neighborhoods predict held-out intervention responses after the models learn more than these early character patterns.

## Generated samples

All three use the same prompt, sampling seed 71, temperature 0.8, top-k 20, and 96 generated characters. Whitespace below is actual output; no sample was selected for fluency. Prompt:

```text
Once upon a time, there was a little
```

### 827,392 parameters

```text
y wed to scid iscot o we t hes an s wasas.

Be pane ald t te wad to s tinofowad st ouy the to w
```

### 3,227,648 parameters

```text
y w s to schede m hed we t hee s are the de litis and ple to s in and aceey pllinghe e t l t tha
```

### 10,739,712 parameters

```text
ast s tor rar lichede w and tin the wasanalad ta t thelas torashe was acacid. is d hecond. t t c
```

## Execution and retained evidence

| Parameters | Training intervals, incl. evaluation/gradient compilation | Final recorded update | Final raw atlas | Final geometry fit |
| ---------: | --------------------------------------------------------: | --------------------: | --------------: | -----------------: |
|    827,392 |                                                   21.16 s |              207.0 ms |        204.2 ms |            77.0 ms |
|  3,227,648 |                                                  100.11 s |              806.9 ms |        613.4 ms |           253.9 ms |
| 10,739,712 |                                                  335.57 s |             1393.1 ms |       1823.6 ms |           460.1 ms |

These are observed timings on the development Mac in headless Chromium 153, not controlled hardware benchmarks. Development checks and rendering also ran on the machine. Initial evaluation compilation precedes training; initialization, atlas capture, geometry, checkpoint export, and generation are additional costs. Browser user-agent strings are preserved; graphics-adapter identity was not captured.

For every preset, the runner verified exact zeroing of one prespecified channel at all prompt positions. Probing, lesion inference, and generation left every parameter, Adam moment, optimizer counter, and training RNG value unchanged. Both intact and ablated 96-character probability distributions are retained. The independent audit reconstructs their maximum probability difference and cross-checks archive/report/source identities; it does not rerun a forward pass.

Small and medium public archives include exact resumable checkpoints. The large public archive is explicitly inspection-only, retaining four full activation maps, metrics, diagnostics, samples, and intervention probabilities. Its complete resumable archive was additionally preserved at `.local-experiments/stories-scale-v1/stories-large-complete.tissue` on this machine (ignored by Git). Its hash matches the report’s `fullArchive.sha256`; the runner’s original capture files remain in `/tmp/tissue-stories-scale-v1`. Use the TinyStories import control to load the complete local archive.

Medium and large public archives are split into assets no larger than 20 MiB. The client concatenates them and verifies the checksum of the complete binary archive before import. This respects the Cloudflare adapter’s asset limit without dropping measurements.

```sh
node scripts/audit-stories.mjs
# On this development machine, also verify complete original checkpoints:
node scripts/audit-stories.mjs --full-archives
```

## Restoration exposes a numerical limit

The complete large checkpoint was imported through the actual interface, restored in a new browser worker, probed, and ablated again at the same step, prompt, and prespecified unit 4608. Held-out loss changed from 2.4340854815563944 to 2.434085411741762 (about 7e-8 nats), with unchanged argmax accuracy. The largest intact-distribution difference across workers was 2.682209e-7; the largest difference between the two lesioned distributions was 4.470348e-7. The original within-worker ablation magnitude was 3.874302e-7 and the restored measurement was 5.215406e-7.

Those quantities have comparable scale. This single cross-worker comparison does not establish a universal numerical noise floor, but it prevents interpreting this tiny selected-unit effect as a substantive causal finding. The [raw 96-way distributions and source checkpoint hash](../static/experiments/stories-large-restoration.json) are retained. UI readouts use scientific notation for tiny nonzero changes rather than rounding them to zero. Repeated intact controls should be part of any subsequent causal neighborhood study.
