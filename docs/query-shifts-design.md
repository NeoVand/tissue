# Paired-query neighborhood transfer: design v1

Locked at **2026-09-17 00:53 UTC**, before reading paired-query measurement results. Earlier full-fingerprint observations motivated this comparison, especially their association with layer membership. This is a prospective protocol for the new paired-query measurements, not an external preregistration or a claim that all research decisions were made before earlier experiments.

## Question and measurement

Does proximity in query-dependent intervention effects on calibration assignments select neighbors with similar query-dependent intervention effects on held-out assignments, beyond the same-layer, effect-strength-matched random expectation?

For each recorded checkpoint, measure 16 fixed calibration assignment groups and 16 disjoint held-out groups. Each group contains the same assignment and presentation order queried as `a`, `b`, and `c`; only the final query token changes. Capture each neuron's final-token post-ReLU activation. For each neuron, zero that unit at every token position and record lesion-minus-intact probabilities for all eight answer values. This is **held-out neighborhood transfer**: effects are measured on both splits. It is not prediction of previously unmeasured output probabilities, a repair experiment, or evidence of a semantic concept.

The model parameters, optimizer, and training RNG must be unchanged by measurement. The raw record includes checkpoint hashing, actual backend, prompt triples, intact probability mass, identical-prefix activation checks, and capture-versus-prediction checks. These checks concern numerical validity and provenance; they do not establish interpretability.

## Fixed contrasts and geometry

For every group and output coordinate, let `a`, `b`, `c` denote the measured effects under each query. Concatenate these two Helmert contrasts, group-major and then contrast-major:

```
h1 = (b − a) / sqrt(2)
h2 = (2c − a − b) / sqrt(6)
```

Apply the same transform to the single activation coordinate. Effect contrasts have 256 coordinates per neuron (16 groups × 2 × 8); activation contrasts have 32. Full effects have 384 coordinates. No further row centering is applied. A common additive query-independent response cancels. For each coordinate, `h1²+h2² = (a−m)²+(b−m)²+(c−m)²`, where `m=(a+b+c)/3`. The transform is an orthonormal coordinate system for within-group query variation; it adds no information and does not remove assignment-specific interactions with the query.

The calibration-only PCA uses L2-normalized effect contrasts for all neurons whose calibration contrast RMS exceeds the fixed floor below. Other rows are unresolved. Its diagnostics describe projection quality, not the transfer score. Its ordinary global nearest-neighbor links differ from the restricted neighbor selections used in the test. Rendering or PCA coordinates never determine the test neighborhoods.

## Shared eligible pool and fixed methods

Constants are fixed at **k = 6**, **candidate pool at most 32**, and **RMS floor = 1e−8**. RMS is `sqrt(sum(x²)/dimension)`. This absolute floor is a numerical heuristic, not a calibrated uncertainty bound; its units differ between probabilities, activations, and weights. Report strengths, exclusions, and test coverage so the heuristic is visible.

A calibration-eligible unit must exceed that floor in all four method vectors: query-effect contrasts, full effects, query-activation contrasts, and outgoing weights. Focal units and candidate units use this same requirement. For each focal unit:

1. Restrict to other calibration-eligible units in the same layer.
2. Rank candidates by the absolute difference between their natural-log **raw full-effect RMS** and the focal unit's. Resolve ties by ascending neuron ID.
3. Keep the closest 32, or every available candidate when fewer exist. If fewer than six exist, label the focal unit as having insufficient candidates; do not silently reduce k.
4. Within this identical candidate pool, each method chooses the six smallest Euclidean distances after L2 normalization of its own vector. This is equivalent to maximizing signed cosine similarity. Resolve ties by neuron ID.

Methods are **query effects** (primary), **full effects**, **query activations**, and **outgoing weights**. Full effects are a baseline on the same held-out query-contrast target, not a separately tuned outcome. Neighbor selection, eligibility, strength matching, and the PCA access calibration measurements only.

## Held-out score, coverage, and exact random expectation

The scoring target is the held-out query-effect contrast. All methods use the same focal cohort: calibration-eligible units with at least six candidates and a held-out query-effect RMS above the floor. A below-floor focal target has no cosine direction and is explicitly excluded, with its identity and strength retained. Do not impute a successful score or hide its exclusion.

For each scored focal unit and each selected neighbor, take the signed cosine of their held-out query-effect contrast vectors. A below-floor **neighbor** target contributes **zero**, and its unresolved status is counted. This declared zero-credit convention prevents each method from dropping inconvenient neighbors. It is not a mathematical cosine for a zero vector. Average over all six selected neighbors. Report each method's resolved-neighbor fraction and the exact selected IDs.

The matched random score is the average of these same cosine-or-zero values over the entire focal candidate pool. For a uniform size-six sample without replacement, the expected sample mean equals this pool average exactly; Monte Carlo draws add unnecessary noise. The per-unit primary contrast is method score minus this exact expectation. Report pool size, pool IDs, random coverage, and strengths. Arithmetic means across scored focal units produce per-layer and per-seed summaries; all methods have identical denominators. A seed summary weights units equally, rather than weighting layers equally. Empty cohorts produce null scores, not zero.

## Descriptive shuffled-identity control

Before scoring a control, deterministically permute held-out query-effect vectors among calibration-eligible units **within each layer and held-out resolved/unresolved stratum**. This preserves each unit's resolved status, the focal cohort, and every method's coverage while disrupting calibration-to-held-out unit identity. The fixed Fisher–Yates shuffle uses a seed derived from the model seed and a versioned constant. Keep calibration selections and pools unchanged, and report the shuffled score and shuffled delta against the shuffled pool expectation. Preserve the exact permutation in the analysis report. One shuffle is a diagnostic, not a permutation significance test or a population uncertainty estimate.

## Decision boundary and limitations

Report both model seeds and each layer, even when results are weak or contrary to the hypothesis. Favorable mean transfer would justify additional seeds, split choices, floor sensitivity checks, stronger matching, and a separately specified repair study. A weaker primary score than full effects or outgoing weights is a useful negative result. Do not redefine the contrast, floor, cohort, or score after inspecting results; any later analysis receives a new version and is labeled exploratory.

Sixteen assignments per split, two seeds, dependent neuron pairs, shared training history, and selection of these checkpoints limit inference. Conditioning on layer and approximate full-effect strength does not control all geometry or representational biases. Thresholded coverage and strength matching can change the population under study. No p-values, confidence intervals, semantic labels, or novelty claims are planned here.

## Literature boundary

Comparing representation geometry has a substantial prior literature; representational similarity analysis explicitly compares dissimilarity structure across measured representations. Our neuron-neighborhood comparison is a small, task-specific analysis using established geometric operations. [Kriegeskorte, Mur & Bandettini (2008)](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/neuro.06.004.2008/full).

Causal tracing uses interventions on model states to investigate their role in behavior, while ROME separately tests weight editing. Our all-position zero lesion is a different intervention, and similarity of its effects does not establish that neighbors can repair one another. [Meng et al. (2022)](https://arxiv.org/abs/2202.05262).

Causal abstraction work tests proposed alignments between neural states and high-level variables using interchange interventions. Query contrasts here do not establish such an abstraction or identify a causal variable implemented by a neuron. [Geiger et al. (2021)](https://arxiv.org/abs/2106.02997).
