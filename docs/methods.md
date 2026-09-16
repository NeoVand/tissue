# Tissue measurement notes

The first maps describe similarity across a fixed set of probes. They are hypotheses about functional organization, not physical anatomy or proof of a circuit. The probe set, model checkpoint, layer/channel identities, and preprocessing must stay identifiable in the experiment record.

## What is a point?

A point represents one measured neuron. Its fingerprint is a row whose columns refer to the same ordered contexts or intervention outputs for every neuron. Activations across different token positions must not be mixed without recording the position convention.

For an activation row $a_i$, the map uses $x_i=(a_i-\bar a_i)/\|a_i-\bar a_i\|_2$. Euclidean distance then obeys $\|x_i-x_j\|^2=2(1-r_{ij})$, where $r$ is the Pearson correlation across those probes. Positive rescaling and constant offsets disappear; anti-correlated neurons remain far apart. RMS variation before normalization is retained as a separate magnitude. Constant or numerically zero histories have unresolved direction and receive no neighborhood.

For an effect row, the model instrumentation supplies a finite vector of intervention-induced output changes. If using logits, center the vocabulary logits **within each probe** before concatenating probes, so a uniform logit shift does not masquerade as a behavioral change. The default map divides each effect row by its L2 norm, comparing directions of effect while displaying pre-normalization RMS strength separately. Setting `normalizeEffects: false` deliberately includes strength in the distance. This choice changes the scientific question and must be recorded. Geometry code does not invent or perform the interventions.

## What does 3D preserve?

PCA first centers the valid fingerprint cloud across neurons. The implementation diagonalizes its symmetric Gram matrix with a deterministic Jacobi method and uses the top three component scores. It reports both the sum and individual fractions of variance captured by those components. It does not normalize the displayed cloud's radius: distances remain in the units of the chosen fingerprint preprocessing. A renderer can apply a fixed visual scale.

`originalDistances` and `projectedDistances` are separate matrices. Original-space nearest neighbors are the basis for displayed undirected similarity edges. These edges are not directed influence or synaptic connections. A per-node and aggregate neighbor-retention score asks whether its projected nearest neighbors are still within its original-space kth-neighbor radius. Original-space distance ties at that radius all count as legitimate matches; index order only breaks selection ties. The numerical tie tolerance is $10^{-9}$ times that node's largest original distance. With no nonzero cloud variation or no neighbors, retention is undefined (`null`), not evidence of perfect preservation.

There is no guarantee that 3D retains behaviorally meaningful neighborhoods. High retained variance also does not guarantee good local neighborhoods. Both diagnostics should remain visible, and comparisons or predictions should use the original measurements as well as the picture.

## Comparing checkpoints

Coordinates are arbitrary up to an orthogonal transformation. Matching neuron identities in successive maps are aligned by orthogonal Procrustes: minimize $\|PR-Q\|_F$ over orthogonal $R$, permitting reflections. A 3×3 SVD is formed through a symmetric eigendecomposition of the cross-covariance product; rank-deficient frames receive a deterministic orthogonal completion. Shared valid neurons determine the alignment. Translation is fitted; dilation is not, so changed scale remains measurable.

This removes irrelevant frame flips and rotations. It cannot resolve a changing PCA subspace, a change in probe distribution, or a neuron-identity permutation. If the third and fourth eigenvalues are tied, the retained 3D subspace itself is not unique; `boundaryDegenerate` flags this case. Motion across checkpoints can therefore reflect projection instability as well as changed function. `alignment.rmsDisplacement` is a descriptive quantity, not an independently established learning signal.

## Numerical and experimental boundaries

- Nonfinite values and inconsistent probe lengths are rejected. Near-zero row norms (at most $10^{-12}$) are unresolved; unsupported numeric overflow is rejected.
- Invalid nodes remain at the origin only as a storage convention. A view should mark or separate them and must not read that shared position as shared function.
- Layer/channel order must identify the same neurons when passing a previous map. Aligning unrelated models by index would be meaningless.
- Nearest-neighbor graphs are not evidence of compensability. Recovery experiments must select candidate neighbors before recovery and test against matched controls on held-out examples.
- PCA and normalization are established tools. Any claim of a new interpretability method must come from independently tested predictive value, not the appearance of a shape.

Unit tests check known rank and variance, distance preservation, activation shift/scale invariance, permutation-equivariant distances, projection contraction, reflected temporal frames, deficient-rank alignment, and degenerate/no-signal data.
