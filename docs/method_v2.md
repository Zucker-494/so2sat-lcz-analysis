# Method V2 pilot: invariant anchor with zoning-tail risk

## Motivation from the real-data diagnostics

The four-city pilot shows that fixed-scale zoning perturbations produce substantial location-level instability even when average accuracy changes little. Simple multi-zoning augmentation reduces this instability only marginally, while a completely partition-invariant whole-patch representation can lose predictive information for some model families.

V2 therefore keeps both representations:
- a partition-invariant whole-patch morphology vector as a stable anchor;
- a zoning-dependent 4x4 representation as local spatial detail.

## Representation

For each fixed 320 m LCZ patch:

**Invariant anchor**
- whole-patch building count density;
- whole-patch footprint density;
- mean building footprint area;
- mean building compactness.

These descriptors do not depend on the internal 4x4 zoning.

**Local zoning representation**
- the existing 64-dimensional 4x4 morphology descriptor under each zoning configuration.

The classifier contains a global branch for invariant features and a local residual branch for zoning-dependent features. Final class logits are the sum of the global logits and a learned residual contribution.

## Training objective

For patch i with zoning views z=1...K:

L_i = L_global + L_mean + lambda_tail L_tail + lambda_anchor L_anchor

where:

- L_global: supervised cross-entropy of the invariant branch;
- L_mean: mean supervised cross-entropy over all zoning views;
- L_tail: mean of the worst three zoning-view losses for that patch (a small-sample CVaR-style zoning risk);
- L_anchor: confidence-weighted divergence from the invariant prediction.

The anchor weight is larger when the invariant branch is confident and smaller when it is uncertain. Thus the method does not force every partitioned prediction toward an unreliable global prediction.

## Pilot comparisons

Three neural strategies use the same architecture and training data:

1. dual_mean:
   invariant + local branches, mean multi-zoning supervised loss only.

2. dual_tail:
   dual_mean + zoning-tail risk.

3. dual_anchor_tail:
   dual_tail + confidence-weighted invariant anchoring.

Evaluation is leave-one-city-out across Guangzhou, Mumbai, Munich and San Francisco.

Primary outcomes:
- PIER;
- prediction disagreement;
- canonical correct -> alternative wrong rate.

Performance constraints:
- canonical accuracy;
- balanced accuracy;
- macro-F1.

V2 is considered promising only if it reduces PIER and correct->wrong risk across multiple held-out cities without a material loss in predictive performance.
