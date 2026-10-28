# Method freeze: accuracy-constrained mitigation of zoning-induced prediction instability

## Research question

How can a supervised urban-morphology model reduce prediction instability caused solely by arbitrary within-patch zoning configurations, while preserving predictive performance and legitimate spatial heterogeneity?

The target is the **zoning/configuration effect** of the modifiable areal unit problem (MAUP), not the scale effect.

## Fixed experimental support

- Prediction/evaluation support: fixed So2Sat LCZ42 patch, 320 m × 320 m.
- Reference outcome: fixed LCZ label.
- Geographic objects: fixed Overture Maps building set within each patch.
- Scale and outer support are held fixed.
- Only the internal analytical representation is varied.

This design isolates configuration sensitivity from scale changes.

## Baseline representation

The hard representation divides each 320 m × 320 m patch into a 4 × 4 internal grid.

For every internal cell, four building-morphology summaries are calculated:

1. building count density,
2. footprint-area density,
3. mean footprint area,
4. mean footprint compactness.

This yields 64 features.

The canonical zoning uses cuts at 80, 160, and 240 m. Eight deterministic alternative zonings perturb only the internal cuts while preserving the patch boundary, number of cells, building set, prediction target, and nominal support scale.

## Stabilizing representation

A continuous building-morphology branch uses 16 spatial anchors located at the canonical 4 × 4 cell centers and Gaussian distance weighting with sigma = 55 m.

Every building contributes smoothly to nearby anchors. The continuous branch is used as a **stabilizing representation**, not claimed as a novel spatial weighting method.

## Accuracy-constrained fusion

Let

- p_h(z) be the class-probability vector from the hard-zoning branch under zoning configuration z;
- p_c be the class-probability vector from the continuous branch;
- alpha in [0,1] be the stabilizing weight.

The fused prediction is

p_alpha(z) = (1 - alpha) p_h(z) + alpha p_c.

The predicted class is argmax of p_alpha(z).

The methodological contribution is not linear probability fusion itself. The key procedure is selecting alpha under a cross-city predictive-performance constraint.

## Zoning-instability metrics

### Partition-Induced Error Rate (PIER)

For a fixed physical patch, PIER = 1 when at least one zoning configuration is classified correctly and at least one is classified incorrectly.

This detects location-level correctness changes that can cancel in mean accuracy.

### Correct-to-wrong rate

A patch is counted when the canonical zoning is correct but at least one alternative zoning is wrong.

### Prediction disagreement

A patch is counted when predicted class labels are not identical across zoning configurations.

Wrong-to-correct changes are retained as diagnostic information but are not interpreted as a pure performance improvement because they also indicate configuration sensitivity.

## Model-selection rule

The main procedure uses nested leave-one-city-out validation.

For every outer held-out city:

1. keep the outer city completely unseen;
2. use the remaining cities for inner leave-one-city-out evaluation;
3. evaluate candidate alpha values from 0 to 1 in increments of 0.05;
4. for each alpha, compute the accuracy loss relative to the hard-zoning baseline separately for every inner validation city;
5. retain only alpha values satisfying the **worst-city non-inferiority constraint**:

   max_c [Accuracy_hard,c - Accuracy_alpha,c] <= delta;

6. among feasible alpha values, select the value minimizing mean PIER, using correct-to-wrong rate and then accuracy as tie-breakers;
7. lock alpha and evaluate it once on the unseen outer city.

The primary margin is:

delta = 0.5 percentage points.

delta = 1.0 percentage point is a secondary trade-off analysis.

## Primary model

HistGradientBoostingClassifier (HGB) is the primary model because it provides the strongest combination of predictive discrimination and transferable stability under the robust constraint.

Extra Trees remains a sensitivity model, not the main method.

## Final ten-city confirmation

### Primary margin: delta = 0.5 pp

Across ten held-out cities:

- all 10/10 cities satisfied the realized accuracy constraint;
- worst observed held-out accuracy change: -0.215 pp;
- mean accuracy: 53.205% -> 53.390% (+0.184 pp);
- mean balanced accuracy: 43.779% -> 44.228% (+0.449 pp);
- mean macro-F1: 34.614% -> 34.853% (+0.239 pp);
- mean PIER: 31.605% -> 27.856% (-3.748 pp; 11.86% relative reduction);
- mean correct-to-wrong: 16.672% -> 14.504% (-2.168 pp; 13.00% relative reduction);
- mean disagreement: 42.043% -> 37.281% (-4.763 pp);
- PIER decreased in all 10/10 cities;
- correct-to-wrong decreased in all 10/10 cities.

City-cluster bootstrap 95% intervals for selected minus baseline:

- accuracy: [-0.029, +0.403] pp;
- balanced accuracy: [+0.141, +0.715] pp;
- macro-F1: [+0.024, +0.476] pp;
- PIER: [-4.997, -2.548] pp;
- correct-to-wrong: [-2.870, -1.533] pp;
- disagreement: [-6.410, -3.240] pp.

Thus the primary result is a stability improvement under predictive non-inferiority, not an accuracy-improvement claim.

## Class-level check

Under delta = 0.5 pp, PIER decreased for every LCZ built class represented in the held-out data.

Class-level accuracy changes were small. The only mean decreases were approximately:

- LCZ 1: -0.132 pp,
- LCZ 2: -0.079 pp,
- LCZ 7: -0.016 pp.

Other built classes increased slightly. This is consistent with the positive aggregate balanced-accuracy and macro-F1 changes.

## Secondary margin: delta = 1.0 pp

All 10/10 held-out cities also satisfied the 1.0-pp constraint.

Mean PIER fell from 31.605% to 25.661% (18.81% relative reduction), but macro-F1 was essentially unchanged. The 0.5-pp margin is therefore the preferred primary specification because it gives a cleaner predictive-performance preservation argument.

## Interpretation

The method should be described as **configuration-robust but not spatially invariant**.

It does not claim to:

- solve MAUP in general;
- remove legitimate scale effects;
- eliminate all dependence on spatial representation;
- make Gaussian weighting novel;
- provide state-of-the-art LCZ classification;
- provide an absolute partition-free representation.

The supported claim is narrower:

> Under fixed outer support and fixed analysis scale, a cross-city accuracy-constrained fusion of hard and continuous morphology representations can reduce zoning-induced prediction instability without degrading predictive discrimination beyond a prespecified margin.

## Novelty boundary

The following are established and therefore are not novelty claims:

- partition-free built-up density / topology-preserving building measures;
- kernel or continuous spatial weighting;
- multi-scale or hierarchical urban-region representations;
- generic ensemble probability fusion;
- sensitivity analysis across alternative spatial units;
- demonstrating that MAUP affects prediction.

The candidate contribution is the combination of:

1. explicit location-level zoning-instability metrics under fixed support;
2. direct optimization of zoning-induced prediction stability;
3. a worst-city predictive non-inferiority constraint;
4. nested cross-city selection to test transfer of the robustness-performance trade-off to unseen cities.

## Literature boundary to cite

- Montero, G., Caruso, G., Hilal, M., & Thomas, I. (2023). A partition-free spatial clustering that preserves topology: application to built-up density. Journal of Geographical Systems, 25, 5-35. DOI: 10.1007/s10109-022-00396-4.
- Ye, Y., Lei, X.-L., Li, S., Cao, Z., Niu, J., & Zhang, X. (2026). Toward Robust Urban Region Representation Learning Through Hierarchical Modeling for Modifiable Areal Unit Problem Mitigation. IEEE Transactions on Geoscience and Remote Sensing. DOI: 10.1109/TGRS.2026.3686276.
- Seong, M. H., Abbasiharofteh, M., Vos, D., & Koster, S. (2026). Smoothing the Edges: Reconceptualizing Space and Dealing With Modifiable Areal Unit Problems in (Un)Related Variety Research. Geographical Analysis, 58(1), e70025. DOI: 10.1111/gean.70025.
- Quantifying Uncertainty in Spatial Prediction for Nonstationary Spatial Processes (2026): use as a recent example of repeated zoning perturbation and robustness evaluation in spatial prediction.
- On the sensitivities to the modifiable areal unit problem (2026): use for the contemporary distinction between scale/zoning sensitivities and information loss under aggregation.

## Remaining work before manuscript drafting

Do not change the representation or tuning unless a new validation exposes a concrete failure.

Remaining tasks are:

1. formalize the algorithm and estimands in manuscript notation;
2. add exact city-level paired inference / sign or permutation tests as a supplement to the city-cluster bootstrap;
3. prepare figures showing the hard-to-stable Pareto trade-off and ten-city held-out effects;
4. complete a targeted novelty review around MAUP mitigation, prediction stability, representation fusion, and non-inferiority constrained model selection;
5. only then draft the paper.
