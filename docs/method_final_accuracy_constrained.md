# Method freeze: geographically constrained mitigation of zoning-induced prediction instability

## Research question

How can a supervised urban-morphology model reduce prediction instability induced by alternative zoning configurations while preserving predictive performance across unseen cities?

The analysis targets the zoning/configuration component of the modifiable areal unit problem (MAUP) under fixed outer support and fixed nominal scale.

## Fixed experimental support

- Prediction support: fixed So2Sat LCZ42 patch, 320 m × 320 m.
- Reference outcome: fixed LCZ label.
- Geographic objects: fixed Overture Maps building set within each patch.
- Outer support, object set, prediction target, and nominal scale remain unchanged.
- The internal analytical zoning varies across nine deterministic configurations.

This design isolates configuration sensitivity from scale changes.

## Baseline representation

The hard representation divides each 320 m × 320 m patch into a 4 × 4 internal grid.

For every internal cell, four building-morphology summaries are calculated:

1. building count density,
2. footprint-area density,
3. mean footprint area,
4. mean footprint compactness.

The resulting hard-zoning representation contains 64 features.

The canonical zoning uses cuts at 80, 160, and 240 m. Eight alternative zonings perturb only the internal cuts while preserving the patch boundary, number of cells, building set, prediction target, and nominal support scale.

## Stabilizing representation

A continuous building-morphology branch uses 16 spatial anchors located at the canonical 4 × 4 cell centers and Gaussian distance weighting with sigma = 55 m.

Every building contributes smoothly to nearby anchors. This branch provides a stable spatial basis against which the hard-zoning predictions can be regularized.

## Location-level zoning-instability estimand

For physical location or patch i and zoning configuration z, define prediction correctness as

C_i(z) = I(ŷ_i(z) = y_i).

A patch is zoning-unstable when its correctness changes across the admissible zoning configurations:

S_i = I(max_z C_i(z) ≠ min_z C_i(z)).

The population estimand is

R_zoning = (1 / N) Σ_i S_i.

Its empirical estimate is reported as the Partition-Induced Error Rate (PIER).

PIER therefore measures the proportion of fixed physical locations whose correct/wrong status depends on zoning configuration. It complements aggregate predictive risk because location-level correctness transitions can cancel when accuracy is averaged over the sample.

Two supporting instability measures are retained:

- correct-to-wrong rate: canonical zoning is correct and at least one alternative zoning is wrong;
- prediction disagreement: predicted class labels differ across zoning configurations.

## Probability fusion

Let p_h(z) denote the class-probability vector from the hard-zoning branch under zoning configuration z, and p_c the class-probability vector from the continuous branch.

For stabilizing weight α in [0, 1],

p_α(z) = (1 - α) p_h(z) + α p_c.

The predicted class is argmax p_α(z).

## Geographically constrained model selection

The primary procedure uses nested leave-one-city-out validation.

For every outer held-out city:

1. keep the outer city unseen during selection;
2. use the remaining cities for inner leave-one-city-out evaluation;
3. evaluate α from 0 to 1 in increments of 0.05;
4. compute the accuracy loss relative to the hard-zoning baseline separately for each inner validation city;
5. retain α values satisfying the geographic non-inferiority condition

   max_c [Accuracy_hard,c - Accuracy_α,c] ≤ δ;

6. among feasible α values, select the value minimizing mean PIER, with correct-to-wrong rate and accuracy used as tie-breakers;
7. lock α and evaluate the selected specification once on the unseen outer city.

The primary margin is δ = 0.5 percentage points. A 1.0-pp margin is retained as a secondary trade-off analysis.

## Constraint ablation

The original nested procedure constrained the mean inner-city accuracy loss. The geographic version constrains the worst inner-city loss. Their held-out behavior differs sharply under HGB with δ = 0.5 pp.

| Selection rule | Mean selected α | Mean held-out accuracy change | Worst held-out accuracy change | Cities within 0.5-pp margin | Mean PIER | PIER change | Cities with lower PIER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mean-city constraint | 0.64 | -0.655 pp | -6.526 pp | 6/10 | 10.61% | -20.99 pp | 10/10 |
| Worst-city constraint | 0.15 | +0.184 pp | -0.215 pp | 10/10 | 27.86% | -3.75 pp | 10/10 |

The mean-city rule selects substantially larger stabilizing weights and produces stronger average PIER reduction, while several unseen cities incur large predictive losses. Santiago shows the largest held-out accuracy decline at -6.526 pp.

The worst-city rule produces a smaller PIER reduction and transfers the predictive-performance margin to all ten held-out cities. This ablation establishes the role of geographic constraint choice in controlling the robustness-performance trade-off.

## Primary model and final confirmation

HistGradientBoostingClassifier (HGB) provides the strongest transferable robustness-performance balance under the geographic constraint. Extra Trees remains a sensitivity model.

For HGB with δ = 0.5 pp across ten held-out cities:

- mean accuracy: 53.205% -> 53.390% (+0.184 pp);
- worst held-out accuracy change: -0.215 pp;
- mean balanced accuracy: 43.779% -> 44.228% (+0.449 pp);
- mean macro-F1: 34.614% -> 34.853% (+0.239 pp);
- mean PIER: 31.605% -> 27.856% (-3.748 pp; 11.86% relative reduction of aggregate means);
- mean correct-to-wrong: 16.672% -> 14.504% (-2.168 pp);
- mean disagreement: 42.043% -> 37.281% (-4.763 pp);
- all 10/10 cities satisfy the 0.5-pp realized accuracy margin;
- all 10/10 cities show lower PIER;
- all 10/10 cities show lower correct-to-wrong rate.

City-cluster bootstrap 95% intervals for selected minus baseline:

- accuracy: [-0.029, +0.403] pp;
- balanced accuracy: [+0.141, +0.715] pp;
- macro-F1: [+0.024, +0.476] pp;
- PIER: [-4.997, -2.548] pp;
- correct-to-wrong: [-2.870, -1.533] pp;
- disagreement: [-6.410, -3.240] pp.

Exact city-level inference gives:

- PIER: 10/10 favorable cities, one-sided exact sign-test p = 0.00098;
- correct-to-wrong: 10/10 favorable cities, p = 0.00098;
- disagreement: 10/10 favorable cities, p = 0.00098;
- balanced accuracy: 9/10 favorable cities, p = 0.01074;
- accuracy: 6/10 favorable cities, p = 0.37695.

The inference pattern supports a systematic reduction in zoning-induced instability while predictive discrimination is preserved across held-out cities.

## Class-level check

Under δ = 0.5 pp, PIER decreases for every LCZ built class represented in the held-out data.

Mean class-level accuracy changes are small. The negative changes are approximately:

- LCZ 1: -0.132 pp,
- LCZ 2: -0.079 pp,
- LCZ 7: -0.016 pp.

Other built classes show small positive changes. The class-level pattern is consistent with the aggregate balanced-accuracy and macro-F1 results.

## Secondary margin

With δ = 1.0 pp, all 10/10 held-out cities also satisfy the realized margin.

Mean PIER falls from 31.605% to 25.661%, corresponding to an 18.81% relative reduction of aggregate means. The 0.5-pp specification remains the primary analysis because it provides the tighter geographic predictive-performance guarantee.

## Methodological contribution

The paper centers on three linked contributions:

1. a location-level estimand for zoning-induced correctness instability under fixed physical support;
2. direct selection against that instability under a prespecified predictive-performance margin;
3. a geographic worst-city constraint evaluated through nested cross-city holdout.

The empirical contribution is the demonstrated separation between aggregate predictive performance and location-level zoning stability. Mean accuracy can remain similar while many fixed locations change between correct and incorrect classification across zoning realizations.

The continuous spatial basis, probability fusion, repeated zoning perturbation, and leave-one-city-out validation provide the implementation components used to test this formulation.

## Literature boundary

Relevant adjacent work includes:

- Montero, G., Caruso, G., Hilal, M., & Thomas, I. (2023). A partition-free spatial clustering that preserves topology: application to built-up density. Journal of Geographical Systems, 25, 5-35. DOI: 10.1007/s10109-022-00396-4.
- Ye, Y., Lei, X.-L., Li, S., Cao, Z., Niu, J., & Zhang, X. (2026). Toward Robust Urban Region Representation Learning Through Hierarchical Modeling for Modifiable Areal Unit Problem Mitigation. IEEE Transactions on Geoscience and Remote Sensing. DOI: 10.1109/TGRS.2026.3686276.
- Seong, M. H., Abbasiharofteh, M., Vos, D., & Koster, S. (2026). Smoothing the Edges: Reconceptualizing Space and Dealing With Modifiable Areal Unit Problems in (Un)Related Variety Research. Geographical Analysis, 58(1), e70025. DOI: 10.1111/gean.70025.
- Quantifying Uncertainty in Spatial Prediction for Nonstationary Spatial Processes (2026): repeated zoning perturbation and robustness evaluation in spatial prediction.
- On the sensitivities to the modifiable areal unit problem (2026): contemporary treatment of scale sensitivity, zoning sensitivity, and information loss under aggregation.

## Manuscript integration

The next manuscript revision should carry this structure into four locations:

1. Introduction: define zoning-induced prediction instability as the target problem and state the three linked contributions.
2. Methods: present C_i(z), S_i, R_zoning, nested geographic selection, and the 0.5-pp primary margin.
3. Results: report the constraint ablation before the final ten-city confirmation.
4. Discussion: interpret the observed separation between aggregate predictive performance and location-level zoning stability.
