# Method: zoning sensitivity under fixed spatial support

## Research question

How can prediction sensitivity to alternative zoning configurations be reduced while predictive accuracy is preserved across cities?

The analysis concerns the zoning component of the modifiable areal unit problem (MAUP). Spatial support and nominal scale remain fixed.

## Spatial support

Each observation is a 320 m × 320 m So2Sat LCZ42 patch. The LCZ label, patch boundary, and Overture Maps building set remain unchanged across all zoning configurations.

Nine internal zonings are evaluated. The canonical zoning uses cut lines at 80, 160, and 240 m along both axes. Eight alternatives shift the internal cut lines while retaining 16 cells and the same outer boundary.

## Discrete zoning representation

Four building morphology measures are calculated within each cell:

1. building count density;
2. footprint area density;
3. mean footprint area;
4. mean footprint compactness.

The 16 cells therefore produce 64 predictors.

## Continuous spatial representation

Sixteen fixed anchors are placed at the centers of the canonical 4 × 4 cells. Building contributions are weighted by Gaussian distance with sigma = 55 m.

This representation describes the same buildings without reallocating them between discrete internal cells.

## Prediction instability across zonings

For patch i and zoning configuration z,

C_i(z) = I(ŷ_i(z) = y_i).

Instability for patch i is

S_i = I(max_z C_i(z) ≠ min_z C_i(z)).

The study reports the proportion

(1 / N) Σ_i S_i.

This is the proportion of fixed patches whose prediction correctness changes across the nine zonings.

Two additional measures describe the same sensitivity:

- correct to wrong rate: the canonical zoning is correct and at least one alternative zoning is wrong;
- prediction disagreement: predicted LCZ labels are not identical across all zonings.

## Combining the two spatial representations

Let p_h(z) be the class probabilities from the discrete zoning representation and p_c the probabilities from the continuous representation.

For α in [0, 1],

p_α(z) = (1 - α) p_h(z) + α p_c.

The predicted LCZ class is the class with the largest combined probability.

## Selection by city

Validation is nested by city.

For each outer city:

1. the outer city is excluded from parameter selection;
2. the remaining cities are used in an inner leave one city out analysis;
3. α is evaluated from 0 to 1 in increments of 0.05;
4. accuracy loss relative to α = 0 is calculated separately for every inner validation city;
5. α is retained when the largest city specific accuracy loss is no greater than δ;
6. among retained values, the α with the lowest mean zoning sensitivity is selected;
7. the selected α is evaluated in the outer city.

The primary value is δ = 0.5 percentage points. A value of 1.0 percentage point is reported as a secondary analysis.

## Comparison of the two accuracy rules

The earlier analysis limited the mean accuracy loss across inner cities. The final analysis limits the largest accuracy loss observed in any inner city.

For HistGradientBoosting at δ = 0.5 percentage points:

| Selection rule | Mean α | Mean accuracy change in outer cities | Largest accuracy decrease | Cities within the 0.5 point limit | Mean zoning sensitivity | Zoning sensitivity change | Cities with lower zoning sensitivity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mean accuracy rule | 0.64 | -0.655 pp | -6.526 pp | 6/10 | 10.61% | -20.99 pp | 10/10 |
| Maximum city loss rule | 0.15 | +0.184 pp | -0.215 pp | 10/10 | 27.86% | -3.75 pp | 10/10 |

The mean accuracy rule selected much larger α values. Santiago showed the largest accuracy decrease, at 6.526 percentage points.

Using the maximum city loss rule kept all ten outer cities within the specified limit. Zoning sensitivity still decreased in every city.

## Main ten city result

HistGradientBoosting is used for the principal analysis.

With δ = 0.5 percentage points:

- mean accuracy: 53.205% to 53.390% (+0.184 pp);
- largest city specific accuracy decrease: 0.215 pp;
- mean balanced accuracy: 43.779% to 44.228% (+0.449 pp);
- mean macro F1: 34.614% to 34.853% (+0.239 pp);
- mean zoning sensitivity: 31.605% to 27.856% (-3.748 pp);
- mean correct to wrong rate: 16.672% to 14.504% (-2.168 pp);
- mean disagreement: 42.043% to 37.281% (-4.763 pp).

All ten cities remain within the 0.5 percentage point accuracy limit. Zoning sensitivity and correct to wrong rate decrease in all ten cities.

City bootstrap 95% intervals for the selected value minus α = 0 are:

- accuracy: [-0.029, +0.403] pp;
- balanced accuracy: [+0.141, +0.715] pp;
- macro F1: [+0.024, +0.476] pp;
- zoning sensitivity: [-4.997, -2.548] pp;
- correct to wrong rate: [-2.870, -1.533] pp;
- disagreement: [-6.410, -3.240] pp.

Exact sign tests across the ten cities give:

- zoning sensitivity: 10/10 decreases, p = 0.00098;
- correct to wrong rate: 10/10 decreases, p = 0.00098;
- disagreement: 10/10 decreases, p = 0.00098;
- balanced accuracy: 9/10 increases, p = 0.01074;
- accuracy: 6/10 increases, p = 0.37695.

## LCZ class results

Zoning sensitivity decreases for every built LCZ class represented in the outer city results.

Mean accuracy changes are small. Decreases occur for LCZ 1 (-0.132 pp), LCZ 2 (-0.079 pp), and LCZ 7 (-0.016 pp). Other built classes show small increases.

## Secondary accuracy limit

With δ = 1.0 percentage point, all ten cities also remain within the specified accuracy loss. Mean zoning sensitivity decreases from 31.605% to 25.661%.

## Contribution

The study contributes a GIScience formulation of zoning sensitivity in spatial prediction at fixed support. It measures changes in prediction correctness for the same physical locations and selects the spatial weighting parameter under an explicit city specific accuracy limit.

The empirical analysis shows that similar mean accuracy can coexist with substantial changes in which individual locations are classified correctly across alternative zonings.

## Related literature

- Montero, G., Caruso, G., Hilal, M., & Thomas, I. (2023). A partition free spatial clustering that preserves topology: application to built up density. Journal of Geographical Systems, 25, 5–35. DOI: 10.1007/s10109-022-00396-4.
- Ye, Y., Lei, X.-L., Li, S., Cao, Z., Niu, J., & Zhang, X. (2026). Toward Robust Urban Region Representation Learning Through Hierarchical Modeling for Modifiable Areal Unit Problem Mitigation. IEEE Transactions on Geoscience and Remote Sensing. DOI: 10.1109/TGRS.2026.3686276.
- Seong, M. H., Abbasiharofteh, M., Vos, D., & Koster, S. (2026). Smoothing the Edges: Reconceptualizing Space and Dealing With Modifiable Areal Unit Problems in (Un)Related Variety Research. Geographical Analysis, 58(1), e70025. DOI: 10.1111/gean.70025.
- Quantifying Uncertainty in Spatial Prediction for Nonstationary Spatial Processes (2026). Annals of the American Association of Geographers.
- Ye, X., & Chen, J. (2026). On the sensitivities to the modifiable areal unit problem. Big Earth Data, 10(3), 1539–1574.
