# Literature position

## Zoning sensitivity in current MAUP research

Recent MAUP studies already cover several approaches relevant to this project.

Montero et al. (2023) developed a partition free measure of built up density that preserves building topology. Seong et al. (2026) used continuous local environments to reduce dependence on administrative boundaries. Ye et al. (2026) introduced hierarchical urban region representations for MAUP mitigation. Ye and Chen (2026) formalized zoning sensitivity under boundary changes with a fixed number of regions.

Recent work on spatial prediction has also evaluated repeated zoning configurations at a fixed scale. The GeoSIMCP study compares prediction uncertainty across grid offsets and therefore provides a close precedent for repeated zoning analysis.

These studies establish that continuous spatial weighting, hierarchical representation, repeated zoning, and general prediction sensitivity are existing parts of the MAUP literature.

## Position of the present study

The present analysis focuses on prediction correctness at the same physical location.

For patch i and zoning z,

C_i(z) = I(ŷ_i(z) = y_i).

A patch contributes to PIER when its prediction is correct under at least one zoning and incorrect under at least one other zoning.

This quantity differs from changes in mean accuracy because opposite location specific transitions can cancel when results are aggregated.

The second element is parameter selection across cities. The weighting parameter α is chosen from inner validation cities while limiting the largest city specific accuracy loss. The selected value is then evaluated in an outer city that was not used for selection.

The current contribution can therefore be stated as:

> The study measures zoning sensitivity through changes in prediction correctness at fixed physical locations and selects the spatial weighting parameter under a city specific accuracy limit.

The ten city analysis provides the empirical evidence for this formulation. Under the 0.5 percentage point accuracy limit, PIER decreases in all ten outer cities and all ten remain within the specified accuracy loss.

## Closest literature

- Montero, G., Caruso, G., Hilal, M., & Thomas, I. (2023). A partition free spatial clustering that preserves topology: application to built up density. Journal of Geographical Systems, 25, 5–35. https://doi.org/10.1007/s10109-022-00396-4
- Seong, M. H., Abbasiharofteh, M., Vos, D., & Koster, S. (2026). Smoothing the Edges: Reconceptualizing Space and Dealing With Modifiable Areal Unit Problems in (Un)Related Variety Research. Geographical Analysis, 58(1), e70025. https://doi.org/10.1111/gean.70025
- Ye, Y., Lei, X.-L., Li, S., Cao, Z., Niu, J., & Zhang, X. (2026). Toward Robust Urban Region Representation Learning Through Hierarchical Modeling for Modifiable Areal Unit Problem Mitigation. IEEE Transactions on Geoscience and Remote Sensing, 64. https://doi.org/10.1109/TGRS.2026.3686276
- Ye, X., & Chen, J. (2026). On the sensitivities to the modifiable areal unit problem. Big Earth Data, 10(3), 1539–1574. https://doi.org/10.1080/20964471.2026.2692263
- Quantifying Uncertainty in Spatial Prediction for Nonstationary Spatial Processes (2026). Annals of the American Association of Geographers. https://doi.org/10.1080/24694452.2026.2639721
- Fine grained local climate zone classification using graph networks: A building centric approach. Building and Environment, 278 (2025), 112928. https://doi.org/10.1016/j.buildenv.2025.112928
