# Method Design V1: Controlled Zoning Perturbation

## Scientific target

The study isolates the **zoning effect** of the Modifiable Areal Unit Problem (MAUP) in urban morphology classification.

The geographic prediction target remains fixed: each So2Sat LCZ sample is a 320 m x 320 m patch with a fixed LCZ reference label and a fixed set of building objects. Only the internal analytical partition changes.

This deliberately separates zoning sensitivity from scale sensitivity.

## Controlled zoning operation

Each 320 m patch is subdivided into 16 rectangular subregions using three vertical and three horizontal cut lines.

Canonical cuts:
- x = 80, 160, 240 m
- y = 80, 160, 240 m

Alternative zoning configurations perturb these internal cut locations while:
- preserving the outer 320 m patch boundary;
- preserving the number of subregions (16);
- preserving the underlying building set;
- preserving the prediction target and LCZ label;
- changing only the internal boundary configuration.

Eight deterministic perturbation templates are generated with a fixed random seed. Each cut is displaced by at most 20 m from its canonical position.

## Building representation

Buildings are selected by centroid membership in the fixed 320 m target patch. The same building set is then reallocated among the 16 subregions under each zoning configuration.

Per subregion:
1. building count density;
2. footprint-area density;
3. mean building footprint area;
4. mean footprint compactness.

Cell area is used for density normalization because perturbed cells can differ in area.

The 16 subregion descriptors are ordered consistently from southwest to northeast and flattened into one patch representation.

## Diagnostic before mitigation

The first real-data experiment does **not** introduce a new robustness loss.

A baseline classifier is trained on the canonical zoning only. The same fixed test samples are then evaluated under all zoning configurations.

Primary diagnostic:

**Partition-Induced Error Rate (PIER)**

A test patch is counted as partition-induced error-exposed when at least one zoning configuration yields a correct prediction and at least one other zoning configuration yields an incorrect prediction.

Because the target, buildings, label, model parameters, and inference procedure are held fixed, the zoning configuration is the only manipulated factor.

Auxiliary measures:
- canonical accuracy;
- average accuracy across zonings;
- worst-zoning accuracy;
- cross-zoning prediction disagreement;
- per-city PIER;
- per-LCZ PIER.

## Decision gate

The mitigation method is developed only if real data show a non-trivial zoning effect. As an exploratory gate, proceed if either:
- overall PIER >= 5%, or
- the canonical-to-worst-zoning accuracy loss >= 2 percentage points,
with the effect present in more than one city.

If the effect is negligible, the research direction should be reconsidered rather than forcing a robustness method onto an absent problem.

## Planned mitigation if the gate passes

The candidate method is a **zoning-robust supervised learning objective**, not a new generic region-embedding architecture.

For each training patch, predictions are produced under multiple controlled zoning configurations. The loss will combine:
- ordinary supervised loss averaged across configurations;
- a tail-risk term emphasizing the worst-performing zoning configurations;
- a selective cross-zoning consistency term.

The consistency term will be weakened for genuine morphology-transition samples so that reducing arbitrary zoning sensitivity does not erase legitimate spatial heterogeneity.

This design is intentionally distinct from hierarchical/multiscale region-representation methods: it targets fixed-scale zoning perturbations and directly optimizes prediction robustness under counterfactual boundary changes.
