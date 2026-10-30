# Spatial zoning sensitivity in LCZ classification

This repository contains data, code, results, and figures for a GIScience study of how alternative internal zonings affect urban morphology based Local Climate Zone (LCZ) classification when the physical analysis unit remains fixed.

## Research question

How much can LCZ predictions change when the same 320 m × 320 m urban patch is represented by different internal zonings, and how can this sensitivity be reduced while preserving predictive accuracy across cities?

## Spatial design

Each So2Sat LCZ42 patch is kept fixed in extent, building content, reference label, and nominal scale. Only the internal zoning is changed.

The discrete representation divides each patch into a 4 × 4 grid and summarizes building morphology within the 16 cells. Nine zoning configurations are evaluated. A continuous spatial representation based on fixed anchors provides an alternative description of the same building set.

The analysis focuses on built LCZ classes 1–10.

## Data

LCZ labels and geolocation information come from So2Sat LCZ42 V4. Building footprints come from Overture Maps.

So2Sat LCZ42:
- https://github.com/zhu-xlab/So2Sat-LCZ42
- https://huggingface.co/datasets/zhu-xlab/So2Sat-LCZ42

## Main measure

Zoning sensitivity is measured by whether prediction correctness for the same physical patch changes across zoning configurations.

For patch i and zoning z,

C_i(z) = I(ŷ_i(z) = y_i)

and

S_i = I(max_z C_i(z) ≠ min_z C_i(z)).

The reported proportion is the mean of S_i across patches.

The analysis also reports correct to wrong transitions, prediction disagreement, accuracy, balanced accuracy, and macro F1.

## Validation across cities

Model selection is nested by city. For each outer city, the remaining cities are used to select the weighting parameter α. The selected value minimizes zoning sensitivity while limiting the largest accuracy loss observed among the inner validation cities.

The primary accuracy loss limit is 0.5 percentage points.

## Main result

For HistGradientBoosting with the 0.5 percentage point limit, all ten outer cities remain within the specified accuracy loss. zoning sensitivity decreases in all ten cities.

Mean values across cities:

- accuracy: 53.205% to 53.390%;
- balanced accuracy: 43.779% to 44.228%;
- macro F1: 34.614% to 34.853%;
- patches with changing prediction correctness: 31.605% to 27.856%;
- correct to wrong rate: 16.672% to 14.504%;
- prediction disagreement: 42.043% to 37.281%.

