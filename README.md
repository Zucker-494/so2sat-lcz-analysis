# so2sat-lcz-analysis

Reproducible GIScience experiments on spatial partition effects in urban morphology and Local Climate Zone (LCZ) classification.

## Current stage

The repository is currently used for **data qualification and matching before model development**.

The first automated workflow downloads the public So2Sat-LCZ42 V4 validation and testing archives from Hugging Face, extracts only the LCZ labels and geolocation metadata, and writes lightweight per-city CSV files. Large Sentinel-1/2 arrays are not retained.

## Research question

How strongly can an artificial spatial partition change urban morphology / LCZ predictions when the underlying geographic objects remain unchanged, and can spatially informed modelling reduce partition-induced errors without removing genuine geographic differences?

## Data

So2Sat-LCZ42 V4:
- 17 LCZ classes
- validation: western halves of 10 held-out cities
- testing: eastern halves of the same 10 held-out cities
- geolocation metadata: city, EPSG, and TFW parameters

Source:
- https://github.com/zhu-xlab/So2Sat-LCZ42
- https://huggingface.co/datasets/zhu-xlab/So2Sat-LCZ42

## Repository outputs

After the extraction workflow finishes:

- `results/samples/validation/<city>.csv`
- `results/samples/testing/<city>.csv`
- `results/validation_class_distribution.csv`
- `results/testing_class_distribution.csv`
- `results/city_class_distribution.csv`
- `results/split_summary.csv`
- `results/audit.json`

Each per-city file contains the original sample index, LCZ class, EPSG code, projected patch centre, and WGS84 longitude/latitude.

## Important methodological rule

The LCZ patch is the fixed prediction/evaluation unit. Building and street data will later provide urban-form features. Spatial partitions will alter how contextual features are organized, while the geographic evaluation unit and reference LCZ label remain fixed.

No predictive model should be developed until the city coverage, class balance, coordinates, and matching feasibility have passed this data gate.
