#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import Transformer
from shapely import (
    area,
    box,
    from_wkb,
    intersection,
    is_empty,
    is_valid,
    make_valid,
    transform as shp_transform,
    union_all,
)
from shapely.strtree import STRtree

PATCH_AREA_M2 = 320.0 * 320.0
BATCH_SIZE = 100_000

def load_samples(city, results):
    frames = []
    for split in ("validation", "testing"):
        p = Path(results) / "samples" / split / f"{city}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(f"No sample CSVs for {city}")
    return pd.concat(frames, ignore_index=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--buildings", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--output-dir", default="results/building_qc")
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args.city, args.results)
    epsgs = sorted(samples["epsg"].unique())
    if len(epsgs) != 1:
        raise ValueError(f"Expected one EPSG for {args.city}, got {epsgs}")
    epsg = int(epsgs[0])

    # Fixed patch geometries in the city's projected CRS.
    patch_proj = np.asarray([
        box(float(r.x_min), float(r.y_min), float(r.x_max), float(r.y_max))
        for _, r in samples.iterrows()
    ], dtype=object)

    # Candidate intersections are identified in WGS84 before building geometries are projected.
    to_wgs84 = Transformer.from_crs(epsg, 4326, always_xy=True)
    to_proj = Transformer.from_crs(4326, epsg, always_xy=True)
    patch_wgs84 = shp_transform(
        patch_proj, to_wgs84.transform, interleaved=False
    )
    patch_tree = STRtree(patch_wgs84)

    building_counts = np.zeros(len(samples), dtype=np.int64)
    coverage_union = np.empty(len(samples), dtype=object)
    coverage_union[:] = None

    parquet = pq.ParquetFile(args.buildings)
    downloaded_n = int(parquet.metadata.num_rows)
    valid_after_cleaning = 0
    invalid_before = 0

    for batch_no, batch in enumerate(
        parquet.iter_batches(batch_size=BATCH_SIZE, columns=["geometry"]), start=1
    ):
        values = batch.column(0).to_numpy(zero_copy_only=False)
        geoms_wgs84 = from_wkb(values)

        nonempty = ~is_empty(geoms_wgs84)
        valid_after_cleaning += int(nonempty.sum())
        if not nonempty.all():
            geoms_wgs84 = geoms_wgs84[nonempty]

        validity = is_valid(geoms_wgs84)
        invalid_before += int((~validity).sum())
        if not validity.all():
            geoms_wgs84 = geoms_wgs84.copy()
            geoms_wgs84[~validity] = make_valid(geoms_wgs84[~validity])

        # Identify buildings intersecting the fixed LCZ patches.
        pairs = patch_tree.query(geoms_wgs84, predicate="intersects")
        if pairs.shape[1] == 0:
            continue

        building_idx = pairs[0]
        patch_idx = pairs[1]

        # Project buildings that intersect at least one LCZ patch.
        unique_building_idx, inverse = np.unique(building_idx, return_inverse=True)
        relevant_proj = shp_transform(
            geoms_wgs84[unique_building_idx],
            to_proj.transform,
            interleaved=False,
        )

        pair_buildings = relevant_proj[inverse]
        pair_patches = patch_proj[patch_idx]
        clipped = intersection(pair_buildings, pair_patches)
        clipped_area = area(clipped)
        positive = clipped_area > 0

        if not positive.any():
            continue

        patch_idx = patch_idx[positive]
        clipped = clipped[positive]

        # Count intersecting building footprints per patch.
        np.add.at(building_counts, patch_idx, 1)

        # Maintain one cumulative building union for each patch.
        for pidx in np.unique(patch_idx):
            parts = clipped[patch_idx == pidx]
            batch_union = union_all(parts)
            current = coverage_union[int(pidx)]
            if current is None:
                coverage_union[int(pidx)] = batch_union
            else:
                coverage_union[int(pidx)] = union_all([current, batch_union])

        if batch_no % 10 == 0:
            print(
                f"{args.city}: processed {batch_no * BATCH_SIZE:,} source buildings",
                flush=True,
            )

    covered_area = np.zeros(len(samples), dtype=float)
    for i, geom in enumerate(coverage_union):
        if geom is not None and not geom.is_empty:
            covered_area[i] = float(geom.area)

    q = pd.DataFrame({
        "city": args.city,
        "split": samples["split"].values,
        "sample_index": samples["sample_index"].astype(int).values,
        "lcz_index_1to17": samples["lcz_index_1to17"].astype(int).values,
        "lcz_code": samples["lcz_code"].values,
        "built_lcz_1to10": (samples["lcz_index_1to17"].astype(int).values <= 10).astype(int),
        "building_count": building_counts,
        "building_coverage_ratio": np.clip(covered_area / PATCH_AREA_M2, 0.0, 1.0),
        "zero_buildings": (building_counts == 0).astype(int),
    })
    q.to_csv(outdir / f"{args.city}_patch_building_qc.csv", index=False)

    class_summary = (
        q.groupby(["city", "lcz_index_1to17", "lcz_code"], as_index=False)
        .agg(
            n=("sample_index", "size"),
            zero_building_rate=("zero_buildings", "mean"),
            median_building_count=("building_count", "median"),
            q10_building_count=("building_count", lambda s: s.quantile(0.10)),
            q90_building_count=("building_count", lambda s: s.quantile(0.90)),
            median_coverage=("building_coverage_ratio", "median"),
            q10_coverage=("building_coverage_ratio", lambda s: s.quantile(0.10)),
            q90_coverage=("building_coverage_ratio", lambda s: s.quantile(0.90)),
        )
    )
    class_summary.to_csv(outdir / f"{args.city}_class_summary.csv", index=False)

    built = q[q["built_lcz_1to10"] == 1]
    summary = {
        "city": args.city,
        "epsg": epsg,
        "sample_n": int(len(q)),
        "built_lcz_n": int(len(built)),
        "overture_buildings_downloaded": downloaded_n,
        "valid_buildings_after_cleaning": int(valid_after_cleaning),
        "invalid_buildings_before_repair": int(invalid_before),
        "built_zero_building_rate": float(built["zero_buildings"].mean()) if len(built) else None,
        "built_median_building_count": float(built["building_count"].median()) if len(built) else None,
        "built_median_coverage": float(built["building_coverage_ratio"].median()) if len(built) else None,
        "built_q10_coverage": float(built["building_coverage_ratio"].quantile(0.10)) if len(built) else None,
        "built_q90_coverage": float(built["building_coverage_ratio"].quantile(0.90)) if len(built) else None,
        "processing_mode": "streamed parquet batches",
        "batch_size": BATCH_SIZE,
    }
    (outdir / f"{args.city}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

if __name__ == "__main__":
    main()
