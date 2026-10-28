#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import box, union_all
from shapely.strtree import STRtree

PATCH_AREA_M2 = 320.0 * 320.0

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

    buildings = gpd.read_parquet(args.buildings)
    downloaded_n = len(buildings)
    if buildings.crs is None:
        buildings = buildings.set_crs(4326)
    buildings = buildings[buildings.geometry.notna() & ~buildings.geometry.is_empty].copy()

    invalid_before = int((~buildings.geometry.is_valid).sum())
    if invalid_before:
        buildings["geometry"] = buildings.geometry.make_valid()
        buildings = buildings[buildings.geometry.notna() & ~buildings.geometry.is_empty].copy()

    buildings = buildings.to_crs(epsg)
    geoms = np.asarray(buildings.geometry.values, dtype=object)
    tree = STRtree(geoms)

    rows = []
    for _, r in samples.iterrows():
        patch = box(float(r.x_min), float(r.y_min), float(r.x_max), float(r.y_max))
        idx = tree.query(patch, predicate="intersects")

        clipped = []
        count = 0
        for i in idx:
            inter = geoms[int(i)].intersection(patch)
            if not inter.is_empty and inter.area > 0:
                clipped.append(inter)
                count += 1

        covered = float(union_all(clipped).area) if clipped else 0.0
        rows.append({
            "city": args.city,
            "split": r["split"],
            "sample_index": int(r["sample_index"]),
            "lcz_index_1to17": int(r["lcz_index_1to17"]),
            "lcz_code": r["lcz_code"],
            "built_lcz_1to10": int(int(r["lcz_index_1to17"]) <= 10),
            "building_count": int(count),
            "building_coverage_ratio": min(1.0, max(0.0, covered / PATCH_AREA_M2)),
            "zero_buildings": int(count == 0),
        })

    q = pd.DataFrame(rows)
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
        "overture_buildings_downloaded": int(downloaded_n),
        "valid_buildings_after_cleaning": int(len(buildings)),
        "invalid_buildings_before_repair": invalid_before,
        "built_zero_building_rate": float(built["zero_buildings"].mean()) if len(built) else None,
        "built_median_building_count": float(built["building_count"].median()) if len(built) else None,
        "built_median_coverage": float(built["building_coverage_ratio"].median()) if len(built) else None,
        "built_q10_coverage": float(built["building_coverage_ratio"].quantile(0.10)) if len(built) else None,
        "built_q90_coverage": float(built["building_coverage_ratio"].quantile(0.90)) if len(built) else None,
    }
    (outdir / f"{args.city}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

if __name__ == "__main__":
    main()
