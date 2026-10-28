#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import Transformer
from shapely import (
    area, centroid, from_wkb, is_empty, is_valid, length, make_valid,
    transform as shp_transform, box
)
from shapely.strtree import STRtree

BATCH_SIZE = 100_000
SIGMA_M = 55.0
ANCHOR_COORDS = np.array(
    [(x, y) for y in (40.0, 120.0, 200.0, 280.0)
            for x in (40.0, 120.0, 200.0, 280.0)],
    dtype=np.float64,
)

def load_samples(city):
    frames = []
    for split in ("validation", "testing"):
        p = Path("results") / "samples" / split / f"{city}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(city)
    df = pd.concat(frames, ignore_index=True)
    return df[df["lcz_index_1to17"] <= 10].copy().reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--buildings", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    samples = load_samples(args.city)
    epsg = int(samples["epsg"].iloc[0])

    patches = np.asarray([
        box(float(r.x_min), float(r.y_min), float(r.x_max), float(r.y_max))
        for _, r in samples.iterrows()
    ], dtype=object)
    tree = STRtree(patches)

    n = len(samples)
    k = len(ANCHOR_COORDS)

    w_sum = np.zeros((n, k), dtype=np.float64)
    area_sum = np.zeros((n, k), dtype=np.float64)
    compact_sum = np.zeros((n, k), dtype=np.float64)
    weighted_obj_area = np.zeros((n, k), dtype=np.float64)

    to_proj = Transformer.from_crs(4326, epsg, always_xy=True)
    pf = pq.ParquetFile(args.buildings)

    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["geometry"]):
        raw = batch.column(0).to_numpy(zero_copy_only=False)
        g = from_wkb(raw)
        good = ~is_empty(g)
        g = g[good]
        valid = is_valid(g)
        if not valid.all():
            g = g.copy()
            g[~valid] = make_valid(g[~valid])

        gp = shp_transform(g, to_proj.transform, interleaved=False)
        c = centroid(gp)
        pairs = tree.query(c, predicate="within")
        if pairs.shape[1] == 0:
            continue

        bi, pi = pairs[0], pairs[1]
        cg = c[bi]
        bg = gp[bi]

        xs = np.asarray([geom.x for geom in cg])
        ys = np.asarray([geom.y for geom in cg])
        xmin = samples.iloc[pi]["x_min"].to_numpy(float)
        ymin = samples.iloc[pi]["y_min"].to_numpy(float)
        xr = xs - xmin
        yr = ys - ymin

        ba = area(bg)
        per = length(bg)
        comp = np.where(per > 0, 4.0 * np.pi * ba / (per * per), 0.0)

        # Each building contributes to every fixed anchor according to distance.
        dx = xr[:, None] - ANCHOR_COORDS[None, :, 0]
        dy = yr[:, None] - ANCHOR_COORDS[None, :, 1]
        weights = np.exp(-(dx * dx + dy * dy) / (2.0 * SIGMA_M * SIGMA_M))

        for j in range(k):
            np.add.at(w_sum[:, j], pi, weights[:, j])
            np.add.at(area_sum[:, j], pi, weights[:, j] * ba)
            np.add.at(compact_sum[:, j], pi, weights[:, j] * comp)
            np.add.at(weighted_obj_area[:, j], pi, weights[:, j] * ba)

    records = []
    for i, r in samples.iterrows():
        feat = {}
        for j in range(k):
            ws = w_sum[i, j]
            # Weighted building count and morphology around each anchor.
            feat[f"a{j:02d}_effective_count"] = ws
            feat[f"a{j:02d}_weighted_area"] = area_sum[i, j] / 10000.0
            feat[f"a{j:02d}_mean_area"] = area_sum[i, j] / ws if ws > 0 else 0.0
            feat[f"a{j:02d}_mean_compactness"] = compact_sum[i, j] / ws if ws > 0 else 0.0
        records.append({
            "city": args.city,
            "split": r["split"],
            "sample_index": int(r["sample_index"]),
            "label": int(r["lcz_index_1to17"]),
            **feat,
        })

    out = pd.DataFrame(records)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)

if __name__ == "__main__":
    main()
