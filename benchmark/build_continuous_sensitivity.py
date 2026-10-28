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
BASE = np.array(
    [(x, y) for y in (40.0, 120.0, 200.0, 280.0)
            for x in (40.0, 120.0, 200.0, 280.0)],
    dtype=np.float64,
)

# Fixed-scale anchor-position perturbations. All anchors remain inside the 320 m patch.
ANCHOR_VARIANTS = {
    "a0": BASE,
    "x_p20": BASE + np.array([20.0, 0.0]),
    "x_m20": BASE + np.array([-20.0, 0.0]),
    "y_p20": BASE + np.array([0.0, 20.0]),
    "y_m20": BASE + np.array([0.0, -20.0]),
    "xy_p15": BASE + np.array([15.0, 15.0]),
    "xy_m15": BASE + np.array([-15.0, -15.0]),
}

JITTER_A = np.array([
    [-12,  8], [ 7,-10], [-5, 13], [10, -6],
    [  9,-12], [-8,  6], [12,  9], [-6,-11],
    [-10, -7], [11,  5], [-7,-13], [ 8, 12],
    [  6, 10], [-9, -5], [13, -8], [-11, 7],
], dtype=np.float64)
ANCHOR_VARIANTS["jitter_a"] = BASE + JITTER_A
ANCHOR_VARIANTS["jitter_b"] = BASE - JITTER_A

BANDWIDTHS = {
    "bw35": 35.0,
    "bw45": 45.0,
    "bw55": 55.0,
    "bw70": 70.0,
    "bw90": 90.0,
}

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

def accumulators(npatch, nvariants):
    shape = (nvariants, npatch, 16)
    return (
        np.zeros(shape, dtype=np.float64),
        np.zeros(shape, dtype=np.float64),
        np.zeros(shape, dtype=np.float64),
    )

def update_variant(acc_w, acc_area, acc_comp, vi, pi, xr, yr, ba, comp, anchors, sigma):
    dx = xr[:, None] - anchors[None, :, 0]
    dy = yr[:, None] - anchors[None, :, 1]
    w = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
    for j in range(16):
        np.add.at(acc_w[vi, :, j], pi, w[:, j])
        np.add.at(acc_area[vi, :, j], pi, w[:, j] * ba)
        np.add.at(acc_comp[vi, :, j], pi, w[:, j] * comp)

def make_records(samples, family, names, wsum, asum, csum):
    recs = []
    for vi, name in enumerate(names):
        for i, r in samples.iterrows():
            feat = {}
            for j in range(16):
                ws = wsum[vi, i, j]
                feat[f"a{j:02d}_effective_count"] = ws
                feat[f"a{j:02d}_weighted_area"] = asum[vi, i, j] / 10000.0
                feat[f"a{j:02d}_mean_area"] = asum[vi, i, j] / ws if ws > 0 else 0.0
                feat[f"a{j:02d}_mean_compactness"] = csum[vi, i, j] / ws if ws > 0 else 0.0
            recs.append({
                "city": r["city"] if "city" in r.index else None,
                "split": r["split"],
                "sample_index": int(r["sample_index"]),
                "label": int(r["lcz_index_1to17"]),
                "family": family,
                "variant": name,
                **feat,
            })
    return recs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True)
    ap.add_argument("--buildings", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    samples = load_samples(args.city)
    samples["city"] = args.city
    epsg = int(samples["epsg"].iloc[0])

    patches = np.asarray([
        box(float(r.x_min), float(r.y_min), float(r.x_max), float(r.y_max))
        for _, r in samples.iterrows()
    ], dtype=object)
    tree = STRtree(patches)

    anchor_names = list(ANCHOR_VARIANTS)
    bw_names = list(BANDWIDTHS)
    aw, aa, ac = accumulators(len(samples), len(anchor_names))
    bw, ba_sum, bc = accumulators(len(samples), len(bw_names))

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

        obj_area = area(bg)
        per = length(bg)
        comp = np.where(per > 0, 4.0 * np.pi * obj_area / (per * per), 0.0)

        for vi, name in enumerate(anchor_names):
            update_variant(aw, aa, ac, vi, pi, xr, yr, obj_area, comp, ANCHOR_VARIANTS[name], 55.0)

        for vi, name in enumerate(bw_names):
            update_variant(bw, ba_sum, bc, vi, pi, xr, yr, obj_area, comp, BASE, BANDWIDTHS[name])

    records = []
    records.extend(make_records(samples, "anchor_position", anchor_names, aw, aa, ac))
    records.extend(make_records(samples, "bandwidth", bw_names, bw, ba_sum, bc))

    out = pd.DataFrame(records)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)

if __name__ == "__main__":
    main()
