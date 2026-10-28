#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import Transformer
from shapely import area, centroid, from_wkb, is_empty, is_valid, length, make_valid, transform as shp_transform, box
from shapely.strtree import STRtree

BATCH_SIZE = 100_000
PATCH_SIZE = 320.0
FREQS = range(4)

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
    # 16 basis functions x 4 channels = 64 features.
    accum = np.zeros((n, 4, 4, 4), dtype=np.float64)

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

        u = np.clip((xs - xmin) / PATCH_SIZE, 0.0, 1.0)
        v = np.clip((ys - ymin) / PATCH_SIZE, 0.0, 1.0)

        ba = np.asarray(area(bg), dtype=np.float64)
        per = np.asarray(length(bg), dtype=np.float64)
        comp = np.where(per > 0, 4.0 * np.pi * ba / (per * per), 0.0)

        # Four object channels. Scaling keeps magnitudes numerically reasonable.
        channels = np.column_stack([
            np.ones_like(ba),
            ba / 1000.0,
            np.sqrt(np.maximum(ba, 0.0)) / 30.0,
            comp,
        ])

        cos_u = np.column_stack([np.cos(np.pi * p * u) for p in FREQS])
        cos_v = np.column_stack([np.cos(np.pi * q * v) for q in FREQS])

        for p in FREQS:
            for q in FREQS:
                phi = cos_u[:, p] * cos_v[:, q]
                for ch in range(4):
                    np.add.at(accum[:, ch, p, q], pi, channels[:, ch] * phi)

    records = []
    for i, r in samples.iterrows():
        feat = {}
        for ch in range(4):
            for p in FREQS:
                for q in FREQS:
                    feat[f"s_ch{ch}_p{p}_q{q}"] = accum[i, ch, p, q]
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
