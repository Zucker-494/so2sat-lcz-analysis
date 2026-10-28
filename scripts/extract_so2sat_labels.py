#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from pyproj import Transformer

LCZ_CODES = ["1","2","3","4","5","6","7","8","9","10","A","B","C","D","E","F","G"]
LCZ_NAMES = [
    "Compact high-rise",
    "Compact midrise",
    "Compact low-rise",
    "Open high-rise",
    "Open midrise",
    "Open low-rise",
    "Lightweight low-rise",
    "Large low-rise",
    "Sparsely built",
    "Heavy industry",
    "Dense trees",
    "Scattered trees",
    "Bush, scrub",
    "Low plants",
    "Bare rock or paved",
    "Bare soil or sand",
    "Water",
]

def decode_city(values):
    out = []
    for v in values:
        if isinstance(v, bytes):
            out.append(v.decode("utf-8").strip("\x00").strip())
        else:
            out.append(str(v).strip())
    return np.asarray(out)

def patch_centres(tfw):
    # TFW convention:
    # x = A*col + B*row + C
    # y = D*col + E*row + F
    # C/F are the centre coordinates of the upper-left pixel.
    # So2Sat patches are 32 x 32 pixels, hence geometric centre = (15.5, 15.5).
    col = 15.5
    row = 15.5
    x = tfw[:,0] * col + tfw[:,2] * row + tfw[:,4]
    y = tfw[:,1] * col + tfw[:,3] * row + tfw[:,5]
    return x, y

def to_wgs84(epsg, x, y):
    lon = np.empty(len(x), dtype=float)
    lat = np.empty(len(y), dtype=float)
    for code in np.unique(epsg):
        m = epsg == code
        transformer = Transformer.from_crs(int(code), 4326, always_xy=True)
        lon[m], lat[m] = transformer.transform(x[m], y[m])
    return lon, lat

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=["validation", "testing"])
    p.add_argument("--data-h5", required=True)
    p.add_argument("--geo-h5", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    output = Path(args.output_dir)
    samples_dir = output / "samples" / args.split
    samples_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.data_h5, "r") as f:
        keys = sorted(f.keys())
        if "label" not in f:
            raise KeyError(f"Expected dataset 'label'; available datasets: {keys}")
        labels = np.asarray(f["label"])

    with h5py.File(args.geo_h5, "r") as f:
        geo_keys = sorted(f.keys())
        required = {"city", "epsg", "tfw"}
        missing = required.difference(f.keys())
        if missing:
            raise KeyError(f"Missing geo datasets {sorted(missing)}; available: {geo_keys}")
        city = decode_city(np.asarray(f["city"]))
        epsg = np.asarray(f["epsg"]).reshape(-1).astype(int)
        tfw = np.asarray(f["tfw"], dtype=float)

    n = len(city)
    if labels.shape[0] != n:
        raise ValueError(f"Label/geo length mismatch: {labels.shape[0]} vs {n}")
    if labels.ndim != 2 or labels.shape[1] != 17:
        raise ValueError(f"Expected N x 17 one-hot labels, got {labels.shape}")

    row_sums = labels.sum(axis=1)
    valid_onehot = np.isclose(row_sums, 1.0) & np.all((labels == 0) | (labels == 1), axis=1)
    class_idx0 = labels.argmax(axis=1)
    class_num = class_idx0 + 1

    x, y = patch_centres(tfw)
    lon, lat = to_wgs84(epsg, x, y)

    df = pd.DataFrame({
        "split": args.split,
        "sample_index": np.arange(n, dtype=int),
        "city": city,
        "lcz_index_1to17": class_num,
        "lcz_code": [LCZ_CODES[i] for i in class_idx0],
        "lcz_name": [LCZ_NAMES[i] for i in class_idx0],
        "label_onehot_valid": valid_onehot,
        "epsg": epsg,
        "x_center": x,
        "y_center": y,
        "longitude": lon,
        "latitude": lat,
    })

    # Keep per-city files small and directly readable through GitHub.
    for c, g in df.groupby("city", sort=True):
        g.to_csv(samples_dir / f"{c}.csv", index=False)

    dist = (
        df.groupby(["split","city","lcz_index_1to17","lcz_code","lcz_name"], as_index=False)
          .size()
          .rename(columns={"size":"n"})
    )
    dist.to_csv(output / f"{args.split}_class_distribution.csv", index=False)

    split_summary = (
        df.groupby(["split","city"], as_index=False)
          .agg(
              n=("sample_index","size"),
              n_classes=("lcz_index_1to17","nunique"),
              onehot_invalid=("label_onehot_valid", lambda x: int((~x).sum())),
              min_lon=("longitude","min"),
              max_lon=("longitude","max"),
              min_lat=("latitude","min"),
              max_lat=("latitude","max"),
          )
    )
    split_summary.to_csv(output / f"{args.split}_summary.csv", index=False)

    audit = {
        "split": args.split,
        "n": int(n),
        "data_h5_keys": keys,
        "geo_h5_keys": geo_keys,
        "label_shape": list(labels.shape),
        "onehot_invalid": int((~valid_onehot).sum()),
        "cities": sorted(df.city.unique().tolist()),
        "epsg_codes": sorted(int(v) for v in np.unique(epsg)),
    }
    with open(output / f"{args.split}_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
