#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
from pyproj import Transformer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--city", required=True)
    p.add_argument("--results", default="results")
    args = p.parse_args()

    frames = []
    for split in ("validation", "testing"):
        path = Path(args.results) / "samples" / split / f"{args.city}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No So2Sat sample CSVs found for {args.city}")

    df = pd.concat(frames, ignore_index=True)
    epsgs = sorted(df["epsg"].unique())
    if len(epsgs) != 1:
        raise ValueError(f"Expected one EPSG for {args.city}, got {epsgs}")
    epsg = int(epsgs[0])
    tr = Transformer.from_crs(epsg, 4326, always_xy=True)

    lons, lats = [], []
    for _, r in df.iterrows():
        for x, y in (
            (r.x_min, r.y_min), (r.x_min, r.y_max),
            (r.x_max, r.y_min), (r.x_max, r.y_max)
        ):
            lon, lat = tr.transform(float(x), float(y))
            lons.append(lon); lats.append(lat)

    margin = 0.0002
    bbox = (min(lons)-margin, min(lats)-margin, max(lons)+margin, max(lats)+margin)
    print(",".join(f"{v:.8f}" for v in bbox))

if __name__ == "__main__":
    main()
