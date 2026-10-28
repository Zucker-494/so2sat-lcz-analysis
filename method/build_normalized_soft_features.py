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
SIGMA_M = 55.0
BASE = np.array(
    [(x, y) for y in (40.0, 120.0, 200.0, 280.0)
            for x in (40.0, 120.0, 200.0, 280.0)],
    dtype=np.float64,
)
JITTER = np.array([
    [-12,  8], [ 7,-10], [-5, 13], [10, -6],
    [  9,-12], [-8,  6], [12,  9], [-6,-11],
    [-10, -7], [11,  5], [-7,-13], [ 8, 12],
    [  6, 10], [-9, -5], [13, -8], [-11, 7],
], dtype=np.float64)

ANCHORS = {
    "a0": BASE,
    "jitter_a": BASE + JITTER,
    "jitter_b": BASE - JITTER,
    "x_m20": BASE + np.array([-20.0, 0.0]),
    "x_p20": BASE + np.array([20.0, 0.0]),
    "xy_m15": BASE + np.array([-15.0, -15.0]),
    "xy_p15": BASE + np.array([15.0, 15.0]),
    "y_m20": BASE + np.array([0.0, -20.0]),
    "y_p20": BASE + np.array([0.0, 20.0]),
}

def load_samples(city):
    frames=[]
    for split in ("validation","testing"):
        p=Path("results")/"samples"/split/f"{city}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(city)
    df=pd.concat(frames,ignore_index=True)
    return df[df["lcz_index_1to17"]<=10].copy().reset_index(drop=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--city",required=True)
    ap.add_argument("--buildings",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    samples=load_samples(args.city)
    epsg=int(samples["epsg"].iloc[0])
    patches=np.asarray([
        box(float(r.x_min),float(r.y_min),float(r.x_max),float(r.y_max))
        for _,r in samples.iterrows()
    ],dtype=object)
    tree=STRtree(patches)

    names=list(ANCHORS)
    nv=len(names); npatch=len(samples); na=16
    count=np.zeros((nv,npatch,na),dtype=np.float64)
    area_sum=np.zeros((nv,npatch,na),dtype=np.float64)
    comp_sum=np.zeros((nv,npatch,na),dtype=np.float64)

    to_proj=Transformer.from_crs(4326,epsg,always_xy=True)
    pf=pq.ParquetFile(args.buildings)

    for batch in pf.iter_batches(batch_size=BATCH_SIZE,columns=["geometry"]):
        raw=batch.column(0).to_numpy(zero_copy_only=False)
        g=from_wkb(raw)
        good=~is_empty(g)
        g=g[good]
        valid=is_valid(g)
        if not valid.all():
            g=g.copy()
            g[~valid]=make_valid(g[~valid])

        gp=shp_transform(g,to_proj.transform,interleaved=False)
        c=centroid(gp)
        pairs=tree.query(c,predicate="within")
        if pairs.shape[1]==0:
            continue

        bi,pi=pairs[0],pairs[1]
        cg=c[bi]; bg=gp[bi]
        xs=np.asarray([geom.x for geom in cg])
        ys=np.asarray([geom.y for geom in cg])
        xmin=samples.iloc[pi]["x_min"].to_numpy(float)
        ymin=samples.iloc[pi]["y_min"].to_numpy(float)
        xr=xs-xmin; yr=ys-ymin

        ba=np.asarray(area(bg),dtype=np.float64)
        per=np.asarray(length(bg),dtype=np.float64)
        comp=np.where(per>0,4.0*np.pi*ba/(per*per),0.0)

        for vi,name in enumerate(names):
            anchors=ANCHORS[name]
            dx=xr[:,None]-anchors[None,:,0]
            dy=yr[:,None]-anchors[None,:,1]
            w=np.exp(-(dx*dx+dy*dy)/(2.0*SIGMA_M*SIGMA_M))
            # Mass-conserving soft allocation: every building contributes
            # exactly one unit of membership across the 16 anchors.
            denom=w.sum(axis=1,keepdims=True)
            w=np.divide(w,denom,out=np.zeros_like(w),where=denom>0)

            for j in range(na):
                np.add.at(count[vi,:,j],pi,w[:,j])
                np.add.at(area_sum[vi,:,j],pi,w[:,j]*ba)
                np.add.at(comp_sum[vi,:,j],pi,w[:,j]*comp)

    records=[]
    for vi,name in enumerate(names):
        for i,r in samples.iterrows():
            feat={}
            for j in range(na):
                mass=count[vi,i,j]
                feat[f"n{j:02d}_soft_count"]=mass
                feat[f"n{j:02d}_soft_area"]=area_sum[vi,i,j]/10000.0
                feat[f"n{j:02d}_mean_area"]=area_sum[vi,i,j]/mass if mass>0 else 0.0
                feat[f"n{j:02d}_mean_compactness"]=comp_sum[vi,i,j]/mass if mass>0 else 0.0
            records.append({
                "city":args.city,
                "split":r["split"],
                "sample_index":int(r["sample_index"]),
                "label":int(r["lcz_index_1to17"]),
                "variant":name,
                **feat,
            })

    out=pd.DataFrame(records)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    out.to_parquet(args.output,index=False)

if __name__=="__main__":
    main()
