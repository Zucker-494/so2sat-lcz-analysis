#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import Transformer
from shapely import from_wkb, centroid, area, length, is_empty, is_valid, make_valid, transform as shp_transform
from shapely.strtree import STRtree
from shapely import box

# Canonical zoning plus eight alternatives at the same 4 × 4 scale.
ZONINGS = {
    "z0": ([80,160,240],[80,160,240]),
    "z1": ([65,160,245],[80,160,240]),
    "z2": ([95,160,225],[80,160,240]),
    "z3": ([80,145,240],[80,175,240]),
    "z4": ([80,175,240],[80,145,240]),
    "z5": ([65,145,225],[95,175,245]),
    "z6": ([95,175,245],[65,145,225]),
    "z7": ([70,170,230],[90,150,250]),
    "z8": ([90,150,250],[70,170,230]),
}
BATCH_SIZE = 100_000

def load_samples(city):
    frames=[]
    for split in ("validation","testing"):
        p=Path("results")/"samples"/split/f"{city}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(city)
    df=pd.concat(frames,ignore_index=True)
    return df[df["lcz_index_1to17"]<=10].copy()

def cell_index(xrel,yrel,xcuts,ycuts):
    xi=np.searchsorted(np.asarray(xcuts),xrel,side="right")
    yi=np.searchsorted(np.asarray(ycuts),yrel,side="right")
    return yi*4+xi

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--city",required=True)
    ap.add_argument("--buildings",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()

    samples=load_samples(args.city).reset_index(drop=True)
    epsg=int(samples["epsg"].iloc[0])
    patch_geoms=np.asarray([
        box(float(r.x_min),float(r.y_min),float(r.x_max),float(r.y_max))
        for _,r in samples.iterrows()
    ],dtype=object)
    tree=STRtree(patch_geoms)

    # Array dimensions: zoning, patch, cell, and morphology measure.
    nz=len(ZONINGS); npatch=len(samples); nc=16
    count=np.zeros((nz,npatch,nc),dtype=np.int32)
    area_sum=np.zeros((nz,npatch,nc),dtype=np.float64)
    compact_sum=np.zeros((nz,npatch,nc),dtype=np.float64)

    to_proj=Transformer.from_crs(4326,epsg,always_xy=True)
    pf=pq.ParquetFile(args.buildings)

    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["geometry"]):
        wkb=batch.column(0).to_numpy(zero_copy_only=False)
        g=from_wkb(wkb)
        good=~is_empty(g)
        g=g[good]
        valid=is_valid(g)
        if not valid.all():
            g=g.copy()
            g[~valid]=make_valid(g[~valid])
        gp=shp_transform(g,to_proj.transform,interleaved=False)
        c=centroid(gp)
        # STRtree applies the predicate as predicate(input_geometry, tree_geometry).
        # Here the input is a building centroid and the tree contains LCZ patches,
        # so the correct relation is Point.within(Polygon).
        pairs=tree.query(c,predicate="within")
        if pairs.shape[1]==0:
            continue
        bi=pairs[0]; pi=pairs[1]
        cg=c[bi]; bg=gp[bi]
        # Building centroid coordinates relative to the fixed patch.
        xs=np.asarray([geom.x for geom in cg])
        ys=np.asarray([geom.y for geom in cg])
        xmin=samples.iloc[pi]["x_min"].to_numpy(float)
        ymin=samples.iloc[pi]["y_min"].to_numpy(float)
        xr=xs-xmin; yr=ys-ymin
        ba=area(bg)
        per=length(bg)
        comp=np.where(per>0,4*np.pi*ba/(per*per),0.0)

        for zi,(name,(xc,yc)) in enumerate(ZONINGS.items()):
            ci=cell_index(xr,yr,xc,yc)
            np.add.at(count[zi],(pi,ci),1)
            np.add.at(area_sum[zi],(pi,ci),ba)
            np.add.at(compact_sum[zi],(pi,ci),comp)

    records=[]
    for zi,name in enumerate(ZONINGS):
        xc,yc=ZONINGS[name]
        xbounds=np.asarray([0]+xc+[320],dtype=float)
        ybounds=np.asarray([0]+yc+[320],dtype=float)
        cell_area=np.array([
            (xbounds[x+1]-xbounds[x])*(ybounds[y+1]-ybounds[y])
            for y in range(4) for x in range(4)
        ])
        for pidx,r in samples.iterrows():
            feat={}
            for ci in range(16):
                n=count[zi,pidx,ci]
                feat[f"c{ci:02d}_count_density"]=n/cell_area[ci]*10000.0
                feat[f"c{ci:02d}_footprint_density"]=area_sum[zi,pidx,ci]/cell_area[ci]
                feat[f"c{ci:02d}_mean_area"]=area_sum[zi,pidx,ci]/n if n else 0.0
                feat[f"c{ci:02d}_mean_compactness"]=compact_sum[zi,pidx,ci]/n if n else 0.0
            records.append({
                "city":args.city,
                "split":r["split"],
                "sample_index":int(r["sample_index"]),
                "label":int(r["lcz_index_1to17"]),
                "zoning":name,
                **feat
            })
    out=pd.DataFrame(records)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    out.to_parquet(args.output,index=False)
    meta={"city":args.city,"n_patches":int(npatch),"zonings":ZONINGS,"feature_count":64}
    Path(args.output).with_suffix(".json").write_text(json.dumps(meta,indent=2),encoding="utf-8")

if __name__=="__main__":
    main()
