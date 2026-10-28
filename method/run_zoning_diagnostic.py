#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-dir",default="work/features")
    ap.add_argument("--out-dir",default="results/zoning_diagnostic")
    args=ap.parse_args()
    files=sorted(Path(args.input_dir).glob("*.parquet"))
    df=pd.concat([pd.read_parquet(f) for f in files],ignore_index=True)

    # City-level external validation: each city held out in turn.
    feature_cols=[c for c in df.columns if c.startswith("c")]
    rows=[]
    per_sample=[]
    cities=sorted(df.city.unique())

    for test_city in cities:
        train=df[(df.city!=test_city)&(df.zoning=="z0")]
        test=df[df.city==test_city]
        model=HistGradientBoostingClassifier(
            learning_rate=0.08,max_iter=250,max_leaf_nodes=31,l2_regularization=1.0,random_state=42
        )
        model.fit(train[feature_cols],train.label)

        z_acc={}
        pred_by_z={}
        truth=None
        ids=None
        for z,g in test.groupby("zoning",sort=True):
            g=g.sort_values(["split","sample_index"]).reset_index(drop=True)
            pred=model.predict(g[feature_cols])
            z_acc[z]=accuracy_score(g.label,pred)
            pred_by_z[z]=pred
            if truth is None:
                truth=g.label.to_numpy()
                ids=g[["split","sample_index"]].copy()

        mat=np.column_stack([pred_by_z[z] for z in sorted(pred_by_z)])
        correct=(mat==truth[:,None])
        exposed=(correct.any(axis=1)&(~correct).any(axis=1))
        disagree=np.apply_along_axis(lambda x: len(set(x.tolist()))>1,1,mat)

        rows.append({
            "city":test_city,
            "n":len(truth),
            "canonical_accuracy":z_acc["z0"],
            "mean_zoning_accuracy":float(np.mean(list(z_acc.values()))),
            "worst_zoning_accuracy":float(np.min(list(z_acc.values()))),
            "canonical_to_worst_drop_pp":100*(z_acc["z0"]-np.min(list(z_acc.values()))),
            "pier":float(exposed.mean()),
            "prediction_disagreement":float(disagree.mean()),
        })
        tmp=ids.copy()
        tmp["city"]=test_city
        tmp["label"]=truth
        tmp["pier_exposed"]=exposed.astype(int)
        tmp["prediction_disagreement"]=disagree.astype(int)
        per_sample.append(tmp)

    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    city=pd.DataFrame(rows)
    city.to_csv(out/"city_summary.csv",index=False)
    pd.concat(per_sample,ignore_index=True).to_csv(out/"sample_flags.csv",index=False)
    overall=pd.DataFrame([{
        "cities":len(city),
        "mean_canonical_accuracy":city.canonical_accuracy.mean(),
        "mean_worst_zoning_accuracy":city.worst_zoning_accuracy.mean(),
        "mean_drop_pp":city.canonical_to_worst_drop_pp.mean(),
        "mean_pier":city.pier.mean(),
        "cities_pier_ge_5pct":int((city.pier>=0.05).sum()),
        "cities_drop_ge_2pp":int((city.canonical_to_worst_drop_pp>=2).sum()),
    }])
    overall.to_csv(out/"overall_summary.csv",index=False)

if __name__=="__main__":
    main()
