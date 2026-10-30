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
    if not files:
        raise FileNotFoundError(f"No feature parquet files found in {args.input_dir}")
    df=pd.concat([pd.read_parquet(f) for f in files],ignore_index=True)

    feature_cols=[
        c for c in df.columns
        if len(c) >= 4 and c[0] == "c" and c[1:3].isdigit() and c[3] == "_"
    ]

    city_rows=[]
    sample_rows=[]
    zoning_rows=[]
    direction_rows=[]
    class_rows=[]

    cities=sorted(df.city.unique())

    for test_city in cities:
        train=df[(df.city!=test_city)&(df.zoning=="z0")]
        test=df[df.city==test_city]

        model=HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=250,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        )
        model.fit(train[feature_cols],train.label)

        z_acc={}
        pred_by_z={}
        truth=None
        ids=None

        for z,g in test.groupby("zoning",sort=True):
            g=g.sort_values(["split","sample_index"]).reset_index(drop=True)
            pred=model.predict(g[feature_cols])
            acc=accuracy_score(g.label,pred)
            z_acc[z]=acc
            pred_by_z[z]=pred
            zoning_rows.append({
                "city":test_city,
                "zoning":z,
                "n":len(g),
                "accuracy":acc,
            })
            if truth is None:
                truth=g.label.to_numpy()
                ids=g[["split","sample_index"]].copy()

        zonings=sorted(pred_by_z)
        mat=np.column_stack([pred_by_z[z] for z in zonings])
        correct=(mat==truth[:,None])

        canonical_idx=zonings.index("z0")
        canonical_correct=correct[:,canonical_idx]
        exposed=(correct.any(axis=1)&(~correct).any(axis=1))
        disagree=np.apply_along_axis(lambda x: len(set(x.tolist()))>1,1,mat)

        alt_correct=np.delete(correct,canonical_idx,axis=1)
        any_alt_wrong=(~alt_correct).any(axis=1)
        any_alt_correct=alt_correct.any(axis=1)

        cw_any=canonical_correct & any_alt_wrong
        wc_any=(~canonical_correct) & any_alt_correct

        city_rows.append({
            "city":test_city,
            "n":len(truth),
            "canonical_accuracy":z_acc["z0"],
            "mean_zoning_accuracy":float(np.mean(list(z_acc.values()))),
            "worst_zoning_accuracy":float(np.min(list(z_acc.values()))),
            "canonical_to_worst_drop_pp":100*(z_acc["z0"]-np.min(list(z_acc.values()))),
            "zoning_sensitivity":float(exposed.mean()),
            "prediction_disagreement":float(disagree.mean()),
            "canonical_correct_any_alt_wrong_rate_all":float(cw_any.mean()),
            "canonical_wrong_any_alt_correct_rate_all":float(wc_any.mean()),
            "canonical_correct_any_alt_wrong_rate_conditional":float(cw_any.sum()/canonical_correct.sum()) if canonical_correct.sum() else np.nan,
            "canonical_wrong_any_alt_correct_rate_conditional":float(wc_any.sum()/(~canonical_correct).sum()) if (~canonical_correct).sum() else np.nan,
        })

        for z in zonings:
            if z=="z0":
                continue
            zc=(pred_by_z[z]==truth)
            cw=canonical_correct & (~zc)
            wc=(~canonical_correct) & zc
            direction_rows.append({
                "city":test_city,
                "zoning":z,
                "n":len(truth),
                "canonical_accuracy":z_acc["z0"],
                "zoning_accuracy":z_acc[z],
                "accuracy_change_pp":100*(z_acc[z]-z_acc["z0"]),
                "correct_to_wrong_rate_all":float(cw.mean()),
                "wrong_to_correct_rate_all":float(wc.mean()),
                "correct_to_wrong_rate_conditional":float(cw.sum()/canonical_correct.sum()) if canonical_correct.sum() else np.nan,
                "wrong_to_correct_rate_conditional":float(wc.sum()/(~canonical_correct).sum()) if (~canonical_correct).sum() else np.nan,
            })

        tmp=ids.copy()
        tmp["city"]=test_city
        tmp["label"]=truth
        tmp["canonical_correct"]=canonical_correct.astype(int)
        tmp["zoning_sensitivity_exposed"]=exposed.astype(int)
        tmp["prediction_disagreement"]=disagree.astype(int)
        tmp["canonical_correct_any_alt_wrong"]=cw_any.astype(int)
        tmp["canonical_wrong_any_alt_correct"]=wc_any.astype(int)
        sample_rows.append(tmp)

        for label in sorted(np.unique(truth)):
            m=(truth==label)
            class_rows.append({
                "city":test_city,
                "label":int(label),
                "n":int(m.sum()),
                "canonical_accuracy":float(canonical_correct[m].mean()),
                "zoning_sensitivity":float(exposed[m].mean()),
                "prediction_disagreement":float(disagree[m].mean()),
                "canonical_correct_any_alt_wrong_rate_all":float(cw_any[m].mean()),
                "canonical_wrong_any_alt_correct_rate_all":float(wc_any[m].mean()),
            })

    out=Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)

    city=pd.DataFrame(city_rows)
    city.to_csv(out/"city_summary.csv",index=False)
    pd.DataFrame(zoning_rows).to_csv(out/"per_zoning_accuracy.csv",index=False)
    pd.DataFrame(direction_rows).to_csv(out/"directional_transitions.csv",index=False)
    pd.DataFrame(class_rows).to_csv(out/"class_summary.csv",index=False)
    pd.concat(sample_rows,ignore_index=True).to_csv(out/"sample_flags.csv",index=False)

    overall=pd.DataFrame([{
        "cities":len(city),
        "mean_canonical_accuracy":city.canonical_accuracy.mean(),
        "mean_worst_zoning_accuracy":city.worst_zoning_accuracy.mean(),
        "mean_drop_pp":city.canonical_to_worst_drop_pp.mean(),
        "mean_zoning_sensitivity":city.zoning_sensitivity.mean(),
        "mean_prediction_disagreement":city.prediction_disagreement.mean(),
        "mean_canonical_correct_any_alt_wrong_rate_all":city.canonical_correct_any_alt_wrong_rate_all.mean(),
        "mean_canonical_wrong_any_alt_correct_rate_all":city.canonical_wrong_any_alt_correct_rate_all.mean(),
        "cities_zoning_sensitivity_ge_5pct":int((city.zoning_sensitivity>=0.05).sum()),
    }])
    overall.to_csv(out/"overall_summary.csv",index=False)

if __name__=="__main__":
    main()
