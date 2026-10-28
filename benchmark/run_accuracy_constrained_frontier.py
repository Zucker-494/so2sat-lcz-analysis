#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ALPHAS = np.round(np.linspace(0.0, 1.0, 21), 2)
DELTAS_PP = [0.0, 0.5, 1.0, 2.0]

def make_model(name):
    if name == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.06, max_iter=300, max_leaf_nodes=31,
            l2_regularization=1.0, class_weight="balanced", random_state=42
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=400, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
    raise ValueError(name)

def hard_cols(df):
    return [c for c in df.columns if len(c)>=4 and c[0]=="c" and c[1:3].isdigit() and c[3]=="_"]

def cont_cols(df):
    return [c for c in df.columns if c.startswith("a") and len(c)>4]

def ordered(df, city, view_col, views):
    groups=[]; keys=None; y=None
    for v in views:
        g=df[(df.city==city)&(df[view_col]==v)].sort_values(["split","sample_index"]).reset_index(drop=True)
        if keys is None:
            keys=g[["split","sample_index"]].copy()
            y=g.label.to_numpy()
        elif not g[["split","sample_index"]].equals(keys):
            raise ValueError(f"key mismatch {city} {v}")
        groups.append(g)
    return keys,y,groups

def metrics(pred, y):
    correct=pred==y[:,None]
    pier=correct.any(1)&(~correct).any(1)
    disagree=np.array([len(set(r.tolist()))>1 for r in pred])
    canon=correct[:,0]
    alt=correct[:,1:]
    return {
        "canonical_accuracy": accuracy_score(y,pred[:,0]),
        "canonical_balanced_accuracy": balanced_accuracy_score(y,pred[:,0]),
        "canonical_macro_f1": f1_score(y,pred[:,0],average="macro",zero_division=0),
        "pier": float(pier.mean()),
        "disagreement": float(disagree.mean()),
        "correct_to_wrong": float((canon&(~alt).any(1)).mean()),
        "wrong_to_correct": float(((~canon)&alt.any(1)).mean()),
    }

def align_proba(model, X):
    p=model.predict_proba(X)
    out=np.zeros((len(X),10),dtype=float)
    for j,c in enumerate(model.classes_):
        out[:,int(c)-1]=p[:,j]
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--hard-dir",default="work/hard")
    ap.add_argument("--continuous-dir",default="work/continuous_sensitivity")
    ap.add_argument("--out-dir",default="results/accuracy_constrained_frontier")
    args=ap.parse_args()

    hard=pd.concat([pd.read_parquet(f) for f in sorted(Path(args.hard_dir).glob("*.parquet"))],ignore_index=True)
    cont=pd.concat([pd.read_parquet(f) for f in sorted(Path(args.continuous_dir).glob("*.parquet"))],ignore_index=True)
    hfeat=hard_cols(hard); cfeat=cont_cols(cont)
    cities=sorted(set(hard.city.unique())&set(cont.city.unique()))
    views=[f"z{i}" for i in range(9)]

    rows=[]

    for city in cities:
        htrain=hard[(hard.city!=city)&(hard.zoning=="z0")]
        strain=cont[(cont.city!=city)&(cont.family=="anchor_position")&(cont.variant=="a0")]

        hkeys,y,hgroups=ordered(hard,city,"zoning",views)
        s0=cont[(cont.city==city)&(cont.family=="anchor_position")&(cont.variant=="a0")].sort_values(["split","sample_index"]).reset_index(drop=True)
        if not s0[["split","sample_index"]].equals(hkeys):
            raise ValueError(f"hard/continuous key mismatch {city}")

        for model_name in ("hgb","extra_trees"):
            hm=make_model(model_name); sm=make_model(model_name)
            hm.fit(htrain[hfeat],htrain.label)
            sm.fit(strain[cfeat],strain.label)

            hp=[align_proba(hm,g[hfeat]) for g in hgroups]
            sp=align_proba(sm,s0[cfeat])

            for alpha in ALPHAS:
                pred=[]
                for p in hp:
                    fused=(1.0-alpha)*p + alpha*sp
                    pred.append(np.argmax(fused,axis=1)+1)
                pred=np.column_stack(pred)
                m=metrics(pred,y)
                rows.append({
                    "city":city,
                    "model":model_name,
                    "alpha":float(alpha),
                    **m,
                })

    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    r=pd.DataFrame(rows)
    r.to_csv(out/"frontier_by_city.csv",index=False)

    agg=(r.groupby(["model","alpha"],as_index=False)
         .agg(
             mean_accuracy=("canonical_accuracy","mean"),
             mean_balanced_accuracy=("canonical_balanced_accuracy","mean"),
             mean_macro_f1=("canonical_macro_f1","mean"),
             mean_pier=("pier","mean"),
             mean_disagreement=("disagreement","mean"),
             mean_correct_to_wrong=("correct_to_wrong","mean"),
             min_city_accuracy=("canonical_accuracy","min"),
         ))
    agg.to_csv(out/"aggregate_frontier.csv",index=False)

    # Diagnostic only: identify whether a feasible region exists under explicit
    # mean-accuracy constraints. This is NOT a leakage-free final selection.
    selected=[]
    for model_name,g in agg.groupby("model"):
        base=g[g.alpha==0.0].iloc[0]
        for delta in DELTAS_PP:
            floor=base.mean_accuracy-delta/100.0
            feasible=g[g.mean_accuracy>=floor].copy()
            feasible=feasible.sort_values(
                ["mean_pier","mean_correct_to_wrong","mean_accuracy"],
                ascending=[True,True,False]
            )
            best=feasible.iloc[0]
            selected.append({
                "model":model_name,
                "delta_pp":delta,
                "baseline_accuracy":base.mean_accuracy,
                "baseline_pier":base.mean_pier,
                "selected_alpha":best.alpha,
                "selected_accuracy":best.mean_accuracy,
                "accuracy_change_pp":100*(best.mean_accuracy-base.mean_accuracy),
                "selected_pier":best.mean_pier,
                "pier_relative_reduction":(base.mean_pier-best.mean_pier)/base.mean_pier if base.mean_pier>0 else np.nan,
                "selected_correct_to_wrong":best.mean_correct_to_wrong,
                "selected_disagreement":best.mean_disagreement,
            })
    pd.DataFrame(selected).to_csv(out/"feasibility_under_accuracy_constraints.csv",index=False)

if __name__=="__main__":
    main()
