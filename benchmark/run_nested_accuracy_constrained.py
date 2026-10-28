#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ALPHAS = np.round(np.linspace(0.0, 1.0, 21), 2)
DELTAS_PP = [0.0, 0.5, 1.0]

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

def aligned_views(hard, cont, city, hfeat, cfeat):
    hviews=[f"z{i}" for i in range(9)]
    hs=[]; keys=None; y=None
    for z in hviews:
        g=hard[(hard.city==city)&(hard.zoning==z)].sort_values(["split","sample_index"]).reset_index(drop=True)
        if keys is None:
            keys=g[["split","sample_index"]].copy()
            y=g.label.to_numpy()
        elif not g[["split","sample_index"]].equals(keys):
            raise ValueError(f"hard key mismatch {city} {z}")
        hs.append(g[hfeat])

    cg=cont[(cont.city==city)&(cont.family=="anchor_position")&(cont.variant=="a0")]
    cg=cg.sort_values(["split","sample_index"]).reset_index(drop=True)
    if not cg[["split","sample_index"]].equals(keys):
        raise ValueError(f"continuous key mismatch {city}")
    return keys,y,hs,cg[cfeat]

def aligned_proba(model, X):
    p=model.predict_proba(X)
    out=np.zeros((len(X),10),dtype=float)
    for j,c in enumerate(model.classes_):
        out[:,int(c)-1]=p[:,j]
    return out

def fit_pair(hard, cont, train_cities, model_name, hfeat, cfeat):
    ht=hard[(hard.city.isin(train_cities))&(hard.zoning=="z0")]
    ct=cont[(cont.city.isin(train_cities))&(cont.family=="anchor_position")&(cont.variant=="a0")]
    hm=make_model(model_name); cm=make_model(model_name)
    hm.fit(ht[hfeat],ht.label)
    cm.fit(ct[cfeat],ct.label)
    return hm,cm

def evaluate_pair(hm,cm,hard,cont,city,hfeat,cfeat,alpha):
    _,y,hx,cx=aligned_views(hard,cont,city,hfeat,cfeat)
    hp=[aligned_proba(hm,x) for x in hx]
    cp=aligned_proba(cm,cx)
    pred=[]
    for p in hp:
        fused=(1.0-alpha)*p + alpha*cp
        pred.append(np.argmax(fused,axis=1)+1)
    pred=np.column_stack(pred)

    correct=pred==y[:,None]
    pier=correct.any(1)&(~correct).any(1)
    disagreement=np.array([len(set(r.tolist()))>1 for r in pred])
    canon=correct[:,0]
    alt=correct[:,1:]
    return {
        "accuracy":accuracy_score(y,pred[:,0]),
        "balanced_accuracy":balanced_accuracy_score(y,pred[:,0]),
        "macro_f1":f1_score(y,pred[:,0],average="macro",zero_division=0),
        "pier":float(pier.mean()),
        "disagreement":float(disagreement.mean()),
        "correct_to_wrong":float((canon&(~alt).any(1)).mean()),
        "wrong_to_correct":float(((~canon)&alt.any(1)).mean()),
        "n":len(y),
    }

def select_alpha_nested(hard,cont,outer_train,model_name,hfeat,cfeat):
    inner_rows=[]
    for val_city in outer_train:
        inner_train=[c for c in outer_train if c!=val_city]
        hm,cm=fit_pair(hard,cont,inner_train,model_name,hfeat,cfeat)
        for alpha in ALPHAS:
            m=evaluate_pair(hm,cm,hard,cont,val_city,hfeat,cfeat,float(alpha))
            inner_rows.append({"val_city":val_city,"alpha":float(alpha),**m})

    inner=pd.DataFrame(inner_rows)
    agg=(inner.groupby("alpha",as_index=False)
         .agg(mean_accuracy=("accuracy","mean"),
              mean_pier=("pier","mean"),
              mean_correct_to_wrong=("correct_to_wrong","mean"),
              mean_disagreement=("disagreement","mean"),
              min_city_accuracy=("accuracy","min")))
    base=agg[agg.alpha==0.0].iloc[0]

    selections={}
    for delta in DELTAS_PP:
        floor=base.mean_accuracy-delta/100.0
        feasible=agg[agg.mean_accuracy>=floor].copy()
        feasible=feasible.sort_values(
            ["mean_pier","mean_correct_to_wrong","mean_accuracy"],
            ascending=[True,True,False]
        )
        best=feasible.iloc[0]
        selections[delta]={
            "alpha":float(best.alpha),
            "inner_baseline_accuracy":float(base.mean_accuracy),
            "inner_selected_accuracy":float(best.mean_accuracy),
            "inner_selected_pier":float(best.mean_pier),
            "inner_selected_correct_to_wrong":float(best.mean_correct_to_wrong),
        }
    return selections,inner,agg

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--hard-dir",default="work/hard")
    ap.add_argument("--continuous-dir",default="work/continuous_sensitivity")
    ap.add_argument("--out-dir",default="results/nested_accuracy_constrained")
    args=ap.parse_args()

    hard=pd.concat([pd.read_parquet(f) for f in sorted(Path(args.hard_dir).glob("*.parquet"))],ignore_index=True)
    cont=pd.concat([pd.read_parquet(f) for f in sorted(Path(args.continuous_dir).glob("*.parquet"))],ignore_index=True)
    hfeat=hard_cols(hard); cfeat=cont_cols(cont)
    cities=sorted(set(hard.city.unique())&set(cont.city.unique()))

    outer_rows=[]
    selection_rows=[]
    inner_curve_rows=[]

    for outer_city in cities:
        outer_train=[c for c in cities if c!=outer_city]

        for model_name in ("hgb","extra_trees"):
            selections,inner,agg=select_alpha_nested(
                hard,cont,outer_train,model_name,hfeat,cfeat
            )
            agg=agg.copy()
            agg["outer_city"]=outer_city
            agg["model"]=model_name
            inner_curve_rows.append(agg)

            hm,cm=fit_pair(hard,cont,outer_train,model_name,hfeat,cfeat)

            # Always evaluate the pure hard baseline once on the held-out city.
            base=evaluate_pair(hm,cm,hard,cont,outer_city,hfeat,cfeat,0.0)

            for delta,sel in selections.items():
                m=evaluate_pair(
                    hm,cm,hard,cont,outer_city,hfeat,cfeat,sel["alpha"]
                )
                outer_rows.append({
                    "outer_city":outer_city,
                    "model":model_name,
                    "delta_pp":delta,
                    "selected_alpha":sel["alpha"],
                    "baseline_accuracy":base["accuracy"],
                    "selected_accuracy":m["accuracy"],
                    "accuracy_change_pp":100*(m["accuracy"]-base["accuracy"]),
                    "baseline_pier":base["pier"],
                    "selected_pier":m["pier"],
                    "pier_relative_reduction":(
                        (base["pier"]-m["pier"])/base["pier"] if base["pier"]>0 else np.nan
                    ),
                    "baseline_correct_to_wrong":base["correct_to_wrong"],
                    "selected_correct_to_wrong":m["correct_to_wrong"],
                    "baseline_disagreement":base["disagreement"],
                    "selected_disagreement":m["disagreement"],
                    "selected_balanced_accuracy":m["balanced_accuracy"],
                    "selected_macro_f1":m["macro_f1"],
                    "n":m["n"],
                })
                selection_rows.append({
                    "outer_city":outer_city,
                    "model":model_name,
                    "delta_pp":delta,
                    **sel,
                })

    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    outer=pd.DataFrame(outer_rows)
    outer.to_csv(out/"outer_city_results.csv",index=False)
    pd.DataFrame(selection_rows).to_csv(out/"inner_selections.csv",index=False)
    pd.concat(inner_curve_rows,ignore_index=True).to_csv(out/"inner_frontiers.csv",index=False)

    agg=(outer.groupby(["model","delta_pp"],as_index=False)
         .agg(
             mean_selected_alpha=("selected_alpha","mean"),
             mean_baseline_accuracy=("baseline_accuracy","mean"),
             mean_selected_accuracy=("selected_accuracy","mean"),
             mean_accuracy_change_pp=("accuracy_change_pp","mean"),
             mean_baseline_pier=("baseline_pier","mean"),
             mean_selected_pier=("selected_pier","mean"),
             mean_pier_relative_reduction=("pier_relative_reduction","mean"),
             mean_baseline_correct_to_wrong=("baseline_correct_to_wrong","mean"),
             mean_selected_correct_to_wrong=("selected_correct_to_wrong","mean"),
             mean_baseline_disagreement=("baseline_disagreement","mean"),
             mean_selected_disagreement=("selected_disagreement","mean"),
             cities_accuracy_not_lower=("accuracy_change_pp",lambda s:int((s>=0).sum())),
             cities_accuracy_within_constraint=("accuracy_change_pp",lambda s:int((s>=-s.name[1] if False else s>=-999).sum()))
         ))
    # Replace the last placeholder with the actual per-delta city count.
    counts=[]
    for (model,delta),g in outer.groupby(["model","delta_pp"]):
        counts.append({
            "model":model,
            "delta_pp":delta,
            "cities_within_realized_accuracy_constraint":int((g.accuracy_change_pp>=-float(delta)).sum()),
            "cities_pier_reduced":int((g.selected_pier<g.baseline_pier).sum()),
            "cities_ctw_reduced":int((g.selected_correct_to_wrong<g.baseline_correct_to_wrong).sum()),
        })
    counts=pd.DataFrame(counts)
    agg=agg.drop(columns=["cities_accuracy_within_constraint"]).merge(counts,on=["model","delta_pp"],how="left")
    agg.to_csv(out/"aggregate_summary.csv",index=False)

if __name__=="__main__":
    main()
