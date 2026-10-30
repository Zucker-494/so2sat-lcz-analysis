#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

ALPHAS = np.round(np.linspace(0.0, 1.0, 21), 2)
DELTAS_PP = [0.0, 0.5, 1.0, 2.0]

def make_model(name):
    if name == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.06, max_iter=300, max_leaf_nodes=31,
            l2_regularization=1.0, class_weight="balanced", random_state=42
        )
    return ExtraTreesClassifier(
        n_estimators=400, max_features="sqrt", min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1
    )

def hcols(df):
    return [c for c in df.columns if len(c)>=4 and c[0]=="c" and c[1:3].isdigit() and c[3]=="_"]

def ccols(df):
    return [c for c in df.columns if c.startswith("a") and len(c)>4]

def aproba(model, X):
    p = model.predict_proba(X)
    out = np.zeros((len(X), 10))
    for j,c in enumerate(model.classes_):
        out[:, int(c)-1] = p[:,j]
    return out

def fit_pair(hard, cont, train_cities, model_name, hf, cf):
    hm, cm = make_model(model_name), make_model(model_name)
    ht = hard[(hard.city.isin(train_cities)) & (hard.zoning=="z0")]
    ct = cont[(cont.city.isin(train_cities)) & (cont.family=="anchor_position") & (cont.variant=="a0")]
    hm.fit(ht[hf], ht.label)
    cm.fit(ct[cf], ct.label)
    return hm, cm

def cache_city(hm, cm, hard, cont, city, hf, cf):
    views = [f"z{i}" for i in range(9)]
    hp = []
    keys = None
    y = None
    for z in views:
        g = hard[(hard.city==city)&(hard.zoning==z)].sort_values(["split","sample_index"]).reset_index(drop=True)
        if keys is None:
            keys = g[["split","sample_index"]].copy()
            y = g.label.to_numpy()
        elif not g[["split","sample_index"]].equals(keys):
            raise ValueError(f"hard key mismatch {city} {z}")
        hp.append(aproba(hm, g[hf]))
    cg = cont[(cont.city==city)&(cont.family=="anchor_position")&(cont.variant=="a0")]
    cg = cg.sort_values(["split","sample_index"]).reset_index(drop=True)
    if not cg[["split","sample_index"]].equals(keys):
        raise ValueError(f"continuous key mismatch {city}")
    cp = aproba(cm, cg[cf])
    return y, hp, cp

def eval_alpha(y, hp, cp, alpha):
    pred = np.column_stack([
        np.argmax((1-alpha)*p + alpha*cp, axis=1)+1 for p in hp
    ])
    corr = pred == y[:,None]
    canon = corr[:,0]
    alt = corr[:,1:]
    zoning_sensitivity = corr.any(1) & (~corr).any(1)
    disagree = np.array([len(set(r.tolist()))>1 for r in pred])
    return {
        "accuracy": float((pred[:,0]==y).mean()),
        "zoning_sensitivity": float(zoning_sensitivity.mean()),
        "correct_to_wrong": float((canon & (~alt).any(1)).mean()),
        "disagreement": float(disagree.mean()),
    }

def inner_curve(hard, cont, outer_train, model_name, hf, cf):
    rows=[]
    for val_city in outer_train:
        train=[c for c in outer_train if c!=val_city]
        hm,cm=fit_pair(hard,cont,train,model_name,hf,cf)
        y,hp,cp=cache_city(hm,cm,hard,cont,val_city,hf,cf)
        for alpha in ALPHAS:
            rows.append({"val_city":val_city,"alpha":float(alpha),**eval_alpha(y,hp,cp,float(alpha))})
    return pd.DataFrame(rows)

def choose_robust(inner, delta):
    base = inner[inner.alpha==0][["val_city","accuracy"]].rename(columns={"accuracy":"base_acc"})
    x = inner.merge(base,on="val_city")
    x["loss_pp"] = 100*(x.base_acc-x.accuracy)
    by_alpha = x.groupby("alpha",as_index=False).agg(
        mean_accuracy=("accuracy","mean"),
        mean_zoning_sensitivity=("zoning_sensitivity","mean"),
        mean_ctw=("correct_to_wrong","mean"),
        mean_disagreement=("disagreement","mean"),
        worst_city_loss_pp=("loss_pp","max"),
        mean_loss_pp=("loss_pp","mean"),
    )
    feasible = by_alpha[by_alpha.worst_city_loss_pp <= delta + 1e-12].copy()
    if feasible.empty:
        feasible = by_alpha[by_alpha.alpha==0].copy()
    feasible = feasible.sort_values(["mean_zoning_sensitivity","mean_ctw","mean_accuracy"],ascending=[True,True,False])
    return feasible.iloc[0], by_alpha

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--hard-dir",default="work/hard")
    ap.add_argument("--continuous-dir",default="work/continuous_sensitivity")
    ap.add_argument("--out-dir",default="results/robust_nested_accuracy_constraint")
    args=ap.parse_args()

    hard=pd.concat([pd.read_parquet(f) for f in sorted(Path(args.hard_dir).glob("*.parquet"))],ignore_index=True)
    cont=pd.concat([pd.read_parquet(f) for f in sorted(Path(args.continuous_dir).glob("*.parquet"))],ignore_index=True)
    hf,cf=hcols(hard),ccols(cont)
    cities=sorted(set(hard.city.unique())&set(cont.city.unique()))

    outer_rows=[]; select_rows=[]; curve_rows=[]

    for outer_city in cities:
        outer_train=[c for c in cities if c!=outer_city]
        for model_name in ("hgb","extra_trees"):
            inner=inner_curve(hard,cont,outer_train,model_name,hf,cf)
            hm,cm=fit_pair(hard,cont,outer_train,model_name,hf,cf)
            y,hp,cp=cache_city(hm,cm,hard,cont,outer_city,hf,cf)
            base=eval_alpha(y,hp,cp,0.0)

            for delta in DELTAS_PP:
                best,curve=choose_robust(inner,delta)
                alpha=float(best.alpha)
                m=eval_alpha(y,hp,cp,alpha)
                outer_rows.append({
                    "outer_city":outer_city,"model":model_name,"delta_pp":delta,
                    "selected_alpha":alpha,
                    "inner_worst_city_loss_pp":float(best.worst_city_loss_pp),
                    "baseline_accuracy":base["accuracy"],"selected_accuracy":m["accuracy"],
                    "accuracy_change_pp":100*(m["accuracy"]-base["accuracy"]),
                    "baseline_zoning_sensitivity":base["zoning_sensitivity"],"selected_zoning_sensitivity":m["zoning_sensitivity"],
                    "zoning_sensitivity_relative_reduction":((base["zoning_sensitivity"]-m["zoning_sensitivity"])/base["zoning_sensitivity"] if base["zoning_sensitivity"]>0 else np.nan),
                    "baseline_ctw":base["correct_to_wrong"],"selected_ctw":m["correct_to_wrong"],
                    "baseline_disagreement":base["disagreement"],"selected_disagreement":m["disagreement"],
                })
                select_rows.append({
                    "outer_city":outer_city,"model":model_name,"delta_pp":delta,
                    "selected_alpha":alpha,
                    "inner_worst_city_loss_pp":float(best.worst_city_loss_pp),
                    "inner_mean_accuracy":float(best.mean_accuracy),
                    "inner_mean_zoning_sensitivity":float(best.mean_zoning_sensitivity),
                })

            curve["outer_city"]=outer_city
            curve["model"]=model_name
            curve_rows.append(curve)

    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    outer=pd.DataFrame(outer_rows)
    outer.to_csv(out/"outer_city_results.csv",index=False)
    pd.DataFrame(select_rows).to_csv(out/"inner_selections.csv",index=False)
    pd.concat(curve_rows,ignore_index=True).to_csv(out/"inner_robust_frontiers.csv",index=False)

    rows=[]
    for (model,delta),g in outer.groupby(["model","delta_pp"]):
        rows.append({
            "model":model,"delta_pp":delta,
            "mean_selected_alpha":g.selected_alpha.mean(),
            "mean_baseline_accuracy":g.baseline_accuracy.mean(),
            "mean_selected_accuracy":g.selected_accuracy.mean(),
            "mean_accuracy_change_pp":g.accuracy_change_pp.mean(),
            "worst_outer_accuracy_change_pp":g.accuracy_change_pp.min(),
            "cities_within_realized_constraint":int((g.accuracy_change_pp>=-delta-1e-12).sum()),
            "mean_baseline_zoning_sensitivity":g.baseline_zoning_sensitivity.mean(),
            "mean_selected_zoning_sensitivity":g.selected_zoning_sensitivity.mean(),
            "mean_zoning_sensitivity_relative_reduction":g.zoning_sensitivity_relative_reduction.mean(),
            "cities_zoning_sensitivity_reduced":int((g.selected_zoning_sensitivity<g.baseline_zoning_sensitivity).sum()),
            "mean_baseline_ctw":g.baseline_ctw.mean(),
            "mean_selected_ctw":g.selected_ctw.mean(),
            "cities_ctw_reduced":int((g.selected_ctw<g.baseline_ctw).sum()),
        })
    pd.DataFrame(rows).to_csv(out/"aggregate_summary.csv",index=False)

if __name__=="__main__":
    main()
