#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

BOOT = 5000
SEED = 42
DELTAS = [0.5, 1.0]

def model():
    return HistGradientBoostingClassifier(
        learning_rate=0.06, max_iter=300, max_leaf_nodes=31,
        l2_regularization=1.0, class_weight="balanced", random_state=42
    )

def hcols(df):
    return [c for c in df.columns if len(c)>=4 and c[0]=="c" and c[1:3].isdigit() and c[3]=="_"]

def ccols(df):
    return [c for c in df.columns if c.startswith("a") and len(c)>4]

def aproba(m, X):
    p=m.predict_proba(X)
    out=np.zeros((len(X),10),dtype=float)
    for j,c in enumerate(m.classes_):
        out[:,int(c)-1]=p[:,j]
    return out

def fit_pair(hard,cont,train_cities,hf,cf):
    hm,cm=model(),model()
    ht=hard[(hard.city.isin(train_cities))&(hard.zoning=="z0")]
    ct=cont[(cont.city.isin(train_cities))&(cont.family=="anchor_position")&(cont.variant=="a0")]
    hm.fit(ht[hf],ht.label)
    cm.fit(ct[cf],ct.label)
    return hm,cm

def cache_city(hm,cm,hard,cont,city,hf,cf):
    views=[f"z{i}" for i in range(9)]
    hp=[]; keys=None; y=None
    for z in views:
        g=hard[(hard.city==city)&(hard.zoning==z)].sort_values(["split","sample_index"]).reset_index(drop=True)
        if keys is None:
            keys=g[["split","sample_index"]].copy()
            y=g.label.to_numpy()
        elif not g[["split","sample_index"]].equals(keys):
            raise ValueError(f"hard key mismatch {city} {z}")
        hp.append(aproba(hm,g[hf]))
    cg=cont[(cont.city==city)&(cont.family=="anchor_position")&(cont.variant=="a0")]
    cg=cg.sort_values(["split","sample_index"]).reset_index(drop=True)
    if not cg[["split","sample_index"]].equals(keys):
        raise ValueError(f"continuous key mismatch {city}")
    cp=aproba(cm,cg[cf])
    return y,hp,cp

def predict(y,hp,cp,alpha):
    return np.column_stack([
        np.argmax((1-alpha)*p + alpha*cp,axis=1)+1 for p in hp
    ])

def metrics(pred,y):
    corr=pred==y[:,None]
    canon=corr[:,0]
    alt=corr[:,1:]
    pier=corr.any(1)&(~corr).any(1)
    disagree=np.array([len(set(row.tolist()))>1 for row in pred])
    return {
        "accuracy":accuracy_score(y,pred[:,0]),
        "balanced_accuracy":balanced_accuracy_score(y,pred[:,0]),
        "macro_f1":f1_score(y,pred[:,0],average="macro",zero_division=0),
        "pier":pier.mean(),
        "correct_to_wrong":(canon&(~alt).any(1)).mean(),
        "wrong_to_correct":((~canon)&alt.any(1)).mean(),
        "disagreement":disagree.mean(),
    }, corr, canon, alt, pier, disagree

def cluster_bootstrap(citydf, metric, rng):
    d=(citydf[f"selected_{metric}"]-citydf[f"baseline_{metric}"]).to_numpy(float)
    n=len(d)
    vals=np.empty(BOOT)
    for i in range(BOOT):
        idx=rng.integers(0,n,size=n)
        vals[i]=d[idx].mean()
    lo,hi=np.quantile(vals,[0.025,0.975])
    return d.mean(),float(lo),float(hi)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--hard-dir",default="work/hard")
    ap.add_argument("--continuous-dir",default="work/continuous_sensitivity")
    ap.add_argument("--selections",default="results/robust_nested_accuracy_constraint/inner_selections.csv")
    ap.add_argument("--out-dir",default="results/final_hgb_confirmation")
    args=ap.parse_args()

    hard=pd.concat([pd.read_parquet(f) for f in sorted(Path(args.hard_dir).glob("*.parquet"))],ignore_index=True)
    cont=pd.concat([pd.read_parquet(f) for f in sorted(Path(args.continuous_dir).glob("*.parquet"))],ignore_index=True)
    sel=pd.read_csv(args.selections)
    sel=sel[(sel.model=="hgb") & (sel.delta_pp.isin(DELTAS))].copy()
    hf,cf=hcols(hard),ccols(cont)
    cities=sorted(set(hard.city.unique())&set(cont.city.unique()))

    rows=[]; class_rows=[]
    for city in cities:
        train=[c for c in cities if c!=city]
        hm,cm=fit_pair(hard,cont,train,hf,cf)
        y,hp,cp=cache_city(hm,cm,hard,cont,city,hf,cf)
        base_pred=predict(y,hp,cp,0.0)
        base_m,*_=metrics(base_pred,y)

        for delta in DELTAS:
            s=sel[(sel.outer_city==city)&(sel.delta_pp==delta)]
            if len(s)!=1:
                raise ValueError(f"selection missing {city} {delta}")
            alpha=float(s.iloc[0].selected_alpha)
            pred=predict(y,hp,cp,alpha)
            m,corr,canon,alt,pier,disagree=metrics(pred,y)

            row={"city":city,"delta_pp":delta,"selected_alpha":alpha,"n":len(y)}
            for k,v in base_m.items(): row[f"baseline_{k}"]=v
            for k,v in m.items(): row[f"selected_{k}"]=v
            row["accuracy_change_pp"]=100*(m["accuracy"]-base_m["accuracy"])
            row["constraint_met"]=row["accuracy_change_pp"]>=-delta-1e-12
            rows.append(row)

            # Recompute baseline indicators for the LCZ class comparison.
            bcorr=base_pred==y[:,None]
            bpier=bcorr.any(1)&(~bcorr).any(1)
            for label in sorted(np.unique(y)):
                mask=y==label
                class_rows.append({
                    "city":city,"delta_pp":delta,"label":int(label),"n":int(mask.sum()),
                    "baseline_accuracy":float((base_pred[mask,0]==y[mask]).mean()),
                    "selected_accuracy":float((pred[mask,0]==y[mask]).mean()),
                    "baseline_pier":float(bpier[mask].mean()),
                    "selected_pier":float(pier[mask].mean()),
                })

    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    citydf=pd.DataFrame(rows)
    citydf.to_csv(out/"city_summary.csv",index=False)
    cdf=pd.DataFrame(class_rows)
    cdf.to_csv(out/"city_class_summary.csv",index=False)

    rng=np.random.default_rng(SEED)
    inf=[]; agg=[]
    metrics_list=["accuracy","balanced_accuracy","macro_f1","pier","correct_to_wrong","disagreement"]
    for delta,g in citydf.groupby("delta_pp"):
        a={
            "delta_pp":delta,
            "mean_alpha":g.selected_alpha.mean(),
            "cities_constraint_met":int(g.constraint_met.sum()),
            "worst_accuracy_change_pp":g.accuracy_change_pp.min(),
        }
        for metric in metrics_list:
            a[f"baseline_{metric}"]=g[f"baseline_{metric}"].mean()
            a[f"selected_{metric}"]=g[f"selected_{metric}"].mean()
            a[f"change_{metric}"]=a[f"selected_{metric}"]-a[f"baseline_{metric}"]
            point,lo,hi=cluster_bootstrap(g,metric,rng)
            inf.append({
                "delta_pp":delta,"metric":metric,
                "selected_minus_baseline":point,
                "city_cluster_bootstrap_ci95_low":lo,
                "city_cluster_bootstrap_ci95_high":hi,
                "cities_improved":int((g[f"selected_{metric}"]>g[f"baseline_{metric}"]).sum()) if metric in ("accuracy","balanced_accuracy","macro_f1") else int((g[f"selected_{metric}"]<g[f"baseline_{metric}"]).sum()),
            })
        a["pier_relative_reduction"]=(a["baseline_pier"]-a["selected_pier"])/a["baseline_pier"]
        a["ctw_relative_reduction"]=(a["baseline_correct_to_wrong"]-a["selected_correct_to_wrong"])/a["baseline_correct_to_wrong"]
        agg.append(a)

    pd.DataFrame(agg).to_csv(out/"aggregate_summary.csv",index=False)
    pd.DataFrame(inf).to_csv(out/"cluster_bootstrap_ci.csv",index=False)

    # Average LCZ class effects across cities without weighting by city sample size.
    class_agg=(cdf.groupby(["delta_pp","label"],as_index=False)
        .agg(
            baseline_accuracy=("baseline_accuracy","mean"),
            selected_accuracy=("selected_accuracy","mean"),
            baseline_pier=("baseline_pier","mean"),
            selected_pier=("selected_pier","mean"),
            cities=("city","nunique"),
            total_n=("n","sum")
        ))
    class_agg["accuracy_change_pp"]=100*(class_agg.selected_accuracy-class_agg.baseline_accuracy)
    class_agg["pier_change_pp"]=100*(class_agg.selected_pier-class_agg.baseline_pier)
    class_agg.to_csv(out/"class_aggregate_summary.csv",index=False)

if __name__=="__main__":
    main()
