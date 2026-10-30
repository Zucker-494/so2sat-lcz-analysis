#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

SEED=42
BOOT=2000

def make_model(name):
    if name=="hgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.06,max_iter=300,max_leaf_nodes=31,
            l2_regularization=1.0,class_weight="balanced",random_state=42
        )
    return ExtraTreesClassifier(
        n_estimators=400,max_features="sqrt",min_samples_leaf=2,
        class_weight="balanced",random_state=42,n_jobs=-1
    )

def hcols(df):
    return [c for c in df.columns if len(c)>=4 and c[0]=="c" and c[1:3].isdigit() and c[3]=="_"]

def ncols(df):
    return [c for c in df.columns if len(c)>=4 and c[0]=="n" and c[1:3].isdigit() and c[3]=="_"]

def ordered(df,city,col,views):
    groups=[]; keys=None; y=None
    for v in views:
        g=df[(df.city==city)&(df[col]==v)].sort_values(["split","sample_index"]).reset_index(drop=True)
        if keys is None:
            keys=g[["split","sample_index"]].copy()
            y=g.label.to_numpy()
        elif not g[["split","sample_index"]].equals(keys):
            raise ValueError(f"key mismatch {city} {v}")
        groups.append(g)
    return keys,y,groups

def flags(pred,y):
    corr=pred==y[:,None]
    zoning_sensitivity=corr.any(1)&(~corr).any(1)
    dis=np.array([len(set(r.tolist()))>1 for r in pred])
    canon=corr[:,0]; alt=corr[:,1:]
    return {
        "canonical_correct":canon.astype(np.int8),
        "zoning_sensitivity":zoning_sensitivity.astype(np.int8),
        "disagreement":dis.astype(np.int8),
        "correct_to_wrong":(canon&(~alt).any(1)).astype(np.int8),
        "wrong_to_correct":((~canon)&alt.any(1)).astype(np.int8),
    }

def predict_rep(df,city,model_name,rep):
    if rep=="hard":
        feat=hcols(df); col="zoning"; views=[f"z{i}" for i in range(9)]; canonical="z0"
    else:
        feat=ncols(df); col="variant"
        views=["a0","jitter_a","jitter_b","x_m20","x_p20","xy_m15","xy_p15","y_m20","y_p20"]
        canonical="a0"
    train=df[(df.city!=city)&(df[col]==canonical)]
    model=make_model(model_name)
    model.fit(train[feat],train.label)
    keys,y,groups=ordered(df,city,col,views)
    pred=np.column_stack([model.predict(g[feat]) for g in groups])
    return keys,y,pred

def bootdiff(a,b,rng):
    d=b.astype(float)-a.astype(float)
    vals=np.empty(BOOT)
    n=len(d)
    for i in range(BOOT):
        idx=rng.integers(0,n,n)
        vals[i]=d[idx].mean()
    lo,hi=np.quantile(vals,[0.025,0.975])
    return float(d.mean()),float(lo),float(hi)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--hard-dir",default="work/hard")
    ap.add_argument("--soft-dir",default="work/normalized_soft")
    ap.add_argument("--out-dir",default="results/normalized_soft_comparison")
    args=ap.parse_args()

    hard=pd.concat([pd.read_parquet(f) for f in sorted(Path(args.hard_dir).glob("*.parquet"))],ignore_index=True)
    soft=pd.concat([pd.read_parquet(f) for f in sorted(Path(args.soft_dir).glob("*.parquet"))],ignore_index=True)
    cities=sorted(set(hard.city.unique())&set(soft.city.unique()))
    rng=np.random.default_rng(SEED)
    rows=[]; cis=[]

    for city in cities:
        for model_name in ("hgb","extra_trees"):
            hk,hy,hp=predict_rep(hard,city,model_name,"hard")
            sk,sy,sp=predict_rep(soft,city,model_name,"soft")
            if not hk.equals(sk) or not np.array_equal(hy,sy):
                raise ValueError(f"sample mismatch {city} {model_name}")
            hf=flags(hp,hy); sf=flags(sp,sy)

            rows.append({
                "city":city,"model":model_name,"n":len(hy),
                "hard_accuracy":accuracy_score(hy,hp[:,0]),
                "soft_accuracy":accuracy_score(sy,sp[:,0]),
                "hard_balanced_accuracy":balanced_accuracy_score(hy,hp[:,0]),
                "soft_balanced_accuracy":balanced_accuracy_score(sy,sp[:,0]),
                "hard_macro_f1":f1_score(hy,hp[:,0],average="macro",zero_division=0),
                "soft_macro_f1":f1_score(sy,sp[:,0],average="macro",zero_division=0),
                "hard_zoning_sensitivity":hf["zoning_sensitivity"].mean(),"soft_zoning_sensitivity":sf["zoning_sensitivity"].mean(),
                "hard_disagreement":hf["disagreement"].mean(),"soft_disagreement":sf["disagreement"].mean(),
                "hard_correct_to_wrong":hf["correct_to_wrong"].mean(),"soft_correct_to_wrong":sf["correct_to_wrong"].mean(),
            })
            for m in hf:
                p,lo,hi=bootdiff(hf[m],sf[m],rng)
                cis.append({"city":city,"model":model_name,"metric":m,
                            "soft_minus_hard":p,"ci95_low":lo,"ci95_high":hi})

    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    s=pd.DataFrame(rows)
    s["accuracy_diff_pp"]=100*(s.soft_accuracy-s.hard_accuracy)
    s.to_csv(out/"city_model_summary.csv",index=False)
    pd.DataFrame(cis).to_csv(out/"paired_bootstrap_ci.csv",index=False)

    agg=(s.groupby("model",as_index=False).agg(
        hard_accuracy=("hard_accuracy","mean"),soft_accuracy=("soft_accuracy","mean"),
        hard_balanced_accuracy=("hard_balanced_accuracy","mean"),soft_balanced_accuracy=("soft_balanced_accuracy","mean"),
        hard_macro_f1=("hard_macro_f1","mean"),soft_macro_f1=("soft_macro_f1","mean"),
        hard_zoning_sensitivity=("hard_zoning_sensitivity","mean"),soft_zoning_sensitivity=("soft_zoning_sensitivity","mean"),
        hard_disagreement=("hard_disagreement","mean"),soft_disagreement=("soft_disagreement","mean"),
        hard_correct_to_wrong=("hard_correct_to_wrong","mean"),soft_correct_to_wrong=("soft_correct_to_wrong","mean")
    ))
    agg["accuracy_diff_pp"]=100*(agg.soft_accuracy-agg.hard_accuracy)
    agg["zoning_sensitivity_relative_reduction"]=(agg.hard_zoning_sensitivity-agg.soft_zoning_sensitivity)/agg.hard_zoning_sensitivity
    agg["disagreement_relative_reduction"]=(agg.hard_disagreement-agg.soft_disagreement)/agg.hard_disagreement
    agg["correct_to_wrong_relative_reduction"]=(agg.hard_correct_to_wrong-agg.soft_correct_to_wrong)/agg.hard_correct_to_wrong
    agg.to_csv(out/"aggregate_summary.csv",index=False)

if __name__=="__main__":
    main()
