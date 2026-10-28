#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

SEED = 42
BOOT = 2000

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

def hard_features(df):
    return [c for c in df.columns if len(c) >= 4 and c[0]=="c" and c[1:3].isdigit() and c[3]=="_"]

def cont_features(df):
    return [c for c in df.columns if c.startswith("a") and len(c)>4]

def ordered_view_arrays(df, city, view_col, views, feat):
    out = []
    truth = None
    keys = None
    for v in views:
        g = df[(df.city==city)&(df[view_col]==v)].sort_values(["split","sample_index"]).reset_index(drop=True)
        if truth is None:
            truth = g.label.to_numpy()
            keys = g[["split","sample_index"]].copy()
        else:
            if not g[["split","sample_index"]].equals(keys):
                raise ValueError(f"Key mismatch for {city} {v}")
        out.append(g)
    return keys, truth, out

def indicators(pred_matrix, truth):
    correct = pred_matrix == truth[:,None]
    exposed = correct.any(1) & (~correct).any(1)
    disagree = np.array([len(set(row.tolist()))>1 for row in pred_matrix], dtype=bool)
    canon = correct[:,0]
    alt = correct[:,1:]
    ctw = canon & (~alt).any(1)
    wtc = (~canon) & alt.any(1)
    return {
        "pier": exposed.astype(np.int8),
        "disagreement": disagree.astype(np.int8),
        "correct_to_wrong": ctw.astype(np.int8),
        "wrong_to_correct": wtc.astype(np.int8),
        "canonical_correct": canon.astype(np.int8),
    }

def bootstrap_diff(a, b, rng, nboot=BOOT):
    # Returns b-a; here b is continuous and a is hard.
    d = b.astype(float)-a.astype(float)
    point = float(d.mean())
    n = len(d)
    vals = np.empty(nboot, dtype=float)
    for i in range(nboot):
        idx = rng.integers(0,n,size=n)
        vals[i] = d[idx].mean()
    lo, hi = np.quantile(vals,[0.025,0.975])
    return point, float(lo), float(hi)

def train_predict_hard(df, city, model_name):
    feat = hard_features(df)
    train = df[(df.city!=city)&(df.zoning=="z0")]
    model = make_model(model_name)
    model.fit(train[feat], train.label)
    views = [f"z{i}" for i in range(9)]
    keys, truth, groups = ordered_view_arrays(df, city, "zoning", views, feat)
    preds = np.column_stack([model.predict(g[feat]) for g in groups])
    return keys, truth, preds

def train_predict_cont(df, city, model_name):
    feat = cont_features(df)
    train = df[(df.city!=city)&(df.family=="anchor_position")&(df.variant=="a0")]
    model = make_model(model_name)
    model.fit(train[feat], train.label)
    views = ["a0","jitter_a","jitter_b","x_m20","x_p20","xy_m15","xy_p15","y_m20","y_p20"]
    keys, truth, groups = ordered_view_arrays(
        df[df.family=="anchor_position"], city, "variant", views, feat
    )
    preds = np.column_stack([model.predict(g[feat]) for g in groups])
    return keys, truth, preds

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--hard-dir",default="work/hard")
    ap.add_argument("--continuous-dir",default="work/continuous_sensitivity")
    ap.add_argument("--out-dir",default="results/paired_comparison")
    args=ap.parse_args()

    hard = pd.concat([pd.read_parquet(f) for f in sorted(Path(args.hard_dir).glob("*.parquet"))],ignore_index=True)
    cont = pd.concat([pd.read_parquet(f) for f in sorted(Path(args.continuous_dir).glob("*.parquet"))],ignore_index=True)

    cities=sorted(set(hard.city.unique()) & set(cont.city.unique()))
    rng=np.random.default_rng(SEED)

    summary=[]
    cirows=[]
    sample_rows=[]

    for city in cities:
        for model_name in ("hgb","extra_trees"):
            hk, hy, hp = train_predict_hard(hard,city,model_name)
            ck, cy, cp = train_predict_cont(cont,city,model_name)
            if not hk.equals(ck) or not np.array_equal(hy,cy):
                raise ValueError(f"Hard/continuous sample mismatch for {city} {model_name}")

            hi=indicators(hp,hy)
            ci=indicators(cp,cy)

            hard_acc=accuracy_score(hy,hp[:,0])
            cont_acc=accuracy_score(cy,cp[:,0])

            summary.append({
                "city":city,
                "model":model_name,
                "n":len(hy),
                "hard_accuracy":hard_acc,
                "continuous_accuracy":cont_acc,
                "accuracy_diff_pp":100*(cont_acc-hard_acc),
                "hard_balanced_accuracy":balanced_accuracy_score(hy,hp[:,0]),
                "continuous_balanced_accuracy":balanced_accuracy_score(cy,cp[:,0]),
                "hard_macro_f1":f1_score(hy,hp[:,0],average="macro",zero_division=0),
                "continuous_macro_f1":f1_score(cy,cp[:,0],average="macro",zero_division=0),
                "hard_pier":hi["pier"].mean(),
                "continuous_pier":ci["pier"].mean(),
                "hard_disagreement":hi["disagreement"].mean(),
                "continuous_disagreement":ci["disagreement"].mean(),
                "hard_correct_to_wrong":hi["correct_to_wrong"].mean(),
                "continuous_correct_to_wrong":ci["correct_to_wrong"].mean(),
            })

            for metric in ("canonical_correct","pier","disagreement","correct_to_wrong","wrong_to_correct"):
                point,lo,high=bootstrap_diff(hi[metric],ci[metric],rng)
                cirows.append({
                    "city":city,
                    "model":model_name,
                    "metric":metric,
                    "continuous_minus_hard":point,
                    "ci95_low":lo,
                    "ci95_high":high,
                    "favors_continuous_if":"negative" if metric!="canonical_correct" else "positive",
                })

            tmp=hk.copy()
            tmp["city"]=city
            tmp["model"]=model_name
            tmp["label"]=hy
            for m in ("pier","disagreement","correct_to_wrong","wrong_to_correct","canonical_correct"):
                tmp[f"hard_{m}"]=hi[m]
                tmp[f"continuous_{m}"]=ci[m]
            sample_rows.append(tmp)

    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    s=pd.DataFrame(summary)
    s.to_csv(out/"city_model_summary.csv",index=False)
    pd.DataFrame(cirows).to_csv(out/"paired_bootstrap_ci.csv",index=False)
    pd.concat(sample_rows,ignore_index=True).to_csv(out/"sample_flags.csv",index=False)

    agg=(s.groupby("model",as_index=False)
         .agg(
             hard_accuracy=("hard_accuracy","mean"),
             continuous_accuracy=("continuous_accuracy","mean"),
             mean_accuracy_diff_pp=("accuracy_diff_pp","mean"),
             hard_pier=("hard_pier","mean"),
             continuous_pier=("continuous_pier","mean"),
             hard_disagreement=("hard_disagreement","mean"),
             continuous_disagreement=("continuous_disagreement","mean"),
             hard_correct_to_wrong=("hard_correct_to_wrong","mean"),
             continuous_correct_to_wrong=("continuous_correct_to_wrong","mean"),
         ))
    agg["pier_relative_reduction"]=(agg.hard_pier-agg.continuous_pier)/agg.hard_pier
    agg["disagreement_relative_reduction"]=(agg.hard_disagreement-agg.continuous_disagreement)/agg.hard_disagreement
    agg["correct_to_wrong_relative_reduction"]=(agg.hard_correct_to_wrong-agg.continuous_correct_to_wrong)/agg.hard_correct_to_wrong
    agg.to_csv(out/"aggregate_summary.csv",index=False)

if __name__=="__main__":
    main()
