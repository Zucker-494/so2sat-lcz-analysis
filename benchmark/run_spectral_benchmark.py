#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="work/spectral")
    ap.add_argument("--out-dir", default="results/spectral_basis")
    args = ap.parse_args()

    files = sorted(Path(args.input_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(args.input_dir)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    feat = [c for c in df.columns if c.startswith("s_")]
    cities = sorted(df.city.unique())
    rows = []

    for test_city in cities:
        tr = df[df.city != test_city]
        te = df[df.city == test_city]
        for model_name in ("hgb", "extra_trees"):
            model = make_model(model_name)
            model.fit(tr[feat], tr.label)
            pred = model.predict(te[feat])
            rows.append({
                "city": test_city,
                "model": model_name,
                "accuracy": accuracy_score(te.label, pred),
                "balanced_accuracy": balanced_accuracy_score(te.label, pred),
                "macro_f1": f1_score(te.label, pred, average="macro", zero_division=0),
            })

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    s = pd.DataFrame(rows)
    s.to_csv(out / "city_summary.csv", index=False)
    agg = (
        s.groupby("model", as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            min_city_accuracy=("accuracy", "min"),
        )
    )
    agg.to_csv(out / "aggregate_summary.csv", index=False)

if __name__ == "__main__":
    main()
