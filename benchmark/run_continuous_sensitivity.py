#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
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

def feature_cols(df):
    return [c for c in df.columns if c.startswith("a") and len(c) > 4]

def evaluate_family(model, train, test, family, canonical_variant, feat, city, model_name):
    model.fit(train[feat], train.label)

    preds = {}
    truth = None
    variant_metrics = []
    for variant, g in test[test.family == family].groupby("variant", sort=True):
        g = g.sort_values(["split","sample_index"]).reset_index(drop=True)
        p = model.predict(g[feat])
        y = g.label.to_numpy()
        preds[variant] = p
        if truth is None:
            truth = y
        variant_metrics.append({
            "city": city,
            "model": model_name,
            "family": family,
            "variant": variant,
            "accuracy": accuracy_score(y,p),
            "balanced_accuracy": balanced_accuracy_score(y,p),
            "macro_f1": f1_score(y,p,average="macro",zero_division=0),
        })

    variants = sorted(preds)
    if canonical_variant not in variants:
        raise ValueError(f"Missing canonical variant {canonical_variant} in {family}")

    mat = np.column_stack([preds[v] for v in variants])
    correct = mat == truth[:,None]
    exposed = correct.any(1) & (~correct).any(1)
    disagree = np.array([len(set(row.tolist())) > 1 for row in mat])
    ci = variants.index(canonical_variant)
    canon = correct[:,ci]
    alt = np.delete(correct, ci, axis=1)

    vdf = pd.DataFrame(variant_metrics)
    crow = vdf[vdf.variant == canonical_variant].iloc[0]

    summary = {
        "city": city,
        "model": model_name,
        "family": family,
        "n": len(truth),
        "canonical_accuracy": float(crow.accuracy),
        "canonical_balanced_accuracy": float(crow.balanced_accuracy),
        "canonical_macro_f1": float(crow.macro_f1),
        "mean_variant_accuracy": float(vdf.accuracy.mean()),
        "worst_variant_accuracy": float(vdf.accuracy.min()),
        "canonical_to_worst_drop_pp": 100.0 * float(crow.accuracy - vdf.accuracy.min()),
        "zoning_sensitivity": float(exposed.mean()),
        "prediction_disagreement": float(disagree.mean()),
        "correct_to_wrong": float((canon & (~alt).any(1)).mean()),
        "wrong_to_correct": float(((~canon) & alt.any(1)).mean()),
    }
    return summary, variant_metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="work/continuous_sensitivity")
    ap.add_argument("--out-dir", default="results/continuous_sensitivity")
    args = ap.parse_args()

    files = sorted(Path(args.input_dir).glob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    feat = feature_cols(df)
    cities = sorted(df.city.unique())

    rows = []
    variants = []

    # Fit the model to the canonical continuous representation (anchors a0, bandwidth 55 m).
    canonical_train = df[(df.family == "anchor_position") & (df.variant == "a0")]

    for city in cities:
        train = canonical_train[canonical_train.city != city]
        test = df[df.city == city]

        for model_name in ("hgb","extra_trees"):
            for family, canonical in (("anchor_position","a0"),("bandwidth","bw55")):
                s, vr = evaluate_family(
                    make_model(model_name), train, test, family, canonical,
                    feat, city, model_name
                )
                rows.append(s)
                variants.extend(vr)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    s = pd.DataFrame(rows)
    s.to_csv(out/"city_summary.csv", index=False)
    pd.DataFrame(variants).to_csv(out/"variant_metrics.csv", index=False)

    agg = (
        s.groupby(["model","family"], as_index=False)
        .agg(
            mean_accuracy=("canonical_accuracy","mean"),
            mean_balanced_accuracy=("canonical_balanced_accuracy","mean"),
            mean_macro_f1=("canonical_macro_f1","mean"),
            mean_zoning_sensitivity=("zoning_sensitivity","mean"),
            mean_disagreement=("prediction_disagreement","mean"),
            mean_correct_to_wrong=("correct_to_wrong","mean"),
            mean_wrong_to_correct=("wrong_to_correct","mean"),
            mean_drop_pp=("canonical_to_worst_drop_pp","mean"),
            min_city_accuracy=("canonical_accuracy","min"),
        )
    )
    agg.to_csv(out/"aggregate_summary.csv", index=False)

if __name__=="__main__":
    main()
