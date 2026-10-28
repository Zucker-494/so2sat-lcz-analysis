#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

CELL_AREA = 80.0 * 80.0

def feature_columns(df):
    return [
        c for c in df.columns
        if len(c) >= 4 and c[0] == "c" and c[1:3].isdigit() and c[3] == "_"
    ]

def invariant_features(g):
    """Create four whole-patch morphology descriptors from canonical 4x4 cells."""
    g = g.copy()
    count_cols = [f"c{i:02d}_count_density" for i in range(16)]
    cover_cols = [f"c{i:02d}_footprint_density" for i in range(16)]
    area_cols = [f"c{i:02d}_mean_area" for i in range(16)]
    comp_cols = [f"c{i:02d}_mean_compactness" for i in range(16)]

    counts = g[count_cols].to_numpy(float) * CELL_AREA / 10000.0
    footprint_area = g[cover_cols].to_numpy(float) * CELL_AREA
    mean_area = g[area_cols].to_numpy(float)
    mean_comp = g[comp_cols].to_numpy(float)

    n = counts.sum(axis=1)
    fp = footprint_area.sum(axis=1)

    # Count-weighted object summaries. Cells with no buildings contribute zero.
    weighted_comp = (mean_comp * counts).sum(axis=1)
    out = pd.DataFrame({
        "patch_count_density": n / (320.0 * 320.0) * 10000.0,
        "patch_footprint_density": fp / (320.0 * 320.0),
        "patch_mean_area": np.divide(fp, n, out=np.zeros_like(fp), where=n > 0),
        "patch_mean_compactness": np.divide(
            weighted_comp, n, out=np.zeros_like(weighted_comp), where=n > 0
        ),
    }, index=g.index)
    return out

def make_model(name):
    if name == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=42,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    raise ValueError(name)

def evaluate_zoning_model(model, train, test, feat_cols, strategy, model_name, test_city):
    model.fit(train[feat_cols], train.label)

    pred_by_z = {}
    truth = None
    zoning_metrics = []

    for z, g in test.groupby("zoning", sort=True):
        g = g.sort_values(["split", "sample_index"]).reset_index(drop=True)
        pred = model.predict(g[feat_cols])
        y = g.label.to_numpy()
        pred_by_z[z] = pred
        if truth is None:
            truth = y
        zoning_metrics.append({
            "city": test_city,
            "model": model_name,
            "strategy": strategy,
            "zoning": z,
            "accuracy": accuracy_score(y, pred),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        })

    zonings = sorted(pred_by_z)
    mat = np.column_stack([pred_by_z[z] for z in zonings])
    correct = mat == truth[:, None]
    exposed = correct.any(axis=1) & (~correct).any(axis=1)
    disagree = np.apply_along_axis(lambda x: len(set(x.tolist())) > 1, 1, mat)
    z0_idx = zonings.index("z0")
    canonical_correct = correct[:, z0_idx]
    any_alt_wrong = (~np.delete(correct, z0_idx, axis=1)).any(axis=1)
    any_alt_correct = np.delete(correct, z0_idx, axis=1).any(axis=1)

    zdf = pd.DataFrame(zoning_metrics)
    z0 = zdf[zdf.zoning == "z0"].iloc[0]
    return {
        "city": test_city,
        "model": model_name,
        "strategy": strategy,
        "canonical_accuracy": float(z0.accuracy),
        "canonical_balanced_accuracy": float(z0.balanced_accuracy),
        "canonical_macro_f1": float(z0.macro_f1),
        "mean_zoning_accuracy": float(zdf.accuracy.mean()),
        "worst_zoning_accuracy": float(zdf.accuracy.min()),
        "pier": float(exposed.mean()),
        "prediction_disagreement": float(disagree.mean()),
        "canonical_correct_any_alt_wrong_rate_all": float(
            (canonical_correct & any_alt_wrong).mean()
        ),
        "canonical_wrong_any_alt_correct_rate_all": float(
            ((~canonical_correct) & any_alt_correct).mean()
        ),
    }, zoning_metrics

def evaluate_invariant(model, train_z0, test_z0, model_name, test_city):
    Xtr = invariant_features(train_z0)
    Xte = invariant_features(test_z0)
    ytr = train_z0.label.to_numpy()
    yte = test_z0.label.to_numpy()
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return {
        "city": test_city,
        "model": model_name,
        "strategy": "whole_patch_invariant",
        "canonical_accuracy": accuracy_score(yte, pred),
        "canonical_balanced_accuracy": balanced_accuracy_score(yte, pred),
        "canonical_macro_f1": f1_score(yte, pred, average="macro", zero_division=0),
        "mean_zoning_accuracy": accuracy_score(yte, pred),
        "worst_zoning_accuracy": accuracy_score(yte, pred),
        "pier": 0.0,
        "prediction_disagreement": 0.0,
        "canonical_correct_any_alt_wrong_rate_all": 0.0,
        "canonical_wrong_any_alt_correct_rate_all": 0.0,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="work/features")
    ap.add_argument("--out-dir", default="results/baseline_benchmark")
    args = ap.parse_args()

    files = sorted(Path(args.input_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No feature parquet files found in {args.input_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    feat_cols = feature_columns(df)

    summary = []
    per_zoning = []
    cities = sorted(df.city.unique())

    for test_city in cities:
        test = df[df.city == test_city].copy()
        train_all = df[df.city != test_city].copy()
        train_z0 = train_all[train_all.zoning == "z0"].copy()
        test_z0 = test[test.zoning == "z0"].sort_values(["split", "sample_index"]).reset_index(drop=True)

        for model_name in ("hgb", "extra_trees"):
            # Baseline 1: canonical zoning only.
            row, zrows = evaluate_zoning_model(
                make_model(model_name),
                train_z0,
                test,
                feat_cols,
                "canonical_only",
                model_name,
                test_city,
            )
            summary.append(row)
            per_zoning.extend(zrows)

            # Baseline 2: obvious multi-zoning augmentation.
            row, zrows = evaluate_zoning_model(
                make_model(model_name),
                train_all,
                test,
                feat_cols,
                "multi_zoning_augmentation",
                model_name,
                test_city,
            )
            summary.append(row)
            per_zoning.extend(zrows)

            # Baseline 3: completely partition-invariant whole-patch morphology.
            summary.append(
                evaluate_invariant(
                    make_model(model_name),
                    train_z0,
                    test_z0,
                    model_name,
                    test_city,
                )
            )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    s = pd.DataFrame(summary)
    s.to_csv(out / "city_model_summary.csv", index=False)
    pd.DataFrame(per_zoning).to_csv(out / "per_zoning_metrics.csv", index=False)

    aggregate = (
        s.groupby(["model", "strategy"], as_index=False)
        .agg(
            mean_accuracy=("canonical_accuracy", "mean"),
            mean_balanced_accuracy=("canonical_balanced_accuracy", "mean"),
            mean_macro_f1=("canonical_macro_f1", "mean"),
            mean_pier=("pier", "mean"),
            mean_disagreement=("prediction_disagreement", "mean"),
            mean_correct_to_wrong=("canonical_correct_any_alt_wrong_rate_all", "mean"),
            mean_wrong_to_correct=("canonical_wrong_any_alt_correct_rate_all", "mean"),
            min_city_accuracy=("canonical_accuracy", "min"),
        )
    )
    aggregate.to_csv(out / "aggregate_summary.csv", index=False)

if __name__ == "__main__":
    main()
