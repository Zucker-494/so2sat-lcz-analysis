#!/usr/bin/env python3
from pathlib import Path
import itertools
import numpy as np
import pandas as pd

INFILE = Path("results/final_hgb_confirmation/city_summary.csv")
OUTDIR = Path("results/exact_city_inference")
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INFILE)
df = df[df["delta_pp"] == 0.5].copy()

metrics = [
    ("accuracy", "higher"),
    ("balanced_accuracy", "higher"),
    ("macro_f1", "higher"),
    ("pier", "lower"),
    ("correct_to_wrong", "lower"),
    ("disagreement", "lower"),
]

def exact_sign_test(diff, direction):
    # Zeros are discarded for the sign test.
    nz = diff[np.abs(diff) > 1e-15]
    n = len(nz)
    if n == 0:
        return np.nan, 0, 0
    if direction == "higher":
        k = int((nz > 0).sum())
    else:
        k = int((nz < 0).sum())
    # One-sided exact Binomial(n, 0.5): P(X >= k)
    probs = [np.math.comb(n, j) for j in range(k, n + 1)]
    p_one = sum(probs) / (2 ** n)
    return p_one, k, n

def exact_signflip_mean_test(diff, direction):
    # Exact paired randomization test under sign symmetry.
    obs = diff.mean()
    if direction == "lower":
        obs_stat = -obs
    else:
        obs_stat = obs
    vals = []
    for signs in itertools.product([-1.0, 1.0], repeat=len(diff)):
        perm = diff * np.asarray(signs)
        stat = perm.mean()
        if direction == "lower":
            stat = -stat
        vals.append(stat)
    vals = np.asarray(vals)
    p_one = float((vals >= obs_stat - 1e-15).mean())
    p_two = float((np.abs(vals) >= abs(obs_stat) - 1e-15).mean())
    return obs, p_one, p_two

rows = []
city_rows = []

for metric, direction in metrics:
    base = df[f"baseline_{metric}"].to_numpy(float)
    sel = df[f"selected_{metric}"].to_numpy(float)
    diff = sel - base

    p_sign, favorable, n_nonzero = exact_sign_test(diff, direction)
    mean_diff, p_perm_one, p_perm_two = exact_signflip_mean_test(diff, direction)

    rows.append({
        "metric": metric,
        "direction_favorable": direction,
        "n_cities": len(diff),
        "n_nonzero_for_sign_test": n_nonzero,
        "cities_favorable": favorable,
        "mean_selected_minus_baseline": mean_diff,
        "exact_sign_test_p_one_sided": p_sign,
        "exact_signflip_mean_p_one_sided": p_perm_one,
        "exact_signflip_mean_p_two_sided": p_perm_two,
    })

    for city, d in zip(df["city"], diff):
        city_rows.append({
            "city": city,
            "metric": metric,
            "selected_minus_baseline": d,
            "favorable": bool(d > 0 if direction == "higher" else d < 0),
        })

pd.DataFrame(rows).to_csv(OUTDIR / "exact_city_inference.csv", index=False)
pd.DataFrame(city_rows).to_csv(OUTDIR / "city_metric_differences.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
