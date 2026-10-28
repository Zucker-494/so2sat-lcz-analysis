#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd

ROOT = Path("results")
parts = []
summaries = []

for split in ["validation", "testing"]:
    dist_file = ROOT / f"{split}_class_distribution.csv"
    sum_file = ROOT / f"{split}_summary.csv"
    if dist_file.exists():
        parts.append(pd.read_csv(dist_file))
    if sum_file.exists():
        summaries.append(pd.read_csv(sum_file))

if parts:
    all_dist = pd.concat(parts, ignore_index=True)
    combined = (
        all_dist.groupby(["city","lcz_index_1to17","lcz_code","lcz_name"], as_index=False)["n"]
        .sum()
    )
    combined.to_csv(ROOT / "city_class_distribution.csv", index=False)

if summaries:
    all_sum = pd.concat(summaries, ignore_index=True)
    all_sum.to_csv(ROOT / "split_summary.csv", index=False)

audit = {}
for split in ["validation", "testing"]:
    p = ROOT / f"{split}_audit.json"
    if p.exists():
        audit[split] = json.loads(p.read_text(encoding="utf-8"))
(ROOT / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
