#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd

root = Path("results/building_qc")
items = []
for p in sorted(root.glob("*_summary.json")):
    items.append(json.loads(p.read_text(encoding="utf-8")))
if items:
    pd.DataFrame(items).to_csv(root / "city_summary.csv", index=False)

classes = []
for p in sorted(root.glob("*_class_summary.csv")):
    classes.append(pd.read_csv(p))
if classes:
    pd.concat(classes, ignore_index=True).to_csv(root / "class_summary_all.csv", index=False)
