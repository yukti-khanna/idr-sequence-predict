#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Labeled TSV with CLASS column")
    ap.add_argument("--out", required=True, help="Output CSV with feature importance + cumulative")
    ap.add_argument("--class-col", default="CLASS")
    ap.add_argument("--name-col", default="NAME")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.train, sep="\t").replace([np.inf,-np.inf], np.nan).fillna(0)

    y = df[args.class_col].astype(int).values
    X = df.drop([args.class_col], axis=1, errors="ignore")
    if args.name_col in X.columns:
        X = X.drop([args.name_col], axis=1)

    # Only numeric
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    feat_names = X.columns.tolist()

    model = ExtraTreesClassifier(n_estimators=500, random_state=args.seed, n_jobs=-1)
    model.fit(X.values, y)

    imp = pd.DataFrame({"feature": feat_names, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    tot = imp["importance"].sum()
    imp["cum_fraction"] = imp["importance"].cumsum() / (tot if tot else 1.0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imp.to_csv(out, index=False)
    print("Wrote", out)

if __name__ == "__main__":
    main()
