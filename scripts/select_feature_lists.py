#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def write_list(path: Path, feats: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(feats) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="CSV from run_feature_importance.py")
    ap.add_argument("--outdir", default="feature_sets")
    ap.add_argument("--thresholds", nargs="*", type=float, default=[0.90, 0.95])
    args = ap.parse_args()

    df = pd.read_csv(args.scores)
    if not {"feature","importance","cum_fraction"}.issubset(df.columns):
        raise ValueError("Expected columns: feature, importance, cum_fraction")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for t in args.thresholds:
        sub = df[df["cum_fraction"] <= t]["feature"].astype(str).tolist()
        # include first feature that crosses threshold (so you truly reach >=t)
        if len(sub) < len(df):
            sub.append(str(df.loc[len(sub), "feature"]))
        write_list(outdir / f"selected_feats_cum{int(t*100)}.txt", sub)
        print(f"Wrote selected_feats_cum{int(t*100)}.txt  n={len(sub)}")

if __name__ == "__main__":
    main()
