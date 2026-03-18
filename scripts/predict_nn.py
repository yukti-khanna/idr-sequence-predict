#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def load_drop_cols(feats: int, drop_cols_dir: str) -> list[str]:
    p = Path(drop_cols_dir) / f"drop_cols_{feats}.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing drop-cols file: {p}")
    return json.loads(p.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict with a single NN model and write legacy-style output name.")
    ap.add_argument("--features", default=os.environ.get("FEATURES_FILE", "inputs/final_all_noNaN.txt"))
    ap.add_argument("--model", required=True, help="Path to NN_{TAG}_{FEATS}_feats_model.sav")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", choices=["PTM", "Random"], required=True)
    ap.add_argument("--mult", type=int, choices=[1, 2, 5, 10], required=True)
    ap.add_argument("--feats", type=int, choices=[69, 126], required=True)
    ap.add_argument("--drop-cols-dir", default="feature_sets")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features).replace([np.inf, -np.inf], np.nan).fillna(0)
    if "NAME" not in df.columns:
        raise ValueError("Feature table must contain NAME column.")
    df = df.set_index("NAME").drop(["idr_name", "CLASS"], axis=1, errors="ignore")

    drop_cols = load_drop_cols(args.feats, args.drop_cols_dir)
    df = df.drop(drop_cols, axis=1, errors="ignore")

    if df.shape[1] != args.feats:
        raise ValueError(f"After dropping, X has {df.shape[1]} features but expected {args.feats}.")

    model = pickle.load(open(args.model, "rb"))
    X = df.to_numpy(dtype=float)
    ids = df.index.tolist()

    y = model.predict(X)
    pred = [ids[i] for i, v in enumerate(y) if int(v) == 0]  # class 0 = positives

    outpath = outdir / f"Predicted_TADs_NN_{args.tag}{args.mult}_{args.feats}_feats.txt"
    outpath.write_text("\n".join(pred) + "\n")


if __name__ == "__main__":
    main()
