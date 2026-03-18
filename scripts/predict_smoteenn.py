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


def predict_ids(model_path: Path, X: np.ndarray, ids: list[str]) -> list[str]:
    model = pickle.load(open(model_path, "rb"))
    y = model.predict(X)
    return [ids[i] for i, v in enumerate(y) if int(v) == 0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict with SMOTEENN BIG models from a single folder.")
    ap.add_argument("--features", default=os.environ.get("FEATURES_FILE", "inputs/final_all_noNaN.txt"))
    ap.add_argument("--models-dir", required=True, help="Folder containing SMOTEENN BIG .sav files")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", choices=["PTM", "Random"], required=True)
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

    X = df.to_numpy(dtype=float)
    ids = df.index.tolist()

    mdir = Path(args.models_dir)
    sgdc = mdir / f"Scaled_SGDC_{args.tag}_{args.feats}_feats.txt_over_under_sampled_finalized_model.sav"
    lr   = mdir / f"Logistic_Regression_{args.tag}_{args.feats}_feats.txt_over_under_sampled_finalized_model.sav"
    pf   = mdir / f"Partial_Fit_SGDC_{args.tag}_{args.feats}_feats.txt_over_under_sampled_finalized_model.sav"

    # Logistic + PartialFit are required
    for p in (lr, pf):
        if not p.exists():
            raise FileNotFoundError(f"Missing model file: {p}")

    out_lr = outdir / f"Predicted_TADs_Logistic_Regression_{args.tag}_BIG_{args.feats}_feats.txt"
    out_pf = outdir / f"Predicted_TADs_Partial_Fit_SGDC_{args.tag}_BIG_{args.feats}_feats.txt"
    out_lr.write_text("\n".join(predict_ids(lr, X, ids)) + "\n")
    out_pf.write_text("\n".join(predict_ids(pf, X, ids)) + "\n")

    # ScaledSGD optional (write only if model exists)
    if sgdc.exists():
        out_sg = outdir / f"Predicted_TADs_Scaled_SGDC_{args.tag}_BIG_{args.feats}_feats.txt"
        out_sg.write_text("\n".join(predict_ids(sgdc, X, ids)) + "\n")


if __name__ == "__main__":
    main()
