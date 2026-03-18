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


def load_features_table(features_path: str) -> pd.DataFrame:
    df = pd.read_csv(features_path)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    if "NAME" not in df.columns:
        raise ValueError("Feature table must contain a NAME column.")
    df = df.set_index("NAME")
    df = df.drop(["idr_name", "CLASS"], axis=1, errors="ignore")
    return df


def predict_ids(model_path: Path, X: np.ndarray, ids: list[str]) -> list[str]:
    model = pickle.load(open(model_path, "rb"))
    y = model.predict(X)
    return [ids[i] for i, v in enumerate(y) if int(v) == 0]  # class 0 = positives


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict with 3 linear models from a single bucket folder.")
    ap.add_argument("--features", default=os.environ.get("FEATURES_FILE", "inputs/final_all_noNaN.txt"))
    ap.add_argument("--models-dir", required=True, help="Bucket folder containing the 3 linear .sav files")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", choices=["PTM", "Random"], required=True)
    ap.add_argument("--mult", type=int, choices=[1, 2, 5, 10], required=True)
    ap.add_argument("--feats", type=int, choices=[69, 126], required=True)
    ap.add_argument("--drop-cols-dir", default="feature_sets")
    ap.add_argument("--model-style", choices=["tagged", "untagged"], default="tagged")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_features_table(args.features)
    drop_cols = load_drop_cols(args.feats, args.drop_cols_dir)
    df = df.drop(drop_cols, axis=1, errors="ignore")

    if df.shape[1] != args.feats:
        raise ValueError(
            f"After dropping, X has {df.shape[1]} features but expected {args.feats}. "
            f"Check drop_cols_{args.feats}.json and feature table columns."
        )

    ids = df.index.tolist()
    X = df.to_numpy(dtype=float)

    models_dir = Path(args.models_dir)

    if args.model_style == "tagged":
        sgdc = models_dir / f"Scaled_SGDC_{args.tag}_{args.feats}_feats.txt_model.sav"
        lr = models_dir / f"Logistic_Regression_{args.tag}_{args.feats}_feats.txt_model.sav"
        pf = models_dir / f"Partial_Fit_SGDC_{args.tag}_{args.feats}_feats.txt_model.sav"
    else:
        sgdc = models_dir / f"Scaled_SGDC_{args.feats}_feats.txt_over_under_sampled_finalized_model.sav"
        lr = models_dir / f"Logistic_Regression_{args.feats}_feats.txt_over_under_sampled_finalized_model.sav"
        pf = models_dir / f"Partial_Fit_SGDC_{args.feats}_feats.txt_over_under_sampled_finalized_model.sav"

    for p in (sgdc, lr, pf):
        if not p.exists():
            raise FileNotFoundError(f"Missing model file: {p}")

    out1 = outdir / f"Predicted_TADs_Scaled_SGDC_{args.tag}{args.mult}_{args.feats}_feats.txt"
    out2 = outdir / f"Predicted_TADs_Logistic_Regression_{args.tag}{args.mult}_{args.feats}_feats.txt"
    out3 = outdir / f"Predicted_TADs_Partial_Fit_SGDC_{args.tag}{args.mult}_{args.feats}_feats.txt"

    out1.write_text("\n".join(predict_ids(sgdc, X, ids)) + "\n")
    out2.write_text("\n".join(predict_ids(lr, X, ids)) + "\n")
    out3.write_text("\n".join(predict_ids(pf, X, ids)) + "\n")


if __name__ == "__main__":
    main()
