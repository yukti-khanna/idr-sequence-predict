#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def load_drop_cols(feats: int, drop_cols_dir: str) -> list[str]:
    p = Path(drop_cols_dir) / f"drop_cols_{feats}.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing drop-cols file: {p}")
    return json.loads(p.read_text())


def load_training_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    if "CLASS" not in df.columns:
        raise ValueError("Training table must contain CLASS column (0/1).")
    if "NAME" in df.columns:
        df = df.set_index("NAME")
    # remove non-feature columns if present
    df = df.drop(["idr_name"], axis=1, errors="ignore")
    return df


def save_model(model, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("wb") as f:
        pickle.dump(model, f)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train linear models and save with standardized filenames.")
    ap.add_argument("--train", required=True, help="Training CSV with CLASS column and feature columns.")
    ap.add_argument("--outdir", required=True, help="Where to save .sav files (location is free).")
    ap.add_argument("--tag", choices=["PTM", "Random"], required=True)
    ap.add_argument("--feats", type=int, choices=[69, 126], required=True)
    ap.add_argument("--drop-cols-dir", default="feature_sets", help="Contains drop_cols_69.json and drop_cols_126.json")
    ap.add_argument("--random-state", type=int, default=0)
    ap.add_argument("--max-iter", type=int, default=2000)

    # Logistic Regression hyperparams
    ap.add_argument("--lr-C", type=float, default=1.0)
    ap.add_argument("--lr-class-weight", default=None, choices=[None, "balanced"])

    # SGD / PartialFit-ish hyperparams
    ap.add_argument("--sgd-alpha", type=float, default=0.0001)
    ap.add_argument("--sgd-penalty", default="l2", choices=["l2", "l1", "elasticnet"])
    ap.add_argument("--sgd-l1-ratio", type=float, default=0.15)

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    drop_cols = load_drop_cols(args.feats, args.drop_cols_dir)
    df = load_training_table(args.train)

    y = df["CLASS"].astype(int).values
    Xdf = df.drop(["CLASS"], axis=1, errors="ignore")
    Xdf = Xdf.drop(drop_cols, axis=1, errors="ignore")

    if Xdf.shape[1] != args.feats:
        raise ValueError(
            f"After dropping, X has {Xdf.shape[1]} features but expected {args.feats}. "
            f"Check drop_cols_{args.feats}.json and the train table columns."
        )

    X = Xdf.to_numpy(dtype=float)

    # 1) Scaled SGDC (we standardize features)
    scaled_sgdc = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", SGDClassifier(
            loss="log_loss",
            alpha=args.sgd_alpha,
            penalty=args.sgd_penalty,
            l1_ratio=args.sgd_l1_ratio if args.sgd_penalty == "elasticnet" else 0.0,
            random_state=args.random_state,
            max_iter=args.max_iter,
            tol=1e-3,
        ))
    ])

    # 2) Logistic Regression
    logreg = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            C=args.lr_C,
            max_iter=args.max_iter,
            class_weight=args.lr_class_weight,
            solver="lbfgs",
            n_jobs=None,
            random_state=args.random_state,
        ))
    ])

    # 3) Partial-fit style SGDC (same classifier; “partial fit” training is a training procedure,
    # but final model is just an SGDClassifier. We keep a separate saved model name for compatibility.)
    partial_fit_sgdc = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", SGDClassifier(
            loss="log_loss",
            alpha=args.sgd_alpha,
            penalty=args.sgd_penalty,
            l1_ratio=args.sgd_l1_ratio if args.sgd_penalty == "elasticnet" else 0.0,
            random_state=args.random_state,
            max_iter=args.max_iter,
            tol=1e-3,
        ))
    ])

    # Fit
    scaled_sgdc.fit(X, y)
    logreg.fit(X, y)
    partial_fit_sgdc.fit(X, y)

    # Save with your exact naming convention
    f_sgdc = outdir / f"Scaled_SGDC_{args.tag}_{args.feats}_feats.txt_model.sav"
    f_lr   = outdir / f"Logistic_Regression_{args.tag}_{args.feats}_feats.txt_model.sav"
    f_pf   = outdir / f"Partial_Fit_SGDC_{args.tag}_{args.feats}_feats.txt_model.sav"

    save_model(scaled_sgdc, f_sgdc)
    save_model(logreg, f_lr)
    save_model(partial_fit_sgdc, f_pf)

    print("Saved:")
    print(" ", f_sgdc)
    print(" ", f_lr)
    print(" ", f_pf)


if __name__ == "__main__":
    main()
