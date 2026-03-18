#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# SMOTEENN
from imblearn.combine import SMOTEENN


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
    df = df.drop(["idr_name"], axis=1, errors="ignore")
    return df


def save_model(model, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("wb") as f:
        pickle.dump(model, f)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train SMOTEENN-resampled models (ScaledSGD, LogReg, PartialFitSGD) and save with standardized filenames."
    )
    ap.add_argument("--train", required=True, help="Training CSV with CLASS column and feature columns.")
    ap.add_argument("--outdir", required=True, help="Where to save .sav files.")
    ap.add_argument("--tag", choices=["PTM", "Random"], required=True)
    ap.add_argument("--feats", type=int, choices=[69, 126], required=True)
    ap.add_argument("--drop-cols-dir", default="feature_sets")
    ap.add_argument("--random-state", type=int, default=0)

    # SMOTEENN knobs
    ap.add_argument("--smote-k", type=int, default=5, help="SMOTE k_neighbors")
    ap.add_argument("--enn-k", type=int, default=3, help="ENN n_neighbors")

    # Classifier knobs
    ap.add_argument("--max-iter", type=int, default=2000)

    ap.add_argument("--lr-C", type=float, default=1.0)
    ap.add_argument("--lr-class-weight", default=None, choices=[None, "balanced"])

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

    # Scale first, then resample in scaled space (keeps ranges stable for SGD/LogReg; matches your MinMaxScaler pipelines)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(Xdf.to_numpy(dtype=float))

    smoteenn = SMOTEENN(
        random_state=args.random_state,
        smote=None,  # default SMOTE
        enn=None,    # default ENN
    )
    # If you want to tune k-neighbors explicitly:
    # from imblearn.over_sampling import SMOTE
    # from imblearn.under_sampling import EditedNearestNeighbours
    # smoteenn = SMOTEENN(
    #   random_state=args.random_state,
    #   smote=SMOTE(k_neighbors=args.smote_k, random_state=args.random_state),
    #   enn=EditedNearestNeighbours(n_neighbors=args.enn_k)
    # )

    X_res, y_res = smoteenn.fit_resample(X_scaled, y)

    # Build models as pipelines with a MinMaxScaler step (so prediction expects 69/126 and uses MinMaxScaler)
    # We fit a fresh scaler inside each pipeline on the resampled data so the saved model is self-contained.
    scaled_sgdc = Pipeline([
        ("scaler", MinMaxScaler()),
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

    logreg = Pipeline([
        ("scaler", MinMaxScaler()),
        ("clf", LogisticRegression(
            C=args.lr_C,
            max_iter=args.max_iter,
            class_weight=args.lr_class_weight,
            solver="lbfgs",
            random_state=args.random_state,
        ))
    ])

    partial_fit_sgdc = Pipeline([
        ("scaler", MinMaxScaler()),
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

    scaled_sgdc.fit(X_res, y_res)
    logreg.fit(X_res, y_res)
    partial_fit_sgdc.fit(X_res, y_res)

    f_sgdc = outdir / f"Scaled_SGDC_{args.tag}_{args.feats}_feats.txt_over_under_sampled_finalized_model.sav"
    f_lr   = outdir / f"Logistic_Regression_{args.tag}_{args.feats}_feats.txt_over_under_sampled_finalized_model.sav"
    f_pf   = outdir / f"Partial_Fit_SGDC_{args.tag}_{args.feats}_feats.txt_over_under_sampled_finalized_model.sav"

    save_model(scaled_sgdc, f_sgdc)
    save_model(logreg, f_lr)
    save_model(partial_fit_sgdc, f_pf)

    print("Saved:")
    print(" ", f_sgdc)
    print(" ", f_lr)
    print(" ", f_pf)


if __name__ == "__main__":
    main()
