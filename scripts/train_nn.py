#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
    ap = argparse.ArgumentParser(description="Train NN/MLP models and save with standardized filename.")
    ap.add_argument("--train", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", choices=["PTM", "Random"], required=True)
    ap.add_argument("--feats", type=int, choices=[69, 126], required=True)
    ap.add_argument("--drop-cols-dir", default="feature_sets")

    # MLP params
    ap.add_argument("--hidden", default="128,64", help="Comma-separated hidden layer sizes, e.g. 128,64")
    ap.add_argument("--alpha", type=float, default=0.0001)
    ap.add_argument("--max-iter", type=int, default=400)
    ap.add_argument("--random-state", type=int, default=0)

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

    hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip())

    mlp = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", MLPClassifier(
            hidden_layer_sizes=hidden,
            alpha=args.alpha,
            max_iter=args.max_iter,
            random_state=args.random_state,
            early_stopping=True,
            n_iter_no_change=20,
        ))
    ])

    mlp.fit(X, y)

    outpath = outdir / f"NN_{args.tag}_{args.feats}_feats_model.sav"
    save_model(mlp, outpath)
    print("Saved:", outpath)


if __name__ == "__main__":
    main()
