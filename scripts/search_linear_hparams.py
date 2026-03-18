#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, fbeta_score

import json
from hparams_utils import load_json, save_json, deep_get, deep_set

def load_drop_cols(feats: int, drop_dir: str) -> list[str]:
    p = Path(drop_dir) / f"drop_cols_{feats}.json"
    return json.loads(p.read_text())


def load_bucket_table(path: str, feats: int, drop_dir: str):
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).fillna(0)
    if "CLASS" not in df.columns:
        raise ValueError(f"{path} missing CLASS")
    if "NAME" in df.columns:
        df = df.set_index("NAME")
    df = df.drop(["idr_name"], axis=1, errors="ignore")

    y = df["CLASS"].astype(int).values
    Xdf = df.drop(["CLASS"], axis=1, errors="ignore")

    drop_cols = load_drop_cols(feats, drop_dir)
    Xdf = Xdf.drop(drop_cols, axis=1, errors="ignore")

    if Xdf.shape[1] != feats:
        raise ValueError(f"{path}: after dropping -> {Xdf.shape[1]} cols, expected {feats}")

    X = Xdf.to_numpy(dtype=float)
    return X, y


def score_objective(obj: str, y_true, y_pred, beta: float) -> float:
    if obj == "f1":
        return f1_score(y_true, y_pred)
    if obj == "max_recall":
        return recall_score(y_true, y_pred)
    if obj == "fbeta":
        return fbeta_score(y_true, y_pred, beta=beta)
    raise ValueError(obj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="inputs")
    ap.add_argument("--drop-cols-dir", default="feature_sets")
    ap.add_argument("--out", default="models_all/hparams.json")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--random-state", type=int, default=0)
    ap.add_argument("--redo", action="store_true", help="redo search even if entry exists")
    args = ap.parse_args()

    cache = load_json(args.out)

    # Shorts buckets
    mults = [1,2,5,10]
    tags = ["PTM","Random"]
    feats_list = [69,126]
    criteria = ["f1","max_recall","fbeta"]

    # Grid (small)
    lr_grid = []
    for solver in ["lbfgs", "liblinear"]:
        if solver == "lbfgs":
            for C in [0.1,1,10]:
                lr_grid.append(dict(solver=solver, penalty="l2", C=C))
        else:
            for penalty in ["l1","l2"]:
                for C in [0.1,1,10]:
                    lr_grid.append(dict(solver=solver, penalty=penalty, C=C))

    sgd_grid = []
    for alpha in [1e-5, 1e-4, 1e-3]:
        for penalty in ["l2", "elasticnet"]:
            if penalty == "l2":
                sgd_grid.append(dict(alpha=alpha, penalty=penalty, l1_ratio=0.0))
            else:
                for l1_ratio in [0.15, 0.5]:
                    sgd_grid.append(dict(alpha=alpha, penalty=penalty, l1_ratio=l1_ratio))

    def bucket_file(tag: str, mult: int) -> str:
        if tag == "PTM":
            return f"{args.data_dir}/ptms_short.txt" if mult==1 else f"{args.data_dir}/ptms_short{mult}.txt"
        else:
            return f"{args.data_dir}/random_short.txt" if mult==1 else f"{args.data_dir}/random_short{mult}.txt"

    for crit in criteria:
        for tag in tags:
            for mult in mults:
                for feats in feats_list:
                    key = ["linear", crit, tag, f"{mult}x", f"feats{feats}"]
                    if (not args.redo) and deep_get(cache, key) is not None:
                        continue

                    infile = bucket_file(tag, mult)
                    X, y = load_bucket_table(infile, feats, args.drop_cols_dir)
                    Xtr, Xte, ytr, yte = train_test_split(
                        X, y, test_size=0.2, stratify=y, random_state=args.random_state
                    )

                    best = {"logreg": None, "scaled_sgdc": None, "partial_fit_sgdc": None}
                    best_score = {"logreg": -1, "scaled_sgdc": -1, "partial_fit_sgdc": -1}

                    # Logistic search
                    for hp in lr_grid:
                        model = Pipeline([
                            ("scaler", StandardScaler()),
                            ("clf", LogisticRegression(
                                solver=hp["solver"], penalty=hp["penalty"], C=hp["C"],
                                max_iter=2000, random_state=args.random_state
                            ))
                        ])
                        model.fit(Xtr, ytr)
                        pred = model.predict(Xte)
                        s = score_objective(crit, yte, pred, args.beta)
                        if s > best_score["logreg"]:
                            best_score["logreg"] = s
                            best["logreg"] = hp

                    # SGD search (use same grid for scaled + partial_fit entry; training method differs,
                    # but final model is SGDClassifier anyway. We'll store separate entries for clarity.)
                    for hp in sgd_grid:
                        model = Pipeline([
                            ("scaler", StandardScaler()),
                            ("clf", SGDClassifier(
                                loss="log_loss",
                                alpha=hp["alpha"],
                                penalty=hp["penalty"],
                                l1_ratio=hp["l1_ratio"],
                                max_iter=2000,
                                random_state=args.random_state,
                                tol=1e-3
                            ))
                        ])
                        model.fit(Xtr, ytr)
                        pred = model.predict(Xte)
                        s = score_objective(crit, yte, pred, args.beta)
                        if s > best_score["scaled_sgdc"]:
                            best_score["scaled_sgdc"] = s
                            best["scaled_sgdc"] = hp
                        if s > best_score["partial_fit_sgdc"]:
                            best_score["partial_fit_sgdc"] = s
                            best["partial_fit_sgdc"] = hp

                    deep_set(cache, key, {"best": best, "scores": best_score})
                    save_json(args.out, cache)
                    print("Saved hparams:", "/".join(key))

    print("Done. Cache:", args.out)


if __name__ == "__main__":
    main()
