#!/usr/bin/env python3
from __future__ import annotations

import argparse, itertools, json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import f1_score

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

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

    return Xdf.to_numpy(float), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="inputs")
    ap.add_argument("--drop-cols-dir", default="feature_sets")
    ap.add_argument("--out", default="models_all/hparams.json")
    ap.add_argument("--random-state", type=int, default=0)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--include-shorts", action="store_true", help="search shorts 1/2/5/10 buckets too")
    args = ap.parse_args()

    cache = load_json(args.out)

    feats_list = [69,126]
    tags = ["PTM","Random"]

    # resampler grid
    smote_k = [3,5]
    enn_k = [3,5]

    # classifier grids (small)
    lr_C = [0.1, 1, 10]
    sgd_alpha = [1e-5, 1e-4, 1e-3]
    sgd_penalty = ["l2", "elasticnet"]
    sgd_l1_ratio = [0.15, 0.5]

    def search_one_bucket(key_prefix: list[str], infile: str):
        for tag in tags:
            for feats in feats_list:
                key = key_prefix + [tag, f"feats{feats}"]
                if (not args.redo) and deep_get(cache, key) is not None:
                    continue

                X, y = load_bucket_table(infile if tag=="PTM" else infile.replace("PTM","random"), feats, args.drop_cols_dir)

                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=args.random_state
                )

                best = {"logreg": None, "sgdc": None, "partial_fit_sgdc": None, "resampler": None}
                best_score = {"logreg": -1, "sgdc": -1, "partial_fit_sgdc": -1}

                # resampler candidates
                for k1, k2 in itertools.product(smote_k, enn_k):
                    res = SMOTEENN(
                        random_state=args.random_state,
                        smote=SMOTE(k_neighbors=k1, random_state=args.random_state),
                        enn=EditedNearestNeighbours(n_neighbors=k2)
                    )
                    Xr, yr = res.fit_resample(MinMaxScaler().fit_transform(Xtr), ytr)

                    # logreg
                    for C in lr_C:
                        model = Pipeline([
                            ("scaler", MinMaxScaler()),
                            ("clf", LogisticRegression(C=C, solver="lbfgs", penalty="l2", max_iter=2000))
                        ])
                        model.fit(Xr, yr)
                        pred = model.predict(MinMaxScaler().fit_transform(Xte))
                        s = f1_score(yte, pred)
                        if s > best_score["logreg"]:
                            best_score["logreg"] = s
                            best["logreg"] = {"C": C, "solver":"lbfgs", "penalty":"l2"}
                            best["resampler"] = {"smote_k": k1, "enn_k": k2}

                    # sgd / partial-fit
                    for a in sgd_alpha:
                        for pen in sgd_penalty:
                            if pen == "l2":
                                hp = {"alpha": a, "penalty": pen, "l1_ratio": 0.0}
                                model = Pipeline([
                                    ("scaler", MinMaxScaler()),
                                    ("clf", SGDClassifier(loss="log_loss", alpha=a, penalty=pen, max_iter=2000, tol=1e-3))
                                ])
                                model.fit(Xr, yr)
                                pred = model.predict(MinMaxScaler().fit_transform(Xte))
                                s = f1_score(yte, pred)
                                if s > best_score["sgdc"]:
                                    best_score["sgdc"] = s
                                    best["sgdc"] = hp
                                    best["resampler"] = {"smote_k": k1, "enn_k": k2}
                                if s > best_score["partial_fit_sgdc"]:
                                    best_score["partial_fit_sgdc"] = s
                                    best["partial_fit_sgdc"] = hp
                                    best["resampler"] = {"smote_k": k1, "enn_k": k2}
                            else:
                                for l1r in sgd_l1_ratio:
                                    hp = {"alpha": a, "penalty": pen, "l1_ratio": l1r}
                                    model = Pipeline([
                                        ("scaler", MinMaxScaler()),
                                        ("clf", SGDClassifier(loss="log_loss", alpha=a, penalty=pen, l1_ratio=l1r, max_iter=2000, tol=1e-3))
                                    ])
                                    model.fit(Xr, yr)
                                    pred = model.predict(MinMaxScaler().fit_transform(Xte))
                                    s = f1_score(yte, pred)
                                    if s > best_score["sgdc"]:
                                        best_score["sgdc"] = s
                                        best["sgdc"] = hp
                                        best["resampler"] = {"smote_k": k1, "enn_k": k2}
                                    if s > best_score["partial_fit_sgdc"]:
                                        best_score["partial_fit_sgdc"] = s
                                        best["partial_fit_sgdc"] = hp
                                        best["resampler"] = {"smote_k": k1, "enn_k": k2}

                deep_set(cache, key, {"best": best, "scores": best_score})
                save_json(args.out, cache)
                print("Saved SMOTEENN hparams:", "/".join(key))

    # BIG buckets (your two files in inputs/)
    # We store under smoteenn_big/<TAG>/feats<feats>
    # PTM file: inputs/final_PTM_no_tfs.txt ; Random file: inputs/final_random.txt
    for feats in feats_list:
        for tag in tags:
            key = ["smoteenn_big", tag, f"feats{feats}"]
            if (not args.redo) and deep_get(cache, key) is not None:
                continue

            infile = f"{args.data_dir}/final_PTM_no_tfs.txt" if tag=="PTM" else f"{args.data_dir}/final_random.txt"
            X, y = load_bucket_table(infile, feats, args.drop_cols_dir)
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=args.random_state
            )

            # same search but simpler key (single bucket)
            best = {"logreg": None, "sgdc": None, "partial_fit_sgdc": None, "resampler": None}
            best_score = {"logreg": -1, "sgdc": -1, "partial_fit_sgdc": -1}

            for k1, k2 in itertools.product(smote_k, enn_k):
                res = SMOTEENN(
                    random_state=args.random_state,
                    smote=SMOTE(k_neighbors=k1, random_state=args.random_state),
                    enn=EditedNearestNeighbours(n_neighbors=k2)
                )
                Xr, yr = res.fit_resample(MinMaxScaler().fit_transform(Xtr), ytr)

                for C in lr_C:
                    model = Pipeline([("scaler", MinMaxScaler()),
                                      ("clf", LogisticRegression(C=C, solver="lbfgs", penalty="l2", max_iter=2000))])
                    model.fit(Xr, yr)
                    pred = model.predict(MinMaxScaler().fit_transform(Xte))
                    s = f1_score(yte, pred)
                    if s > best_score["logreg"]:
                        best_score["logreg"] = s
                        best["logreg"] = {"C": C, "solver":"lbfgs", "penalty":"l2"}
                        best["resampler"] = {"smote_k": k1, "enn_k": k2}

                for a in sgd_alpha:
                    for pen in sgd_penalty:
                        if pen == "l2":
                            hp = {"alpha": a, "penalty": pen, "l1_ratio": 0.0}
                            model = Pipeline([("scaler", MinMaxScaler()),
                                              ("clf", SGDClassifier(loss="log_loss", alpha=a, penalty=pen, max_iter=2000, tol=1e-3))])
                            model.fit(Xr, yr)
                            pred = model.predict(MinMaxScaler().fit_transform(Xte))
                            s = f1_score(yte, pred)
                            if s > best_score["sgdc"]:
                                best_score["sgdc"] = s
                                best["sgdc"] = hp
                                best["resampler"] = {"smote_k": k1, "enn_k": k2}
                            if s > best_score["partial_fit_sgdc"]:
                                best_score["partial_fit_sgdc"] = s
                                best["partial_fit_sgdc"] = hp
                                best["resampler"] = {"smote_k": k1, "enn_k": k2}
                        else:
                            for l1r in sgd_l1_ratio:
                                hp = {"alpha": a, "penalty": pen, "l1_ratio": l1r}
                                model = Pipeline([("scaler", MinMaxScaler()),
                                                  ("clf", SGDClassifier(loss="log_loss", alpha=a, penalty=pen, l1_ratio=l1r, max_iter=2000, tol=1e-3))])
                                model.fit(Xr, yr)
                                pred = model.predict(MinMaxScaler().fit_transform(Xte))
                                s = f1_score(yte, pred)
                                if s > best_score["sgdc"]:
                                    best_score["sgdc"] = s
                                    best["sgdc"] = hp
                                    best["resampler"] = {"smote_k": k1, "enn_k": k2}
                                if s > best_score["partial_fit_sgdc"]:
                                    best_score["partial_fit_sgdc"] = s
                                    best["partial_fit_sgdc"] = hp
                                    best["resampler"] = {"smote_k": k1, "enn_k": k2}

            deep_set(cache, key, {"best": best, "scores": best_score})
            save_json(args.out, cache)
            print("Saved SMOTEENN BIG hparams:", "/".join(key))

    print("Done. Cache:", args.out)

if __name__ == "__main__":
    main()
