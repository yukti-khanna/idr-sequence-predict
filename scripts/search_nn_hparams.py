#!/usr/bin/env python3
from __future__ import annotations

import argparse, itertools, json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

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
    args = ap.parse_args()

    cache = load_json(args.out)

    mults = [1,2,5,10]
    tags = ["PTM","Random"]
    feats_list = [69,126]

    # small grid
    hidden_grid = [(64,), (128,), (128,64)]
    alpha_grid = [1e-5, 1e-4, 1e-3]
    lr_grid = [1e-4, 1e-3]

    def bucket_file(tag: str, mult: int) -> str:
        if tag == "PTM":
            return f"{args.data_dir}/ptms_short.txt" if mult==1 else f"{args.data_dir}/ptms_short{mult}.txt"
        else:
            return f"{args.data_dir}/random_short.txt" if mult==1 else f"{args.data_dir}/random_short{mult}.txt"

    for tag in tags:
        for mult in mults:
            for feats in feats_list:
                key = ["nn", tag, f"{mult}x", f"feats{feats}"]
                if (not args.redo) and deep_get(cache, key) is not None:
                    continue

                infile = bucket_file(tag, mult)
                X, y = load_bucket_table(infile, feats, args.drop_cols_dir)
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=args.random_state
                )

                best_hp = None
                best_score = -1.0

                for hidden, alpha, lr in itertools.product(hidden_grid, alpha_grid, lr_grid):
                    model = Pipeline([
                        ("scaler", StandardScaler()),
                        ("clf", MLPClassifier(
                            hidden_layer_sizes=hidden,
                            alpha=alpha,
                            learning_rate_init=lr,
                            max_iter=600,
                            early_stopping=True,
                            n_iter_no_change=20,
                            random_state=args.random_state
                        ))
                    ])
                    model.fit(Xtr, ytr)
                    pred = model.predict(Xte)
                    s = f1_score(yte, pred)
                    if s > best_score:
                        best_score = s
                        best_hp = {"hidden": list(hidden), "alpha": alpha, "learning_rate_init": lr}

                deep_set(cache, key, {"best": best_hp, "score_f1": best_score})
                save_json(args.out, cache)
                print("Saved NN hparams:", "/".join(key))

    print("Done. Cache:", args.out)

if __name__ == "__main__":
    main()
