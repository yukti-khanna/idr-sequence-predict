#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


def load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())


def save_model(model, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("wb") as f:
        pickle.dump(model, f)


def load_drop_cols(feats: int, drop_dir: str) -> list[str]:
    return json.loads((Path(drop_dir) / f"drop_cols_{feats}.json").read_text())


def load_bucket_table(path: str, feats: int, drop_dir: str):
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan).fillna(0)
    if "CLASS" not in df.columns:
        raise ValueError(f"{path} missing CLASS")
    if "NAME" in df.columns:
        df = df.set_index("NAME")
    df = df.drop(["idr_name"], axis=1, errors="ignore")

    y = df["CLASS"].astype(int).values
    Xdf = df.drop(["CLASS"], axis=1, errors="ignore")
    Xdf = Xdf.drop(load_drop_cols(feats, drop_dir), axis=1, errors="ignore")

    if Xdf.shape[1] != feats:
        raise ValueError(f"{path}: after dropping -> {Xdf.shape[1]} cols, expected {feats}")
    return Xdf.to_numpy(float), y


def bucket_file(data_dir: str, tag: str, mult: int) -> str:
    if tag == "PTM":
        return f"{data_dir}/ptms_short.txt" if mult == 1 else f"{data_dir}/ptms_short{mult}.txt"
    return f"{data_dir}/random_short.txt" if mult == 1 else f"{data_dir}/random_short{mult}.txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="inputs")
    ap.add_argument("--drop-cols-dir", default="feature_sets")
    ap.add_argument("--hparams", default="models_all/hparams.json")
    ap.add_argument("--models-root", default="models_all/nn")
    ap.add_argument("--random-state", type=int, default=0)
    args = ap.parse_args()

    hp = load_json(args.hparams)
    tags = ["PTM", "Random"]
    mults = [1,2,5,10]
    feats_list = [69,126]

    for tag in tags:
        for mult in mults:
            infile = bucket_file(args.data_dir, tag, mult)
            for feats in feats_list:
                key = hp.get("nn", {}).get(tag, {}).get(f"{mult}x", {}).get(f"feats{feats}")
                if key is None:
                    raise KeyError(f"Missing hparams for nn/{tag}/{mult}x/feats{feats}")
                best = key["best"]
                hidden = tuple(best["hidden"])
                alpha = best["alpha"]
                lr = best["learning_rate_init"]

                X, y = load_bucket_table(infile, feats, args.drop_cols_dir)

                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", MLPClassifier(
                        hidden_layer_sizes=hidden,
                        alpha=alpha,
                        learning_rate_init=lr,
                        max_iter=800,
                        early_stopping=True,
                        n_iter_no_change=20,
                        random_state=args.random_state
                    ))
                ])
                model.fit(X, y)

                outdir = Path(args.models_root) / tag / f"{mult}x" / f"feats{feats}"
                save_model(model, outdir / f"NN_{tag}_{feats}_feats_model.sav")
                print("Saved NN:", outdir)

    print("DONE NN training from cached hparams.")


if __name__ == "__main__":
    main()
