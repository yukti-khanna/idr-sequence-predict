#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours


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


def load_table(path: str, feats: int, drop_dir: str):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="inputs")
    ap.add_argument("--drop-cols-dir", default="feature_sets")
    ap.add_argument("--hparams", default="models_all/hparams.json")
    ap.add_argument("--models-root", default="models_all/smoteenn_big")
    ap.add_argument("--random-state", type=int, default=0)
    args = ap.parse_args()

    hp = load_json(args.hparams)
    feats_list = [69,126]
    tags = ["PTM","Random"]

    file_map = {
        "PTM": f"{args.data_dir}/final_PTM_no_tfs.txt",
        "Random": f"{args.data_dir}/final_random.txt"
    }

    for tag in tags:
        infile = file_map[tag]
        for feats in feats_list:
            key = hp.get("smoteenn_big", {}).get(tag, {}).get(f"feats{feats}")
            if key is None:
                raise KeyError(f"Missing hparams for smoteenn_big/{tag}/feats{feats}")
            best = key["best"]
            res_hp = best["resampler"]
            smote_k = res_hp["smote_k"]
            enn_k = res_hp["enn_k"]

            X, y = load_table(infile, feats, args.drop_cols_dir)

            # Resample in scaled space
            scaler0 = MinMaxScaler()
            X_scaled = scaler0.fit_transform(X)

            resampler = SMOTEENN(
                random_state=args.random_state,
                smote=SMOTE(k_neighbors=smote_k, random_state=args.random_state),
                enn=EditedNearestNeighbours(n_neighbors=enn_k)
            )
            Xr, yr = resampler.fit_resample(X_scaled, y)

            # Build & fit three models (each includes its own MinMaxScaler step)
            # Logistic
            lr_hp = best["logreg"]
            logreg = Pipeline([
                ("scaler", MinMaxScaler()),
                ("clf", LogisticRegression(C=lr_hp["C"], solver="lbfgs", penalty="l2", max_iter=2000))
            ])

            # SGD
            sgd_hp = best["sgdc"]
            sgdc = Pipeline([
                ("scaler", MinMaxScaler()),
                ("clf", SGDClassifier(
                    loss="log_loss",
                    alpha=sgd_hp["alpha"],
                    penalty=sgd_hp["penalty"],
                    l1_ratio=sgd_hp.get("l1_ratio", 0.0),
                    max_iter=2000,
                    tol=1e-3,
                    random_state=args.random_state
                ))
            ])

            # Partial-fit SGDC (saved separately, same final estimator type)
            pf_hp = best["partial_fit_sgdc"]
            pf = Pipeline([
                ("scaler", MinMaxScaler()),
                ("clf", SGDClassifier(
                    loss="log_loss",
                    alpha=pf_hp["alpha"],
                    penalty=pf_hp["penalty"],
                    l1_ratio=pf_hp.get("l1_ratio", 0.0),
                    max_iter=2000,
                    tol=1e-3,
                    random_state=args.random_state
                ))
            ])

            logreg.fit(Xr, yr)
            sgdc.fit(Xr, yr)
            pf.fit(Xr, yr)

            outdir = Path(args.models_root) / tag / f"feats{feats}"
            save_model(sgdc, outdir / f"Scaled_SGDC_{tag}_{feats}_feats.txt_over_under_sampled_finalized_model.sav")
            save_model(logreg, outdir / f"Logistic_Regression_{tag}_{feats}_feats.txt_over_under_sampled_finalized_model.sav")
            save_model(pf, outdir / f"Partial_Fit_SGDC_{tag}_{feats}_feats.txt_over_under_sampled_finalized_model.sav")
            print("Saved SMOTEENN BIG:", outdir)

    print("DONE SMOTEENN BIG training from cached hparams.")


if __name__ == "__main__":
    main()
