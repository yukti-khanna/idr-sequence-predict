#!/usr/bin/env python3
import argparse, json, pickle, os
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.environ.get("FEATURES_FILE", "inputs/final_all_noNaN.txt"))
    ap.add_argument("--models-dir", required=True, help="Folder containing 3 SMOTEENN .sav files")
    ap.add_argument("--tag", choices=["PTM","Random"], required=True)
    ap.add_argument("--feats", type=int, choices=[69,126], required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--drop-cols-dir", default="feature_sets")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    mdir = Path(args.models_dir)

    sgdc = mdir / f"Scaled_SGDC_{args.tag}_{args.feats}_feats.txt_over_under_sampled_finalized_model.sav"
    lr   = mdir / f"Logistic_Regression_{args.tag}_{args.feats}_feats.txt_over_under_sampled_finalized_model.sav"
    pf   = mdir / f"Partial_Fit_SGDC_{args.tag}_{args.feats}_feats.txt_over_under_sampled_finalized_model.sav"
    for p in (sgdc, lr, pf):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    df = pd.read_csv(args.features).replace([np.inf,-np.inf], np.nan).fillna(0)
    if "NAME" not in df.columns:
        raise ValueError("Feature table must contain NAME column")
    df = df.set_index("NAME").drop(["idr_name","CLASS"], axis=1, errors="ignore")

    drop_cols = json.loads((Path(args.drop_cols_dir)/f"drop_cols_{args.feats}.json").read_text())
    df = df.drop(drop_cols, axis=1, errors="ignore")

    if df.shape[1] != args.feats:
        raise ValueError(f"After dropping: X has {df.shape[1]} cols, expected {args.feats}")

    X = df.to_numpy(dtype=float)
    ids = df.index.tolist()

    def pred(model_path):
        model = pickle.load(open(model_path,"rb"))
        y = model.predict(X)
        return [ids[i] for i,v in enumerate(y) if int(v)==0]

    (outdir / f"ScaledSGD__{args.tag}_{args.feats}_feats.txt").write_text("\n".join(pred(sgdc))+"\n")
    (outdir / f"LogReg__{args.tag}_{args.feats}_feats.txt").write_text("\n".join(pred(lr))+"\n")
    (outdir / f"PartialFit__{args.tag}_{args.feats}_feats.txt").write_text("\n".join(pred(pf))+"\n")

if __name__ == "__main__":
    main()
