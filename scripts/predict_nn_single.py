#!/usr/bin/env python3
import argparse, json, pickle, os
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.environ.get("FEATURES_FILE", "inputs/final_all_noNaN.txt"))
    ap.add_argument("--model", required=True, help="Path to NN_*.sav")
    ap.add_argument("--out", required=True, help="Output txt file")
    ap.add_argument("--feats", type=int, choices=[69,126], required=True)
    ap.add_argument("--drop-cols-dir", default="feature_sets")
    args = ap.parse_args()

    df = pd.read_csv(args.features).replace([np.inf, -np.inf], np.nan).fillna(0)
    if "NAME" not in df.columns:
        raise ValueError("Feature table must contain NAME column")
    df = df.set_index("NAME").drop(["idr_name","CLASS"], axis=1, errors="ignore")

    drop_cols = json.loads((Path(args.drop_cols_dir)/f"drop_cols_{args.feats}.json").read_text())
    df = df.drop(drop_cols, axis=1, errors="ignore")

    if df.shape[1] != args.feats:
        raise ValueError(f"After dropping: X has {df.shape[1]} cols, expected {args.feats}")

    model = pickle.load(open(args.model, "rb"))
    X = df.to_numpy(dtype=float)
    ids = df.index.tolist()
    y = model.predict(X)
    pred = [ids[i] for i,v in enumerate(y) if int(v)==0]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(pred) + "\n")

if __name__ == "__main__":
    main()
