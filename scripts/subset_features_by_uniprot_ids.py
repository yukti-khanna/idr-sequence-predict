#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def extract_uniprot(name: str) -> str:
    # NAME looks like Q5M775_28:166 or Q9UKS7:243
    left = str(name).split(":", 1)[0]
    return left

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big-feats", required=True, help="TSV from RES (NAME as first column)")
    ap.add_argument("--ids", required=True, help="UniProt IDs, one per line")
    ap.add_argument("--out", required=True, help="Output TSV")
    args = ap.parse_args()

    ids = {x.strip() for x in Path(args.ids).read_text().splitlines() if x.strip()}
    df = pd.read_csv(args.big_feats, sep="\t")
    if "NAME" not in df.columns:
        raise ValueError("Expected NAME column in big-feats")

    df["__uniprot__"] = df["NAME"].map(extract_uniprot)
    sub = df[df["__uniprot__"].isin(ids)].drop(columns=["__uniprot__"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {len(sub)} rows -> {args.out}")

if __name__ == "__main__":
    main()
