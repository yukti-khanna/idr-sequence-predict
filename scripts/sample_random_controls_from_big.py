#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def extract_uniprot(name: str) -> str:
    return str(name).split(":", 1)[0]

def load_ids(path: str) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    return {x.strip() for x in p.read_text().splitlines() if x.strip()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big-feats", required=True, help="TSV from RES")
    ap.add_argument("--out", required=True, help="Output TSV")
    ap.add_argument("--n", type=int, required=True, help="How many rows to sample")
    ap.add_argument("--exclude-ids", nargs="*", default=[], help="ID files (UniProt IDs) to exclude")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    exclude = set()
    for f in args.exclude_ids:
        exclude |= load_ids(f)

    df = pd.read_csv(args.big_feats, sep="\t")
    if "NAME" not in df.columns:
        raise ValueError("Expected NAME column in big-feats")
    df["__uniprot__"] = df["NAME"].map(extract_uniprot)
    if exclude:
        df = df[~df["__uniprot__"].isin(exclude)]
    df = df.drop(columns=["__uniprot__"])

    if args.n > len(df):
        raise ValueError(f"Requested {args.n} but only {len(df)} available after exclusion")

    samp = df.sample(n=args.n, random_state=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    samp.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {len(samp)} rows -> {args.out}")

if __name__ == "__main__":
    main()
