#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Normalize feature table (legacy logic: fill NaN with 0).")
    ap.add_argument("--infile", required=True, help="Input feature table (TSV from idr.mol.feats)")
    ap.add_argument("--outfile", required=True, help="Output TSV")
    args = ap.parse_args()

    df = pd.read_csv(args.infile, sep="\t", index_col=0)
    df.fillna(0, inplace=True)

    out = Path(args.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t")
    print(f"Wrote normalized -> {out}")

if __name__ == "__main__":
    main()

