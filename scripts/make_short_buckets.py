#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positives", required=True, help="Positives feature TSV")
    ap.add_argument("--control-pool", required=True, help="Control pool TSV (PTM or Random)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prefix", required=True, help="e.g. ptms_short or random_short")
    ap.add_argument("--mults", nargs="*", type=int, default=[1,2,5,10])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pos = pd.read_csv(args.positives, sep="\t")
    ctrl = pd.read_csv(args.control_pool, sep="\t")
    npos = len(pos)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for m in args.mults:
        n = npos * m
        if n > len(ctrl):
            raise ValueError(f"Need {n} controls for {m}x but only {len(ctrl)} in pool")

        samp = ctrl.sample(n=n, random_state=args.seed + m)
        suffix = "" if m == 1 else str(m)
        out = outdir / f"{args.prefix}{suffix}.txt"
        samp.to_csv(out, sep="\t", index=False)
        print("Wrote", out)

if __name__ == "__main__":
    main()
