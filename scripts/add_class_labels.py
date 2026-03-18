#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positives", required=True, help="CSV")
    ap.add_argument("--controls", required=True, help="CSV")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pos = pd.read_csv(args.positives)
    neg = pd.read_csv(args.controls)

    pos["CLASS"] = 0
    neg["CLASS"] = 1

    out = pd.concat([pos, neg], ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()
