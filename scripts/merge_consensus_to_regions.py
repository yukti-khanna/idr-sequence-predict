#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description="Merge consensus fragments into regions TSVs.")
    ap.add_argument("--consensus-dir", default="results/consensus")
    ap.add_argument("--outdir", default="results/regions")
    ap.add_argument("--thresholds", nargs="*", type=int, default=[100, 95, 90, 85])
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--min-frags", type=int, default=2)
    args = ap.parse_args()

    cons_dir = Path(args.consensus_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for p in args.thresholds:
        frags = cons_dir / f"frags__pct{p}.txt"
        if not frags.exists():
            raise FileNotFoundError(frags)

        out_tsv = outdir / f"regions__pct{p}__min{args.min_frags}frags.tsv"
        run([
            "python", "scripts/get_boundaries.py",
            "--frags", str(frags),
            "--window", str(args.window),
            "--min-frags", str(args.min_frags),
            "--out", str(out_tsv),
        ])

    print("DONE.")


if __name__ == "__main__":
    main()
