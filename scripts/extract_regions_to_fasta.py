#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description="Extract merged-region FASTA from region TSVs.")
    ap.add_argument("--regions-dir", default="results/regions")
    ap.add_argument("--outdir", default="results/fasta")
    ap.add_argument("--thresholds", nargs="*", type=int, default=[100, 95, 90, 85])
    ap.add_argument("--min-frags", type=int, default=2)
    ap.add_argument("--fasta", action="append", required=True,
                    help="Parent FASTA(s). Provide twice: TF fasta and disordered fasta.")
    args = ap.parse_args()

    reg_dir = Path(args.regions_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for p in args.thresholds:
        regions_tsv = reg_dir / f"regions__pct{p}__min{args.min_frags}frags.tsv"
        if not regions_tsv.exists():
            raise FileNotFoundError(regions_tsv)

        out_fa = outdir / f"regions__pct{p}__min{args.min_frags}frags.fasta"
        cmd = ["python", "scripts/extract_regions.py", "--regions", str(regions_tsv)]
        for f in args.fasta:
            cmd += ["--fasta", f]
        cmd += ["--out", str(out_fa)]
        run(cmd)

    print("DONE.")


if __name__ == "__main__":
    main()
