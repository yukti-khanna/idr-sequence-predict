#!/usr/bin/env python3
import argparse
import subprocess

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-fasta", required=True)
    ap.add_argument("--out-fasta", required=True)
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--step", type=int, default=1)
    args = ap.parse_args()

    # Assumes your sliding_window.py supports positional args OR update this call accordingly.
    # If your sliding_window.py already has argparse, just call it directly.
    subprocess.run([
        "python", "scripts/sliding_window.py",
        args.in_fasta, args.out_fasta, str(args.window), str(args.step)
    ], check=True)

if __name__ == "__main__":
    main()
