#!/usr/bin/env python3
import argparse
import subprocess

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # adjust to match your normal.py interface
    subprocess.run(["python", "scripts/normal.py", args.inp, args.out], check=True)

if __name__ == "__main__":
    main()
