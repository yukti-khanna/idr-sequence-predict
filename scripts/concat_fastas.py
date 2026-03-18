#!/usr/bin/env python3
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fastas", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as w:
        for f in args.fastas:
            w.write(Path(f).read_text())
            if not str(w).endswith("\n"):
                w.write("\n")

if __name__ == "__main__":
    main()
