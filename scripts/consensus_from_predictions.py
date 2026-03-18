#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description="Create consensus fragment lists directly from prediction lists (dedup per-file)."
    )
    ap.add_argument("--pred-root", default="results/predictions", help="Root containing prediction .txt files")
    ap.add_argument("--outdir", default="results/consensus", help="Output folder")
    ap.add_argument("--thresholds", nargs="*", type=int, default=[100, 95, 90, 85])
    ap.add_argument("--include", default="", help="Only include prediction files whose path contains this substring")
    ap.add_argument("--glob", default="*.txt", help="Prediction filename glob")
    ap.add_argument("--top", type=int, default=20, help="Print top-N fragments by support")
    args = ap.parse_args()

    pred_root = Path(args.pred_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(pred_root.rglob(args.glob))
    if args.include:
        files = [f for f in files if args.include in str(f)]

    if not files:
        raise SystemExit(f"No prediction files found under {pred_root} (include='{args.include}')")

    cnt = Counter()
    used = 0
    for f in files:
        # Deduplicate within each prediction list
        ids = {line.strip() for line in f.read_text().splitlines() if line.strip()}
        if not ids:
            continue
        cnt.update(ids)
        used += 1

    if used == 0 or not cnt:
        raise SystemExit("All prediction files were empty after parsing.")

    N = used
    max_support = max(cnt.values())
    print(f"Using {N} prediction lists (include='{args.include}')")
    print(f"Unique fragments observed: {len(cnt)}")
    print(f"Max support of any fragment: {max_support}/{N} ({100.0*max_support/N:.2f}%)")

    print(f"\nTop {args.top} fragments by support:")
    for frag, c in cnt.most_common(args.top):
        print(f"{c:>4}/{N}  {frag}")

    print("\nConsensus outputs:")
    for p in args.thresholds:
        k = math.ceil((p / 100.0) * N)
        kept = sorted([frag for frag, c in cnt.items() if c >= k])
        out = outdir / f"frags__pct{p}.txt"
        out.write_text("\n".join(kept) + ("\n" if kept else ""))
        print(f"{out.name}: {len(kept)} fragments (k={k}/{N})")


if __name__ == "__main__":
    main()
