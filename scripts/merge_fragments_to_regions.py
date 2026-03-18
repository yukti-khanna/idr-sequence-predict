#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path


def parse_frag_id(s: str) -> tuple[str, int]:
    """
    Expected format: <seq_id>:<start>
    Example: Q5M775_28:166
    """
    s = s.strip()
    if not s:
        raise ValueError("Empty fragment id")
    if ":" not in s:
        raise ValueError(f"Fragment id lacks ':': {s}")
    left, right = s.rsplit(":", 1)
    return left, int(right)


def merge_starts(starts: list[int], window: int) -> list[tuple[int, int, int]]:
    """
    Given sorted window start positions (1-based), merge overlapping windows of fixed length.
    Returns list of (start, end, n_frags).
    Overlap condition: next_start <= current_end (where end = start + window - 1)
    """
    starts = sorted(set(starts))
    if not starts:
        return []

    regions = []
    cur_start = starts[0]
    cur_end = cur_start + window - 1
    n = 1

    for st in starts[1:]:
        en = st + window - 1
        if st <= cur_end:   # overlaps (or touches if exactly equals)
            cur_end = max(cur_end, en)
            n += 1
        else:
            regions.append((cur_start, cur_end, n))
            cur_start, cur_end, n = st, en, 1

    regions.append((cur_start, cur_end, n))
    return regions


def main():
    ap = argparse.ArgumentParser(description="Merge fragment windows into regions and keep min supporting fragments.")
    ap.add_argument("--frags", required=True, help="Consensus fragment list (one ID per line, e.g. Qxxx:123)")
    ap.add_argument("--window", type=int, default=15, help="Window length used to create fragment IDs")
    ap.add_argument("--min-frags", type=int, default=2, help="Keep only regions supported by >= this many fragments")
    ap.add_argument("--out", required=True, help="Output TSV")
    args = ap.parse_args()

    frag_file = Path(args.frags)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    starts_by_seq = defaultdict(list)
    for line in frag_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        seq_id, st = parse_frag_id(line)
        starts_by_seq[seq_id].append(st)

    rows = []
    for seq_id, starts in starts_by_seq.items():
        for rstart, rend, n in merge_starts(starts, args.window):
            if n >= args.min_frags:
                rows.append((seq_id, rstart, rend, n))

    rows.sort(key=lambda x: (x[0], x[1], x[2]))

    with out.open("w") as f:
        f.write("seq_id\tstart\tend\tn_frags\n")
        for seq_id, s, e, n in rows:
            f.write(f"{seq_id}\t{s}\t{e}\t{n}\n")

    print(f"Wrote {len(rows)} regions to {out}")


if __name__ == "__main__":
    main()
