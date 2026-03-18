#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from itertools import groupby

def parse_fasta(path: str):
    fh = open(path, "r")
    faiter = (x[1] for x in groupby(fh, lambda line: line.startswith(">")))
    for header in faiter:
        header_str = next(header)[1:].strip().split()[0]
        seq = "".join(s.strip() for s in next(faiter))
        yield header_str, seq
    fh.close()

def write_fasta(records, out_path: str):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for h, s in records:
            f.write(f">{h}\n{s}\n")

def main():
    ap = argparse.ArgumentParser(description="Sliding window FASTA generator.")
    ap.add_argument("--in-fasta", required=True)
    ap.add_argument("--out-fasta", required=True)
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--min-len", type=int, default=15)
    ap.add_argument("--id-mode", choices=["start", "index"], default="start",
                    help="start: seq_id:<1-based start>; index: seq_id:<k>")
    args = ap.parse_args()

    out_records = []
    for header, seq in parse_fasta(args.in_fasta):
        if len(seq) < args.min_len:
            continue
        k = 0
        # i is 0-based start
        for i in range(0, len(seq) - args.window + 1, args.step):
            sub = seq[i:i+args.window]
            if len(sub) != args.window:
                continue
            k += 1
            if args.id_mode == "start":
                sub_header = f"{header}:{i+1}"  # 1-based
            else:
                sub_header = f"{header}:{k}"
            out_records.append((sub_header, sub))

    write_fasta(out_records, args.out_fasta)
    print(f"Wrote {len(out_records)} windows -> {args.out_fasta}")

if __name__ == "__main__":
    main()
