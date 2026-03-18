#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_fasta(path: Path) -> dict[str, str]:
    seqs = {}
    cur = None
    buf = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur is not None:
                    seqs[cur] = "".join(buf)
                cur = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if cur is not None:
            seqs[cur] = "".join(buf)
    return seqs


def main():
    ap = argparse.ArgumentParser(description="Extract merged regions (TSV) into FASTA from parent FASTAs.")
    ap.add_argument("--regions", required=True, help="TSV with columns: seq_id, start, end, n_frags")
    ap.add_argument("--fasta", action="append", required=True,
                    help="Parent FASTA(s). Provide multiple; later ones fill missing IDs.")
    ap.add_argument("--out", required=True, help="Output FASTA")
    ap.add_argument("--one-based", action="store_true", default=True,
                    help="Treat start/end as 1-based inclusive (default: True)")
    args = ap.parse_args()

    # load all sequences (2 FASTAs in your case)
    seq_db = {}
    for fp in args.fasta:
        seq_db.update(parse_fasta(Path(fp)))

    regions = Path(args.regions)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with regions.open() as f_in, out.open("w") as f_out:
        header = f_in.readline().strip().split("\t")
        idx = {name: i for i, name in enumerate(header)}
        for line in f_in:
            parts = line.rstrip("\n").split("\t")
            seq_id = parts[idx["seq_id"]]
            start = int(parts[idx["start"]])
            end = int(parts[idx["end"]])
            n_frags = int(parts[idx["n_frags"]])

            if seq_id not in seq_db:
                # skip missing ids
                continue

            seq = seq_db[seq_id]
            # 1-based inclusive -> python slice
            s0 = start - 1
            e0 = end
            if s0 < 0 or e0 > len(seq) or s0 >= e0:
                continue

            subseq = seq[s0:e0]
            f_out.write(f">{seq_id}:{start}-{end}|n_frags={n_frags}\n")
            # wrap 60
            for i in range(0, len(subseq), 60):
                f_out.write(subseq[i:i+60] + "\n")
            n_written += 1

    print(f"Wrote {n_written} FASTA entries to {out}")


if __name__ == "__main__":
    main()
