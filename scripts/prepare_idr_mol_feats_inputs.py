#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import shutil

def main():
    ap = argparse.ArgumentParser(description="Prepare DATA folder and input_file_feats.txt for idr.mol.feats.")
    ap.add_argument("--fastas", nargs="+", required=True, help="FASTA files to process")
    ap.add_argument("--data-dir", default="DATA")
    ap.add_argument("--list-file", default="input_file_feats.txt")
    ap.add_argument("--mode", choices=["copy", "symlink"], default="copy")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    names = []
    for f in args.fastas:
        src = Path(f)
        dst = data_dir / src.name
        if dst.exists():
            dst.unlink()
        if args.mode == "copy":
            shutil.copy2(src, dst)
        else:
            dst.symlink_to(src.resolve())
        names.append(dst.name)

    Path(args.list_file).write_text("\n".join(names) + "\n")
    print(f"Wrote {args.list_file} with {len(names)} entries")
    print(f"DATA dir: {data_dir}")

if __name__ == "__main__":
    main()
