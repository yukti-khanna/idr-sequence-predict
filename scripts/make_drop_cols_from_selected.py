#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_columns(path: Path) -> list[str]:
    """Read a feature table and return candidate model feature columns."""
    df = pd.read_csv(path, sep=None, engine="python", nrows=1)
    return [c for c in df.columns if c not in {"NAME", "CLASS", "idr_name"}]


def read_selected(path: Path) -> list[str]:
    selected = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not selected:
        raise ValueError(f"No features found in {path}")
    duplicates = sorted({x for x in selected if selected.count(x) > 1})
    if duplicates:
        raise ValueError(f"Duplicate features in {path}: {duplicates}")
    return selected


def write_drop_list(
    all_columns: list[str], selected_path: Path, out_path: Path, expected_n: int
) -> None:
    selected = read_selected(selected_path)
    missing = [f for f in selected if f not in all_columns]
    if missing:
        raise ValueError(
            f"{selected_path} contains {len(missing)} features absent from the table; "
            f"examples: {missing[:10]}"
        )
    if len(selected) != expected_n:
        raise ValueError(
            f"{selected_path} contains {len(selected)} features, but the legacy downstream "
            f"scripts expect {expected_n}. Regenerate the selected-feature list or update "
            f"the downstream scripts to use cumulative-set names instead of fixed counts."
        )

    selected_set = set(selected)
    drop_cols = [c for c in all_columns if c not in selected_set]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(drop_cols, indent=2) + "\n")
    print(
        f"Wrote {out_path}: keep={len(selected)}, drop={len(drop_cols)}, "
        f"total={len(all_columns)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Convert generated selected-feature text files into the legacy "
            "drop_cols_69.json and drop_cols_126.json files."
        )
    )
    ap.add_argument(
        "--table",
        required=True,
        help="A labeled feature table containing the complete feature-column set",
    )
    ap.add_argument("--selected-69", required=True)
    ap.add_argument("--selected-126", required=True)
    ap.add_argument("--out-69", required=True)
    ap.add_argument("--out-126", required=True)
    args = ap.parse_args()

    all_columns = read_columns(Path(args.table))
    if not all_columns:
        raise ValueError(f"No candidate feature columns found in {args.table}")

    write_drop_list(
        all_columns,
        Path(args.selected_69),
        Path(args.out_69),
        expected_n=69,
    )
    write_drop_list(
        all_columns,
        Path(args.selected_126),
        Path(args.out_126),
        expected_n=126,
    )


if __name__ == "__main__":
    main()
