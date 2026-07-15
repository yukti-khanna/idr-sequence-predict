#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SELECTED_BY_COUNT = {
    69: "selected_feats_cum90.txt",
    126: "selected_feats_cum95.txt",
}


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
) -> list[str]:
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
            f"scripts expect {expected_n}. The cumulative thresholds generated a different "
            f"feature count, so the downstream scripts must be migrated from fixed 69/126 "
            f"labels before continuing."
        )

    selected_set = set(selected)
    drop_cols = [c for c in all_columns if c not in selected_set]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(drop_cols, indent=2) + "\n")
    print(
        f"Wrote {out_path}: keep={len(selected)}, drop={len(drop_cols)}, "
        f"total={len(all_columns)}"
    )
    return drop_cols


def ensure_drop_cols(table: str | Path, feats: int, feature_dir: str | Path) -> list[str]:
    """Return a legacy drop list, generating it from the selected-feature TXT if absent."""
    feature_dir = Path(feature_dir)
    out_path = feature_dir / f"drop_cols_{feats}.json"

    if out_path.exists():
        return json.loads(out_path.read_text())

    try:
        selected_name = SELECTED_BY_COUNT[feats]
    except KeyError as exc:
        raise ValueError(f"No generated feature-set mapping is defined for {feats}") from exc

    selected_path = feature_dir / selected_name
    if not selected_path.exists():
        raise FileNotFoundError(
            f"Neither {out_path} nor its source feature list {selected_path} exists. "
            "Run the feature-importance and select_feature_lists Snakemake rules first."
        )

    all_columns = read_columns(Path(table))
    if not all_columns:
        raise ValueError(f"No candidate feature columns found in {table}")

    return write_drop_list(all_columns, selected_path, out_path, expected_n=feats)


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
