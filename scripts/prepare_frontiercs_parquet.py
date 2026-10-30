#!/usr/bin/env python3
"""
Prepare Frontier-CS algorithmic problems as Parquet for VERL RL training.

Only includes problems with numeric IDs (编号). Creates train.parquet and val.parquet.

Usage:
  python scripts/prepare_frontiercs_parquet.py
  python scripts/prepare_frontiercs_parquet.py --val-ratio 0.1 --output-dir data/frontiercs
  python scripts/prepare_frontiercs_parquet.py --full-for-both   # all 172 for both train & val
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
PROBLEMS_DIR = PROJECT_ROOT / "Frontier-CS" / "algorithmic" / "problems"
DEFAULT_OUT = PROJECT_ROOT / "data" / "frontiercs"


def build_prompt(statement: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": f"""You are a competitive programmer. Solve the following problem in C++. Output ONLY the C++ code wrapped in ```cpp and ```. No explanation.

{statement}

Generate solution code:""",
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction for validation")
    parser.add_argument("--full-for-both", action="store_true", help="Use all problems for both train and val (creates full.parquet)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ids",
        type=Path,
        default=None,
        help="JSON manifest restricting the problem set, e.g. data/sample_lists/frontiercs_val_172.json "
        "(uses its 'paper_172.ids'; a plain list of ids also works)",
    )
    args = parser.parse_args()

    if args.full_for_both:
        args.val_ratio = 0.0

    keep = None
    if args.ids is not None:
        import json

        manifest = json.loads(args.ids.read_text())
        ids = manifest["paper_172"]["ids"] if isinstance(manifest, dict) else manifest
        keep = {str(i) for i in ids}

    rows = []
    for pid_dir in sorted(PROBLEMS_DIR.iterdir()):
        if not pid_dir.is_dir():
            continue
        if not pid_dir.name.isdigit():
            continue
        if keep is not None and pid_dir.name not in keep:
            continue
        stmt = pid_dir / "statement.txt"
        if not stmt.exists():
            continue
        statement = stmt.read_text(encoding="utf-8")
        rows.append(
            {
                "prompt": build_prompt(statement),
                "reward_model": {"ground_truth": pid_dir.name},
                "data_source": "frontiercs",
            }
        )

    if not rows:
        print("No numeric-ID problems found.")
        return

    df = pd.DataFrame(rows)
    if args.val_ratio > 0:
        df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        n_val = max(1, int(len(df) * args.val_ratio))
        df_val = df.iloc[:n_val]
        df_train = df.iloc[n_val:]
    else:
        df_train = df
        df_val = df.iloc[:0]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.parquet"
    val_path = args.output_dir / "val.parquet"

    df_train.to_parquet(train_path, index=False)
    print(f"Saved {len(df_train)} train -> {train_path}")
    if len(df_val) > 0:
        df_val.to_parquet(val_path, index=False)
        print(f"Saved {len(df_val)} val -> {val_path}")

    if args.full_for_both:
        full_path = args.output_dir / "full.parquet"
        df.to_parquet(full_path, index=False)
        print(f"Saved {len(df)} full (use for both train & val) -> {full_path}")


if __name__ == "__main__":
    main()
