from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver_parquet", required=True, help="Silver parquet base")
    ap.add_argument(
        "--gold_parquet",
        default="",
        help="Gold parquet with chunk_id. Optional if --gold_csv is provided.",
    )
    ap.add_argument(
        "--gold_csv",
        default="",
        help="Annotated/labeled CSV with chunk_id. Optional if --gold_parquet is provided.",
    )
    ap.add_argument("--out_parquet", required=True, help="Output silver train parquet without golden rows")
    args = ap.parse_args()

    if not args.gold_parquet and not args.gold_csv:
        raise ValueError("Provide at least one of --gold_parquet or --gold_csv")

    silver = pd.read_parquet(args.silver_parquet).copy()
    if "chunk_id" not in silver.columns:
        raise ValueError("Silver parquet must contain column 'chunk_id'")
    silver["chunk_id"] = silver["chunk_id"].astype(str)

    gold_ids: set[str] = set()

    if args.gold_parquet:
        gpar = pd.read_parquet(args.gold_parquet)
        if "chunk_id" not in gpar.columns:
            raise ValueError("Gold parquet must contain column 'chunk_id'")
        gold_ids.update(gpar["chunk_id"].astype(str).tolist())

    if args.gold_csv:
        gcsv = pd.read_csv(args.gold_csv)
        if "chunk_id" not in gcsv.columns:
            raise ValueError("Gold csv must contain column 'chunk_id'")
        gold_ids.update(gcsv["chunk_id"].astype(str).tolist())

    before = len(silver)
    out = silver[~silver["chunk_id"].isin(gold_ids)].copy()
    after = len(out)
    removed = before - after

    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    print("Saved:", out_path)
    print("Rows before:", before)
    print("Rows removed (gold overlap):", removed)
    print("Rows after:", after)


if __name__ == "__main__":
    main()

