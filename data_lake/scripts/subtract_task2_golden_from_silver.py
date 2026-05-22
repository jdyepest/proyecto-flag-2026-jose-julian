from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def pick_id_column(df: pd.DataFrame, candidates: list[str], source_name: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"{source_name} is missing id columns. "
        f"Expected one of: {candidates}. Found: {list(df.columns)}"
    )


def load_gold_ids(gold_parquet: str, gold_csv: str) -> tuple[set[str], set[str]]:
    fragment_ids: set[str] = set()
    chunk_ids: set[str] = set()

    if gold_parquet:
        gpar = pd.read_parquet(gold_parquet).copy()
        col = pick_id_column(gpar, ["fragment_id", "source_chunk_id", "chunk_id"], "Gold parquet")
        vals = gpar[col].dropna().astype(str).str.strip()
        if col == "fragment_id":
            fragment_ids.update(vals.tolist())
        else:
            chunk_ids.update(vals.tolist())

    if gold_csv:
        gcsv = pd.read_csv(gold_csv).copy()
        col = pick_id_column(gcsv, ["fragment_id", "source_chunk_id", "chunk_id"], "Gold csv")
        vals = gcsv[col].dropna().astype(str).str.strip()
        if col == "fragment_id":
            fragment_ids.update(vals.tolist())
        else:
            chunk_ids.update(vals.tolist())

    return fragment_ids, chunk_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver_parquet", required=True, help="Task2 silver parquet base")
    ap.add_argument("--out_parquet", required=True, help="Output Task2 silver train parquet without golden rows")
    ap.add_argument("--gold_parquet", default="", help="Optional labeled gold parquet")
    ap.add_argument("--gold_csv", default="", help="Optional labeled gold csv")
    args = ap.parse_args()

    if not args.gold_parquet and not args.gold_csv:
        raise ValueError("Provide at least one of --gold_parquet or --gold_csv")

    silver = pd.read_parquet(args.silver_parquet).copy()
    if "fragment_id" not in silver.columns:
        raise ValueError("Task2 silver parquet must contain 'fragment_id'")

    silver["fragment_id"] = silver["fragment_id"].astype(str).str.strip()
    if "source_chunk_id" in silver.columns:
        silver["source_chunk_id"] = silver["source_chunk_id"].astype(str).str.strip()
    if "chunk_id" in silver.columns:
        silver["chunk_id"] = silver["chunk_id"].astype(str).str.strip()

    gold_fragment_ids, gold_chunk_ids = load_gold_ids(args.gold_parquet, args.gold_csv)

    mask_keep = pd.Series(True, index=silver.index)

    removed_by_fragment = 0
    removed_by_chunk = 0

    if gold_fragment_ids:
        drop_fragment = silver["fragment_id"].isin(gold_fragment_ids)
        removed_by_fragment = int(drop_fragment.sum())
        mask_keep &= ~drop_fragment

    if gold_chunk_ids:
        if "source_chunk_id" in silver.columns:
            drop_chunk = silver["source_chunk_id"].isin(gold_chunk_ids)
        elif "chunk_id" in silver.columns:
            drop_chunk = silver["chunk_id"].isin(gold_chunk_ids)
        else:
            drop_chunk = pd.Series(False, index=silver.index)
        removed_by_chunk = int((drop_chunk & mask_keep).sum())
        mask_keep &= ~drop_chunk

    out = silver[mask_keep].copy()

    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    before = len(silver)
    after = len(out)
    removed_total = before - after

    print("Saved:", out_path)
    print("Rows before:", before)
    print("Rows removed total:", removed_total)
    print(" - removed by fragment_id overlap:", removed_by_fragment)
    print(" - removed by chunk overlap:", removed_by_chunk)
    print("Rows after:", after)

    if "is_contribution" in out.columns:
        print("Final is_contribution counts:")
        print(out["is_contribution"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
