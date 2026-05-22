from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

LABELS = ["INTRO", "BACK", "METH", "RESU", "DISC", "CONTR", "LIM", "CONC"]


def pct(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return round(100.0 * num / den, 4)


def compute_stats(df: pd.DataFrame) -> dict[str, Any]:
    total = len(df)
    changed_mask = df["initial_label"] != df["reviewed_label"]
    changed = int(changed_mask.sum())

    by_label: dict[str, Any] = {}
    for label in LABELS:
        sub = df[df["initial_label"] == label]
        n = len(sub)
        c = int((sub["initial_label"] != sub["reviewed_label"]).sum()) if n else 0
        by_label[label] = {
            "total_initial": int(n),
            "changed_after_review": int(c),
            "changed_pct": pct(c, n),
        }

    confusion = (
        df.groupby(["initial_label", "reviewed_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    top_transitions = confusion.head(40).to_dict(orient="records")

    stats = {
        "global": {
            "total_rows": int(total),
            "changed_rows": int(changed),
            "changed_pct": pct(changed, total),
            "unchanged_rows": int(total - changed),
            "unchanged_pct": pct(total - changed, total),
        },
        "by_initial_label": by_label,
        "top_transitions": top_transitions,
    }
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--initial_parquet", required=True)
    ap.add_argument("--reviewed_csv", required=True)
    ap.add_argument("--out_compare_parquet", required=True)
    ap.add_argument("--out_final_curated_parquet", required=True)
    ap.add_argument("--out_stats_json", required=True)
    args = ap.parse_args()

    initial = pd.read_parquet(args.initial_parquet).copy()
    reviewed = pd.read_csv(args.reviewed_csv).copy()

    required_h = {"chunk_id", "label", "text"}
    missing_h = required_h - set(initial.columns)
    if missing_h:
        raise ValueError(f"Initial parquet missing columns: {sorted(missing_h)}")

    required_l = {"chunk_id", "reviewed_label", "reviewed_confidence"}
    missing_l = required_l - set(reviewed.columns)
    if missing_l:
        raise ValueError(f"Reviewed csv missing columns: {sorted(missing_l)}")

    initial["chunk_id"] = initial["chunk_id"].astype(str)
    initial["label"] = initial["label"].astype(str).str.upper().str.strip()
    reviewed["chunk_id"] = reviewed["chunk_id"].astype(str)
    reviewed["reviewed_label"] = reviewed["reviewed_label"].astype(str).str.upper().str.strip()
    reviewed["reviewed_confidence"] = pd.to_numeric(reviewed["reviewed_confidence"], errors="coerce").fillna(0.0)
    reviewed["reviewed_confidence"] = reviewed["reviewed_confidence"].clip(0.0, 1.0)

    # Keep last seen review per chunk_id in case of resume duplicates.
    reviewed = reviewed.drop_duplicates(subset=["chunk_id"], keep="last")

    compare = initial.merge(
        reviewed[["chunk_id", "reviewed_label", "reviewed_confidence"]],
        how="left",
        on="chunk_id",
    )

    compare = compare.rename(columns={"label": "initial_label"})
    compare["reviewed_label"] = compare["reviewed_label"].fillna("")
    compare["reviewed_confidence"] = compare["reviewed_confidence"].fillna(0.0)
    compare["changed"] = compare["initial_label"] != compare["reviewed_label"]

    # Final label policy: use reviewed label when available/valid, fallback to initial label.
    compare["final_label"] = compare["reviewed_label"].where(compare["reviewed_label"].isin(LABELS), compare["initial_label"])

    out_compare = Path(args.out_compare_parquet)
    out_compare.parent.mkdir(parents=True, exist_ok=True)
    compare.to_parquet(out_compare, index=False)

    final_curated = compare.copy()
    final_curated = final_curated.rename(columns={"final_label": "label"})
    final_curated = final_curated.drop(columns=["initial_label", "reviewed_label", "changed"])
    out_final = Path(args.out_final_curated_parquet)
    out_final.parent.mkdir(parents=True, exist_ok=True)
    final_curated.to_parquet(out_final, index=False)

    stats = compute_stats(compare)
    out_stats = Path(args.out_stats_json)
    out_stats.parent.mkdir(parents=True, exist_ok=True)
    out_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved compare:", out_compare)
    print("Saved final curated:", out_final)
    print("Saved stats:", out_stats)
    print("Global:", stats["global"])


if __name__ == "__main__":
    main()
