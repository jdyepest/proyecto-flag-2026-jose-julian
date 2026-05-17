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
    changed_mask = df["heuristic_label"] != df["llm_label"]
    changed = int(changed_mask.sum())

    by_label: dict[str, Any] = {}
    for label in LABELS:
        sub = df[df["heuristic_label"] == label]
        n = len(sub)
        c = int((sub["heuristic_label"] != sub["llm_label"]).sum()) if n else 0
        by_label[label] = {
            "total_heuristic": int(n),
            "changed_to_llm": int(c),
            "changed_pct": pct(c, n),
        }

    confusion = (
        df.groupby(["heuristic_label", "llm_label"])
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
        "by_heuristic_label": by_label,
        "top_transitions": top_transitions,
    }
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heuristic_parquet", required=True)
    ap.add_argument("--llm_csv", required=True)
    ap.add_argument("--out_compare_parquet", required=True)
    ap.add_argument("--out_final_llm_parquet", required=True)
    ap.add_argument("--out_stats_json", required=True)
    args = ap.parse_args()

    heuristic = pd.read_parquet(args.heuristic_parquet).copy()
    llm = pd.read_csv(args.llm_csv).copy()

    required_h = {"chunk_id", "label", "text"}
    missing_h = required_h - set(heuristic.columns)
    if missing_h:
        raise ValueError(f"Heuristic parquet missing columns: {sorted(missing_h)}")

    required_l = {"chunk_id", "llm_label", "llm_confidence"}
    missing_l = required_l - set(llm.columns)
    if missing_l:
        raise ValueError(f"LLM csv missing columns: {sorted(missing_l)}")

    heuristic["chunk_id"] = heuristic["chunk_id"].astype(str)
    heuristic["label"] = heuristic["label"].astype(str).str.upper().str.strip()
    llm["chunk_id"] = llm["chunk_id"].astype(str)
    llm["llm_label"] = llm["llm_label"].astype(str).str.upper().str.strip()
    llm["llm_confidence"] = pd.to_numeric(llm["llm_confidence"], errors="coerce").fillna(0.0)
    llm["llm_confidence"] = llm["llm_confidence"].clip(0.0, 1.0)

    # Keep last seen prediction per chunk_id in case of resume duplicates.
    llm = llm.drop_duplicates(subset=["chunk_id"], keep="last")

    compare = heuristic.merge(
        llm[["chunk_id", "llm_label", "llm_confidence"]],
        how="left",
        on="chunk_id",
    )

    compare = compare.rename(columns={"label": "heuristic_label"})
    compare["llm_label"] = compare["llm_label"].fillna("")
    compare["llm_confidence"] = compare["llm_confidence"].fillna(0.0)
    compare["changed"] = compare["heuristic_label"] != compare["llm_label"]

    # Final label policy: use LLM when available/valid, fallback to heuristic.
    compare["final_label"] = compare["llm_label"].where(compare["llm_label"].isin(LABELS), compare["heuristic_label"])

    out_compare = Path(args.out_compare_parquet)
    out_compare.parent.mkdir(parents=True, exist_ok=True)
    compare.to_parquet(out_compare, index=False)

    final_llm = compare.copy()
    final_llm = final_llm.rename(columns={"final_label": "label"})
    final_llm = final_llm.drop(columns=["heuristic_label", "llm_label", "changed"])
    out_final = Path(args.out_final_llm_parquet)
    out_final.parent.mkdir(parents=True, exist_ok=True)
    final_llm.to_parquet(out_final, index=False)

    stats = compute_stats(compare)
    out_stats = Path(args.out_stats_json)
    out_stats.parent.mkdir(parents=True, exist_ok=True)
    out_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved compare:", out_compare)
    print("Saved final llm:", out_final)
    print("Saved stats:", out_stats)
    print("Global:", stats["global"])


if __name__ == "__main__":
    main()

