from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def pct(n: int, d: int) -> float:
    return round((100.0 * n / d), 4) if d else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled_csv", required=True, help="CSV with silver_label and gold_label")
    ap.add_argument("--out_json", required=True, help="Output JSON stats")
    args = ap.parse_args()

    df = pd.read_csv(args.labeled_csv).copy()
    required = {"silver_label", "gold_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in labeled csv: {sorted(missing)}")

    df["silver_label"] = df["silver_label"].astype(str).str.upper().str.strip()
    df["gold_label"] = df["gold_label"].astype(str).str.upper().str.strip()
    df = df[df["gold_label"] != ""].copy()

    total = len(df)
    changed = int((df["silver_label"] != df["gold_label"]).sum())
    unchanged = total - changed

    per_label = []
    for label, g in df.groupby("silver_label"):
        n = len(g)
        c = int((g["silver_label"] != g["gold_label"]).sum())
        per_label.append(
            {
                "silver_label": label,
                "rows": int(n),
                "changed_rows": int(c),
                "changed_pct": pct(c, n),
                "unchanged_rows": int(n - c),
                "unchanged_pct": pct(n - c, n),
            }
        )
    per_label = sorted(per_label, key=lambda x: x["rows"], reverse=True)

    transitions = (
        df.groupby(["silver_label", "gold_label"]).size().reset_index(name="count").sort_values("count", ascending=False)
    )

    payload: dict[str, Any] = {
        "global": {
            "rows": int(total),
            "changed_rows": int(changed),
            "changed_pct": pct(changed, total),
            "unchanged_rows": int(unchanged),
            "unchanged_pct": pct(unchanged, total),
        },
        "by_silver_label": per_label,
        "top_transitions": transitions.head(50).to_dict(orient="records"),
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved stats:", out)
    print("Global:", payload["global"])


if __name__ == "__main__":
    main()

