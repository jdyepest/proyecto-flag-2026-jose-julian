from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import pandas as pd
from openai import OpenAI
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True), override=False)

LABELS = ["INTRO", "BACK", "METH", "RESU", "DISC", "CONTR", "LIM", "CONC"]

SYSTEM_PROMPT = """You are labeling Spanish academic fragments with exactly one rhetorical label.

Allowed labels:
INTRO, BACK, METH, RESU, DISC, CONTR, LIM, CONC

Rules:
- Return one label per item.
- Choose the MAIN rhetorical function.
- BACK: related work, references, prior studies, bibliography context.
- RESU: findings/results with minimal interpretation.
- DISC: interpretation/implications/comparison of results.
- CONTR: explicit novelty/contribution by the current work.
- LIM: limitations/threats/constraints of the current work.

Output format:
Return ONLY valid JSON object:
{
  "labels": [
    {
      "chunk_id": "string",
      "gold_label": "INTRO|BACK|METH|RESU|DISC|CONTR|LIM|CONC",
      "confidence": 0.0,
      "notes": "short reason"
    }
  ]
}
No markdown, no extra keys.
"""


def chunk_rows(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]


def normalize_notes(notes: str) -> str:
    return (notes or "").strip().replace("\n", " ")[:200]


def parse_json_safely(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Empty model output")
    return json.loads(raw)


def call_review_batch(
    client: OpenAI,
    model: str,
    items: list[dict[str, Any]],
    max_retries: int = 5,
) -> dict[str, Any]:
    payload = [{"chunk_id": str(r["chunk_id"]), "text": str(r["text"])} for r in items]
    user_prompt = "Label each item.\nItems JSON:\n" + json.dumps(payload, ensure_ascii=False)

    last_err: Exception | None = None
    backoff = 1.0
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = (resp.choices[0].message.content or "").strip()
            return parse_json_safely(content)
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)

    raise RuntimeError(f"Batch review failed after retries: {last_err}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_parquet", required=True, help="Parquet with chunk_id,label,text")
    ap.add_argument("--output_csv", required=True, help="Output CSV with reviewed labels")
    ap.add_argument("--model", default="meta-llama/llama-3.3-70b-instruct:nitro")
    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--max_chars", type=int, default=2500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    api_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    df = pd.read_parquet(args.input_parquet).copy()
    required = {"chunk_id", "label", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if args.limit is not None:
        df = df.head(args.limit).copy()

    df["chunk_id"] = df["chunk_id"].astype(str)
    df["label"] = df["label"].astype(str)
    df["text"] = df["text"].astype(str).str.slice(0, args.max_chars)

    done_ids: set[str] = set()
    if args.resume and os.path.exists(args.output_csv):
        prev = pd.read_csv(args.output_csv)
        if "chunk_id" in prev.columns:
            done_ids = set(prev["chunk_id"].astype(str).tolist())
            print(f"Resume: {len(done_ids)} rows already present")

    work_df = df[~df["chunk_id"].isin(done_ids)].copy()
    if work_df.empty:
        print("Nothing to do.")
        return

    rows = work_df.to_dict(orient="records")
    batches = chunk_rows(rows, args.batch_size)
    print(f"Total rows to label: {len(rows)}")
    print(f"Batches: {len(batches)}")

    write_header = not (args.resume and os.path.exists(args.output_csv))

    for i, batch in enumerate(batches, start=1):
        print(f"[{i}/{len(batches)}] labeling {len(batch)}")
        result = call_review_batch(client=client, model=args.model, items=batch)
        returned = result.get("labels") or []
        by_id = {str(x.get("chunk_id", "")): x for x in returned if isinstance(x, dict)}

        out_rows: list[dict[str, Any]] = []
        for r in batch:
            cid = str(r["chunk_id"])
            base_row = {
                "chunk_id": cid,
                "initial_label": str(r["label"]),
                "text": str(r["text"]),
            }

            pred = by_id.get(cid)
            if not pred:
                out_rows.append(
                    {
                        **base_row,
                        "reviewed_label": "",
                        "reviewed_confidence": 0.0,
                        "review_notes": "MISSING_FROM_MODEL_OUTPUT",
                    }
                )
                continue

            reviewed_label = str(pred.get("gold_label", "")).strip().upper()
            if reviewed_label not in LABELS:
                reviewed_label = ""

            try:
                conf = float(pred.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0
            conf = max(0.0, min(1.0, conf))

            out_rows.append(
                {
                    **base_row,
                    "reviewed_label": reviewed_label,
                    "reviewed_confidence": conf,
                    "review_notes": normalize_notes(str(pred.get("notes", ""))),
                }
            )

        pd.DataFrame(out_rows).to_csv(args.output_csv, mode="a", header=write_header, index=False)
        write_header = False

    print("Done. Output:", args.output_csv)


if __name__ == "__main__":
    main()
