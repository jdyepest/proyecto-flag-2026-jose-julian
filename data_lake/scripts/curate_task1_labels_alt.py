import argparse
import json
import os
import sys
import time
from typing import Dict, List

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

LABELS = ["INTRO", "BACK", "METH", "RESU", "DISC", "CONTR", "LIM", "CONC"]

INSTRUCTIONS = """You label Spanish academic text fragments with exactly ONE rhetorical label.

LABELS (choose exactly one):
INTRO  = motivation, problem, objectives, initial context
BACK   = background, related work, prior studies, theory, references
METH   = methodology, materials & methods, procedure
RESU   = results / findings / reported data (minimal interpretation)
DISC   = interpretation/discussion of results, implications, comparison
CONTR  = explicit contributions/novelty ("we propose/present...")
LIM    = limitations, threats, constraints
CONC   = conclusions, closing, future work

RULES:
- Choose the MAIN function of the fragment.
- If it describes OTHER authors/studies or is bibliography -> BACK.
- If it reports data without interpreting -> RESU.
- If it interprets/explains results -> DISC.
- Output must follow the provided JSON schema exactly.
"""

# Structured output schema for the Responses API (text.format json_schema).
# This makes the model adhere strictly to the schema. :contentReference[oaicite:2]{index=2}
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "gold_label": {"type": "string", "enum": LABELS},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "notes": {"type": "string"},
                },
                "required": ["chunk_id", "gold_label", "confidence", "notes"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["labels"],
    "additionalProperties": False,
}


def chunk_list(rows: List[Dict], batch_size: int) -> List[List[Dict]]:
    return [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]


def normalize_notes(s: str) -> str:
    s = (s or "").strip().replace("\n", " ")
    # keep it short-ish, but don’t be aggressive
    return s[:120]


def call_model(client: OpenAI, model: str, batch_rows: List[Dict], max_retries: int = 5) -> Dict:
    # Keep input compact to reduce cost: chunk_id + text only.
    payload = [{"chunk_id": r["chunk_id"], "text": r["text"]} for r in batch_rows]

    # We ask the model to label ALL items in one response.
    user_input = (
        "Label each item. Return JSON that matches the schema.\n\n"
        f"Items:\n{json.dumps(payload, ensure_ascii=False)}"
    )

    backoff = 1.0
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.responses.create(
                model=model,
                instructions=INSTRUCTIONS,
                input=user_input,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "task1_labels",
                        "schema": JSON_SCHEMA,
                        "strict": True,
                    }
                },
            )
            # The SDK provides aggregated output_text. :contentReference[oaicite:3]{index=3}
            raw = resp.output_text
            return json.loads(raw)
        except Exception as e:
            last_err = e
            # simple exponential backoff
            time.sleep(backoff)
            backoff = min(backoff * 2, 20)

    raise RuntimeError(f"Model call failed after {max_retries} retries: {last_err}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="CSV with columns: chunk_id,silver_label,text")
    parser.add_argument("--output_csv", required=True, help="Final labeled CSV path")
    parser.add_argument("--model", default="gpt-5-mini", help="Model name for assisted review")
    parser.add_argument("--batch_size", type=int, default=25, help="How many rows per API call")
    parser.add_argument("--max_chars", type=int, default=2500, help="Truncate each text to this many chars")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output_csv if present")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY env var not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    df = pd.read_csv(args.input_csv)
    required = {"chunk_id", "silver_label", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    # Truncate to keep token usage reasonable
    df["text"] = df["text"].astype(str).str.slice(0, args.max_chars)

    # Resume support: skip already-labeled chunk_ids
    done_ids = set()
    if args.resume and os.path.exists(args.output_csv):
        out_df = pd.read_csv(args.output_csv)
        if "chunk_id" in out_df.columns and "gold_label" in out_df.columns:
            done_ids = set(out_df["chunk_id"].astype(str).tolist())
            print(f"Resume enabled: found {len(done_ids)} already labeled rows in {args.output_csv}")

    work_df = df[~df["chunk_id"].astype(str).isin(done_ids)].copy()
    if work_df.empty:
        print("Nothing to do: all rows already labeled.")
        return

    rows = work_df.to_dict(orient="records")
    batches = chunk_list(rows, args.batch_size)

    # We’ll append results incrementally to avoid losing progress.
    # If output exists and resume is on, we append; otherwise we create new.
    write_header = not (args.resume and os.path.exists(args.output_csv))

    for i, batch in enumerate(batches, start=1):
        print(f"[{i}/{len(batches)}] Labeling batch of {len(batch)} ...")
        result = call_model(client, args.model, batch)

        # Validate returned ids and build rows
        returned = result.get("labels", [])
        by_id = {r["chunk_id"]: r for r in returned}

        out_rows = []
        for r in batch:
            cid = str(r["chunk_id"])
            if cid not in by_id:
                # If model missed an item, mark for manual review
                out_rows.append(
                    {
                        "chunk_id": cid,
                        "silver_label": r.get("silver_label", ""),
                        "text": r.get("text", ""),
                        "gold_label": "",
                        "confidence": 0.0,
                        "notes": "MISSING_FROM_MODEL_OUTPUT",
                    }
                )
                continue

            mr = by_id[cid]
            out_rows.append(
                {
                    "chunk_id": cid,
                    "silver_label": r.get("silver_label", ""),
                    "text": r.get("text", ""),
                    "gold_label": mr["gold_label"],
                    "confidence": float(mr["confidence"]),
                    "notes": normalize_notes(mr["notes"]),
                }
            )

        batch_out_df = pd.DataFrame(out_rows)
        batch_out_df.to_csv(args.output_csv, mode="a", header=write_header, index=False)
        write_header = False

    print("Done. Reviewed CSV saved to:", args.output_csv)


if __name__ == "__main__":
    main()
