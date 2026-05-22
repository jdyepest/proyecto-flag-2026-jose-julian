import argparse
import json
import os
import sys
import time
from typing import Dict, List

import pandas as pd
from google import genai
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
Return JSON that matches the provided schema exactly.
"""

# JSON Schema for structured output
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
                    "confidence": {"type": "number"},
                    "notes": {"type": "string"},
                },
                "required": ["chunk_id", "gold_label", "confidence", "notes"],
            },
        }
    },
    "required": ["labels"],
}


def chunk_list(rows: List[Dict], batch_size: int) -> List[List[Dict]]:
    return [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]


def normalize_notes(s: str) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s[:120]


def call_model(
    client: genai.Client,
    model: str,
    batch_rows: List[Dict],
    max_retries: int = 1,
) -> Dict:
    payload = [{"chunk_id": str(r["chunk_id"]), "text": r["text"]} for r in batch_rows]
    prompt = (
        f"{INSTRUCTIONS}\n\n"
        "Items (JSON):\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )

    backoff = 1.0
    last_err = None

    for _ in range(max_retries):
        try:
            # Structured output via JSON schema
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": JSON_SCHEMA,
                },
            )
            # SDK returns text; should be valid JSON due to schema.
            return json.loads(resp.text)
        except Exception as e:
            last_err = e
            time.sleep(backoff)
            backoff = min(backoff * 2, 20)

    raise RuntimeError(f"Model call failed after {max_retries} retries: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True, help="CSV with: chunk_id,silver_label,text")
    ap.add_argument("--output_csv", required=True, help="Output CSV with gold labels")
    ap.add_argument("--model", default="gemini-2.5-flash", help="Modelo de apoyo para la revision por lotes")
    ap.add_argument("--batch_size", type=int, default=25)
    ap.add_argument("--max_chars", type=int, default=2500)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    # The SDK picks up GEMINI_API_KEY automatically if set.
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("ERROR: set GEMINI_API_KEY (or GOOGLE_API_KEY).", file=sys.stderr)
        sys.exit(1)

    client = genai.Client()

    df = pd.read_csv(args.input_csv)
    required = {"chunk_id", "silver_label", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["chunk_id"] = df["chunk_id"].astype(str)
    df["text"] = df["text"].astype(str).str.slice(0, args.max_chars)

    done_ids = set()
    if args.resume and os.path.exists(args.output_csv):
        out_df = pd.read_csv(args.output_csv)
        if "chunk_id" in out_df.columns and "gold_label" in out_df.columns:
            done_ids = set(out_df["chunk_id"].astype(str).tolist())
            print(f"Resume: {len(done_ids)} rows already labeled in {args.output_csv}")

    work_df = df[~df["chunk_id"].isin(done_ids)].copy()
    if work_df.empty:
        print("Nothing to do: all rows already labeled.")
        return

    rows = work_df.to_dict(orient="records")
    batches = chunk_list(rows, args.batch_size)
    print(len(batches))
    

    write_header = not (args.resume and os.path.exists(args.output_csv))

    for i, batch in enumerate(batches, start=1):
        print(f"[{i}/{len(batches)}] Labeling {len(batch)} rows…")
        result = call_model(client, args.model, batch)

        returned = result.get("labels", [])
        by_id = {str(r["chunk_id"]): r for r in returned}

        out_rows = []
        for r in batch:
            cid = str(r["chunk_id"])
            if cid not in by_id:
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

        pd.DataFrame(out_rows).to_csv(args.output_csv, mode="a", header=write_header, index=False)
        write_header = False

    print("Done. Saved:", args.output_csv)


if __name__ == "__main__":
    main()
