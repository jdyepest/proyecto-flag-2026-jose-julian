import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv
from google import genai


load_dotenv()  # Load environment variables from .env file

RHETORICAL_LABELS = ["INTRO", "BACK", "METH", "RESU", "DISC", "CONTR", "LIM", "CONC"]

INSTRUCTIONS = """You label Spanish scientific fragments for Task2 (contribution detection).

You must output:
1) gold_is_contribution: whether the fragment EXPLICITLY expresses a scientific contribution/novelty.
2) gold_rhetorical_label: confirm the rhetorical role using exactly ONE label from the list.

IMPORTANT:
- Be conservative: only mark positive when the contribution is explicit (e.g., "proponemos", "presentamos", "nuestra contribución", "ponemos a disposición").
- Do not hallucinate: rely only on the fragment text.
- Return ONLY JSON (no Markdown). Output MUST be an object with this shape:
  {"labels": [ {fragment_id, gold_is_contribution, gold_rhetorical_label, confidence, notes}, ... ]}
"""

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fragment_id": {"type": "string"},
                    "gold_is_contribution": {"type": "boolean"},
                    "gold_rhetorical_label": {"type": "string", "enum": RHETORICAL_LABELS},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "notes": {"type": "string"},
                },
                "required": [
                    "fragment_id",
                    "gold_is_contribution",
                    "gold_rhetorical_label",
                    "confidence",
                    "notes",
                ],
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
    return s[:160]

def safe_float(v, default: float = 0.0) -> float:
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return default
    # accept "0,87"
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return default


def parse_json_loose(text: str) -> Dict:
    """
    El modelo a veces devuelve JSON rodeado por ```json ...```.
    Parseamos de forma tolerante.
    """
    t = (text or "").strip()
    if not t:
        raise ValueError("Empty model output")
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", t, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return json.loads(fenced.group(1))

    start_list = t.find("[")
    end_list = t.rfind("]")
    if start_list != -1 and end_list != -1 and end_list > start_list:
        return json.loads(t[start_list : end_list + 1])

    start_obj = t.find("{")
    end_obj = t.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        return json.loads(t[start_obj : end_obj + 1])

    raise ValueError("Could not parse JSON from model output")


def validate_result(result, expected_ids: List[str]) -> Dict[str, Dict]:
    # Model might return either {"labels":[...]} or directly a list [...]
    if isinstance(result, list):
        labels = result
    elif isinstance(result, dict):
        labels = result.get("labels", [])
    else:
        raise ValueError(f"Invalid output: expected dict or list, got {type(result).__name__}")

    if not isinstance(labels, list):
        raise ValueError("Invalid output: missing 'labels' array")
    by_id = {}
    for item in labels:
        if not isinstance(item, dict):
            continue
        fid = str(item.get("fragment_id") or "")
        if not fid:
            continue
        by_id[fid] = item

    # Ensure all expected ids exist (we'll handle missing later)
    return by_id


def call_model(client: genai.Client, model: str, batch_rows: List[Dict], max_retries: int = 3) -> Dict:
    payload = [
        {
            "fragment_id": str(r["fragment_id"]),
            "silver_is_contribution": bool(r.get("silver_is_contribution", False)),
            "silver_rhetorical_label": str(r.get("silver_rhetorical_label", "")),
            "text": str(r["text"]),
        }
        for r in batch_rows
    ]

    prompt = (
        f"{INSTRUCTIONS}\n\n"
        "Items (JSON):\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )

    backoff = 1.0
    last_err = None
    for _ in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    # NOTE: Evitamos response_schema por incompatibilidades del validador del SDK (pydantic).
                    # Validamos localmente después de parsear JSON.
                    "response_mime_type": "application/json",
                },
            )
            return parse_json_loose(resp.text)
        except Exception as e:
            last_err = e
            time.sleep(backoff)
            backoff = min(backoff * 2, 20)

    raise RuntimeError(f"Model call failed after {max_retries} retries: {last_err}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True, help="CSV con columnas: fragment_id,text (+ silver_* opcional)")
    ap.add_argument("--output_csv", required=True, help="Output CSV con etiquetas gold revisadas")
    ap.add_argument("--model", default="gemini-2.5-flash", help="Modelo de apoyo para la revision por lotes")
    ap.add_argument("--batch_size", type=int, default=25)
    ap.add_argument("--max_chars", type=int, default=3500)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("ERROR: set GEMINI_API_KEY (or GOOGLE_API_KEY).", file=sys.stderr)
        sys.exit(1)

    client = genai.Client()

    df = pd.read_csv(args.input_csv)
    required = {"fragment_id", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["fragment_id"] = df["fragment_id"].astype(str)
    df["text"] = df["text"].astype(str).str.slice(0, args.max_chars)
    if "silver_is_contribution" not in df.columns:
        df["silver_is_contribution"] = False
    if "silver_rhetorical_label" not in df.columns:
        df["silver_rhetorical_label"] = ""

    done_ids = set()
    if args.resume and os.path.exists(args.output_csv):
        out_df = pd.read_csv(args.output_csv)
        if "fragment_id" in out_df.columns and "gold_is_contribution" in out_df.columns:
            done_ids = set(out_df["fragment_id"].astype(str).tolist())
            print(f"Resume: {len(done_ids)} rows already labeled in {args.output_csv}")

    work_df = df[~df["fragment_id"].isin(done_ids)].copy()
    if work_df.empty:
        print("Nothing to do: all rows already labeled.")
        return

    rows = work_df.to_dict(orient="records")
    batches = chunk_list(rows, args.batch_size)
    write_header = not (args.resume and os.path.exists(args.output_csv))

    for i, batch in enumerate(batches, start=1):
        print(f"[{i}/{len(batches)}] Labeling {len(batch)} rows…")
        result = call_model(client, args.model, batch)

        expected_ids = [str(r["fragment_id"]) for r in batch]
        by_id = validate_result(result, expected_ids)

        # Keep output columns stable even if an older output file already has extra columns
        existing_cols = None
        if args.resume and os.path.exists(args.output_csv):
            try:
                existing_cols = pd.read_csv(args.output_csv, nrows=0).columns.tolist()
            except Exception:
                existing_cols = None

        base_cols = [
            "fragment_id",
            "silver_is_contribution",
            "silver_rhetorical_label",
            "text",
            "gold_is_contribution",
            "gold_rhetorical_label",
            "confidence",
            "notes",
        ]
        out_cols = existing_cols if existing_cols else base_cols

        out_rows = []
        for r in batch:
            fid = str(r["fragment_id"])
            if fid not in by_id:
                row = {
                    "fragment_id": fid,
                    "silver_is_contribution": bool(r.get("silver_is_contribution", False)),
                    "silver_rhetorical_label": str(r.get("silver_rhetorical_label", "")),
                    "text": r.get("text", ""),
                    "gold_is_contribution": "",
                    "gold_rhetorical_label": "",
                    "confidence": 0.0,
                    "notes": "MISSING_FROM_MODEL_OUTPUT",
                }
                # compatibility with older outputs
                if "gold_contribution_type" in out_cols:
                    row["gold_contribution_type"] = ""
                out_rows.append(row)
                continue

            mr = by_id[fid]
            row = {
                "fragment_id": fid,
                "silver_is_contribution": bool(r.get("silver_is_contribution", False)),
                "silver_rhetorical_label": str(r.get("silver_rhetorical_label", "")),
                "text": r.get("text", ""),
                "gold_is_contribution": bool(mr.get("gold_is_contribution", False)),
                "gold_rhetorical_label": str(mr.get("gold_rhetorical_label", "")),
                "confidence": safe_float(mr.get("confidence", 0.0), default=0.0),
                "notes": normalize_notes(mr.get("notes", "")),
            }
            # compatibility with older outputs
            if "gold_contribution_type" in out_cols:
                row["gold_contribution_type"] = ""
            out_rows.append(row)

        pd.DataFrame(out_rows, columns=out_cols).to_csv(args.output_csv, mode="a", header=write_header, index=False)
        write_header = False

    print("Done. Saved:", args.output_csv)


if __name__ == "__main__":
    main()
