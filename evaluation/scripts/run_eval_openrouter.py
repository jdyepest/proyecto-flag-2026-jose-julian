#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import eval_gold as eg


def _safe_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(":", "_")


def _default_output(repo_root: Path, model_name: str) -> Path:
    name = f"openrouter_{_safe_name(model_name)}.json"
    return repo_root / "artifacts" / "eval_results" / name


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa Golden Set con OpenRouter (Task1+Task2).")
    parser.add_argument("--model", default=os.environ.get("OPENROUTER_MODEL") or "meta-llama/llama-3.3-70b-instruct:free")
    parser.add_argument("--batch-task1", type=int, default=1)
    parser.add_argument("--batch-task2", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="")
    parser.add_argument("--mlflow-experiment", default="")
    args = parser.parse_args()

    os.environ["LOCAL_LLM_PROVIDER"] = "openrouter"
    os.environ["OPENROUTER_MODEL"] = args.model
    os.environ["OPENROUTER_LOG_PROMPT"] = os.environ.get("OPENROUTER_LOG_PROMPT") or "1"

    if args.mlflow_experiment:
        os.environ["MLFLOW_EXPERIMENT"] = args.mlflow_experiment

    repo_root = Path(__file__).resolve().parents[2]
    out_path = Path(args.output) if args.output else _default_output(repo_root, args.model)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[OpenRouter] Iniciando evaluación")
    print(f"[OpenRouter] model={args.model}")
    print(f"[OpenRouter] batch_task1={args.batch_task1} batch_task2={args.batch_task2} limit={args.limit}")
    print(f"[OpenRouter] MLFLOW_TRACKING_URI={os.environ.get('MLFLOW_TRACKING_URI')}")

    res_t1 = eg.evaluate_task1(
        "llm",
        encoder_variant="roberta",
        batch_size=args.batch_task1,
        limit=args.limit,
    )
    res_t2 = eg.evaluate_task2(
        "llm",
        encoder_variant="roberta",
        batch_size=args.batch_task2,
        limit=args.limit,
    )

    payload = {
        "meta": {
            "model_label": f"openrouter-{args.model}",
            "run_at": datetime.now(timezone.utc).isoformat(),
            "params": {
                "model": args.model,
                "batch_task1": args.batch_task1,
                "batch_task2": args.batch_task2,
                "limit": args.limit,
                "provider": "openrouter",
            },
        },
        "results": [res_t1.__dict__, res_t2.__dict__],
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[OK] Guardado: {out_path}")

    try:
        import pandas as pd

        df = pd.DataFrame([res_t1.__dict__, res_t2.__dict__])
        print(df[["task", "accuracy", "precision", "recall", "f1", "total_time_s", "time_per_doc_s"]])
    except Exception:
        pass


if __name__ == "__main__":
    main()
