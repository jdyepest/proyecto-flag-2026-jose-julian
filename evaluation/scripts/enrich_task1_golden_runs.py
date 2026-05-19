#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import mlflow
import pandas as pd
import torch
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABEL_ORDER = ["INTRO", "BACK", "METH", "RES", "DISC", "CONTR", "LIM", "CONC"]


def normalize_label(label: str) -> str:
    lbl = str(label or "").strip().upper()
    if lbl == "RESU":
        return "RES"
    return lbl


def clean_text(text: str) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def load_golden(csv_path: Path, limit: int | None = None) -> tuple[list[str], list[str]]:
    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)
    if "text" not in df.columns or "gold_label" not in df.columns:
        raise ValueError("Golden CSV must include columns: text, gold_label")

    texts: list[str] = []
    y_true: list[str] = []
    for t, g in zip(df["text"], df["gold_label"], strict=False):
        txt = clean_text(t)
        if not txt:
            continue
        gl = normalize_label(g)
        if gl not in LABEL_ORDER:
            continue
        texts.append(txt)
        y_true.append(gl)
    return texts, y_true


def model_is_complete(model_dir: Path) -> bool:
    if not (model_dir / "config.json").exists():
        return False
    if not (model_dir / "tokenizer.json").exists():
        return False
    has_weights = any(
        (model_dir / name).exists()
        for name in [
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        ]
    )
    return has_weights


def predict_labels(model_dir: Path, texts: list[str], batch_size: int) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    id2label = getattr(model.config, "id2label", None) or {}

    y_pred: list[str] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs)
            preds = torch.argmax(out.logits, dim=-1).tolist()
            for p in preds:
                raw = id2label.get(str(p)) or id2label.get(p) or str(p)
                label = normalize_label(raw)
                if label not in LABEL_ORDER:
                    label = "BACK"
                y_pred.append(label)
    return y_pred


def log_metrics_to_run(client: MlflowClient, run_id: str, accuracy: float, recall_macro: float, recall_weighted: float) -> None:
    client.log_metric(run_id, "eval_gold_accuracy", float(accuracy))
    client.log_metric(run_id, "eval_gold_recall_macro", float(recall_macro))
    client.log_metric(run_id, "eval_gold_recall_weighted", float(recall_weighted))


def main() -> None:
    ap = argparse.ArgumentParser(description="Enrich Task1 runs with golden-set accuracy/recall and confusion matrix.")
    ap.add_argument("--tracking-uri", default="file:///workspace/mlruns_recovered")
    ap.add_argument("--experiment-id", default="325221045469690006")
    ap.add_argument("--golden-csv", default="/workspace/data_lake/datasets/task1_gold_labeled.csv")
    ap.add_argument("--out-dir", default="/workspace/artifacts/task1_golden_eval")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    texts, y_true = load_golden(Path(args.golden_csv), limit=args.limit)
    if not texts:
        raise RuntimeError("No golden rows loaded. Check golden CSV content.")

    runs = client.search_runs(
        experiment_ids=[args.experiment_id],
        max_results=5000,
        order_by=["attributes.start_time DESC"],
    )
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []

    for run in runs:
        run_id = run.info.run_id
        run_dir = Path(args.tracking_uri.replace("file://", "")) / args.experiment_id / run_id
        model_dir = run_dir / "artifacts" / "model"
        if not model_is_complete(model_dir):
            print(f"[skip] {run_id} model incomplete at {model_dir}")
            continue

        try:
            y_pred = predict_labels(model_dir, texts, batch_size=args.batch_size)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {run_id} failed to load/predict: {e}")
            continue

        if len(y_pred) != len(y_true):
            print(f"[skip] {run_id} length mismatch y_pred={len(y_pred)} y_true={len(y_true)}")
            continue

        acc = accuracy_score(y_true, y_pred)
        rec_macro = recall_score(y_true, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0)
        rec_weighted = recall_score(y_true, y_pred, labels=LABEL_ORDER, average="weighted", zero_division=0)

        cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
        cm_df = pd.DataFrame(cm, index=LABEL_ORDER, columns=LABEL_ORDER)

        run_out = out_root / run_id
        run_out.mkdir(parents=True, exist_ok=True)
        cm_csv = run_out / "confusion_matrix_by_class.csv"
        cm_df.to_csv(cm_csv, index=True)

        metrics_json = run_out / "gold_eval_metrics.json"
        metrics_payload = {
            "run_id": run_id,
            "n_samples": len(y_true),
            "eval_gold_accuracy": float(acc),
            "eval_gold_recall_macro": float(rec_macro),
            "eval_gold_recall_weighted": float(rec_weighted),
        }
        metrics_json.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        log_metrics_to_run(client, run_id, acc, rec_macro, rec_weighted)
        client.log_artifact(run_id, str(cm_csv), artifact_path="gold_eval")
        client.log_artifact(run_id, str(metrics_json), artifact_path="gold_eval")

        summary_rows.append(metrics_payload)
        print(
            f"[ok] {run_id} acc={acc:.4f} recall_macro={rec_macro:.4f} recall_weighted={rec_weighted:.4f}"
        )

    if not summary_rows:
        raise RuntimeError("No runs were evaluated. Check model artifacts completeness.")

    summary_df = pd.DataFrame(summary_rows).sort_values("eval_gold_accuracy", ascending=False)
    summary_csv = out_root / "task1_golden_eval_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[done] summary: {summary_csv}")


if __name__ == "__main__":
    main()

