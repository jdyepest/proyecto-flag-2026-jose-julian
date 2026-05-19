#!/usr/bin/env python3
"""Detailed golden-set evaluation with weighted metrics and confusion matrices."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "app" / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.segmentation import analyze_segments  # noqa: E402
from services.contributions import analyze_contributions  # noqa: E402

TASK1_LABELS = ["INTRO", "BACK", "METH", "RES", "DISC", "CONTR", "LIM", "CONC"]


@dataclass
class TaskMetrics:
    task: str
    model: str
    encoder_variant: str
    n: int
    accuracy: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    total_time_s: float
    time_per_doc_s: float


def _clean_text(text: str) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def _normalize_label(label: Any) -> str:
    if label is None:
        return ""
    lbl = str(label).strip().upper()
    return "RES" if lbl == "RESU" else lbl


def _safe_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"true", "1", "yes", "y", "si", "sí"}


def _random_wrong_task1_label(gold: str, seed: int) -> str:
    pool = [lbl for lbl in TASK1_LABELS if lbl != gold]
    if not pool:
        pool = TASK1_LABELS[:]
    return random.Random(seed).choice(pool)


def _save_confusion_plot(
    matrix: list[list[int]],
    labels: list[str],
    out_png: Path,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return

    arr = np.array(matrix)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(arr, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, int(arr[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _compute_metrics(
    task: str,
    model: str,
    encoder_variant: str,
    y_true: list[Any],
    y_pred: list[Any],
    elapsed_s: float,
) -> TaskMetrics:
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    p_m, r_m, f_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    n = len(y_true)
    return TaskMetrics(
        task=task,
        model=model,
        encoder_variant=encoder_variant,
        n=n,
        accuracy=float(acc),
        precision_weighted=float(p_w),
        recall_weighted=float(r_w),
        f1_weighted=float(f_w),
        precision_macro=float(p_m),
        recall_macro=float(r_m),
        f1_macro=float(f_m),
        total_time_s=float(round(elapsed_s, 4)),
        time_per_doc_s=float(round(elapsed_s / max(n, 1), 6)),
    )


def eval_task1(
    df: pd.DataFrame,
    model: str,
    encoder_variant: str,
    batch_size: int,
) -> tuple[TaskMetrics, pd.DataFrame, list[str], list[list[int]]]:
    rows = [
        {"text": _clean_text(t), "gold": _normalize_label(lbl)}
        for t, lbl in zip(df["text"], df["gold_label"], strict=False)
    ]
    y_true: list[str] = []
    y_pred: list[str] = []
    debug_rows: list[dict[str, Any]] = []

    started = time.perf_counter()
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        text_blob = "\n\n".join([r["text"] for r in batch])
        try:
            result = analyze_segments(text_blob, model, encoder_variant=encoder_variant)
            segments = result.get("segments") or []
        except Exception:
            segments = []
        for j, item in enumerate(batch):
            gold = item["gold"]
            row_seed = i + j + 17
            pred = ""
            if j < len(segments) and isinstance(segments[j], dict):
                pred = _normalize_label(segments[j].get("label"))

            if j >= len(segments) or pred not in TASK1_LABELS:
                # If model output is missing/invalid, force an incorrect random class.
                pred = _random_wrong_task1_label(gold, row_seed)

            y_true.append(gold)
            y_pred.append(pred)
            debug_rows.append(
                {
                    "task": "task1",
                    "index": i + j,
                    "gold": gold,
                    "pred": pred,
                    "ok": int(gold == pred),
                    "text": item["text"],
                }
            )
    elapsed = time.perf_counter() - started
    metrics = _compute_metrics("task1", model, encoder_variant, y_true, y_pred, elapsed)

    labels = TASK1_LABELS[:]
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return metrics, pd.DataFrame(debug_rows), labels, cm


def eval_task2(
    df: pd.DataFrame,
    model: str,
    encoder_variant: str,
    batch_size: int,
) -> tuple[TaskMetrics, pd.DataFrame, list[str], list[list[int]]]:
    rows = [
        {
            "text": _clean_text(t),
            "gold_bool": _safe_bool(lbl),
            "gold_rhet": _normalize_label(lbl_r),
        }
        for t, lbl, lbl_r in zip(
            df["text"], df["gold_is_contribution"], df["gold_rhetorical_label"], strict=False
        )
    ]
    y_true: list[bool] = []
    y_pred: list[bool] = []
    debug_rows: list[dict[str, Any]] = []

    started = time.perf_counter()
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        segments = [
            {"paragraph_index": idx, "text": r["text"], "label": r["gold_rhet"]}
            for idx, r in enumerate(batch)
        ]
        try:
            result = analyze_contributions(segments, model, encoder_variant=encoder_variant)
            frags = result.get("fragments") or []
        except Exception:
            frags = []
        for j, item in enumerate(batch):
            gold = item["gold_bool"]
            if j < len(frags) and isinstance(frags[j], dict) and ("is_contribution" in frags[j]):
                pred = bool(frags[j].get("is_contribution"))
            else:
                # For binary task, if output is missing/invalid, force an incorrect class.
                pred = not gold
            y_true.append(gold)
            y_pred.append(pred)
            debug_rows.append(
                {
                    "task": "task2",
                    "index": i + j,
                    "gold": int(gold),
                    "pred": int(pred),
                    "ok": int(gold == pred),
                    "text": item["text"],
                }
            )
    elapsed = time.perf_counter() - started
    metrics = _compute_metrics("task2", model, encoder_variant, y_true, y_pred, elapsed)

    labels_bool = [False, True]
    cm = confusion_matrix(y_true, y_pred, labels=labels_bool).tolist()
    labels = ["0", "1"]
    return metrics, pd.DataFrame(debug_rows), labels, cm


def main() -> None:
    parser = argparse.ArgumentParser(description="Detailed golden-set eval with confusion matrix.")
    parser.add_argument("--task", choices=["task1", "task2", "both"], default="both")
    parser.add_argument("--model", choices=["encoder", "llm", "api"], default="encoder")
    parser.add_argument("--encoder-variant", choices=["roberta", "scibert"], default="roberta")
    parser.add_argument("--task1-path", default=str(REPO_ROOT / "data_lake" / "datasets" / "task1_gold_labeled.csv"))
    parser.add_argument("--task2-path", default=str(REPO_ROOT / "data_lake" / "datasets" / "task2_gold_labeled.csv"))
    parser.add_argument("--batch-task1", type=int, default=200)
    parser.add_argument("--batch-task2", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts" / "gold_eval_detailed"))
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument("--mlflow-experiment", default="golden-set-detailed")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[TaskMetrics] = []
    all_preds: list[pd.DataFrame] = []

    if args.task in {"task1", "both"}:
        df1 = pd.read_csv(args.task1_path)
        if args.limit:
            df1 = df1.head(args.limit)
        m1, pred1, labels1, cm1 = eval_task1(df1, args.model, args.encoder_variant, args.batch_task1)
        all_metrics.append(m1)
        all_preds.append(pred1)
        cm1_df = pd.DataFrame(cm1, index=labels1, columns=labels1)
        cm1_df.to_csv(out_dir / "task1_confusion_matrix.csv")
        _save_confusion_plot(cm1, labels1, out_dir / "task1_confusion_matrix.png", "Task1 Confusion Matrix")

    if args.task in {"task2", "both"}:
        df2 = pd.read_csv(args.task2_path)
        if args.limit:
            df2 = df2.head(args.limit)
        m2, pred2, labels2, cm2 = eval_task2(df2, args.model, args.encoder_variant, args.batch_task2)
        all_metrics.append(m2)
        all_preds.append(pred2)
        cm2_df = pd.DataFrame(cm2, index=labels2, columns=labels2)
        cm2_df.to_csv(out_dir / "task2_confusion_matrix.csv")
        _save_confusion_plot(cm2, labels2, out_dir / "task2_confusion_matrix.png", "Task2 Confusion Matrix")

    metrics_df = pd.DataFrame([m.__dict__ for m in all_metrics])
    metrics_df.to_csv(out_dir / "metrics_summary.csv", index=False)
    (out_dir / "metrics_summary.json").write_text(
        json.dumps([m.__dict__ for m in all_metrics], ensure_ascii=False, indent=2)
    )

    if all_preds:
        pd.concat(all_preds, ignore_index=True).to_csv(out_dir / "predictions_debug.csv", index=False)

    if args.log_mlflow:
        try:
            import mlflow
        except Exception as e:
            print(f"[MLflow] no disponible: {e}")
        else:
            mlflow.set_experiment(args.mlflow_experiment)
            for row in [m.__dict__ for m in all_metrics]:
                run_name = f"{row['task']}-{row['model']}-{row['encoder_variant']}-detailed"
                with mlflow.start_run(run_name=run_name):
                    mlflow.log_params(
                        {
                            "task": row["task"],
                            "model": row["model"],
                            "encoder_variant": row["encoder_variant"],
                            "n": int(row["n"]),
                        }
                    )
                    for key in [
                        "accuracy",
                        "precision_weighted",
                        "recall_weighted",
                        "f1_weighted",
                        "precision_macro",
                        "recall_macro",
                        "f1_macro",
                        "total_time_s",
                        "time_per_doc_s",
                    ]:
                        mlflow.log_metric(key, float(row[key]))

    print(f"[OK] results in: {out_dir}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
