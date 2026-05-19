#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _read_yaml_name(meta_path: Path) -> str:
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("name:"):
            return line.split(":", 1)[1].strip().strip("'").strip('"')
    return "migrated-runs"


def _locate_run(source_root: Path, run_id: str) -> tuple[str, Path]:
    for exp_dir in source_root.iterdir():
        if not exp_dir.is_dir():
            continue
        if not exp_dir.name.isdigit():
            continue
        run_dir = exp_dir / run_id
        if run_dir.is_dir():
            return exp_dir.name, run_dir
    raise FileNotFoundError(f"run_id not found under {source_root}: {run_id}")


def _log_params(client: MlflowClient, new_run_id: str, run_dir: Path) -> None:
    params_dir = run_dir / "params"
    if not params_dir.is_dir():
        return
    for p in sorted(params_dir.iterdir()):
        if p.is_file():
            client.log_param(new_run_id, p.name, _read_text(p))


def _log_tags(client: MlflowClient, new_run_id: str, run_dir: Path, source_run_id: str) -> None:
    tags_dir = run_dir / "tags"
    client.set_tag(new_run_id, "source_run_id", source_run_id)
    if not tags_dir.is_dir():
        return
    for t in sorted(tags_dir.iterdir()):
        if not t.is_file():
            continue
        key = t.name
        if key.startswith("mlflow."):
            continue
        client.set_tag(new_run_id, key, _read_text(t))


def _log_metrics(client: MlflowClient, new_run_id: str, run_dir: Path) -> int:
    metrics_dir = run_dir / "metrics"
    if not metrics_dir.is_dir():
        return 0
    n = 0
    for mfile in sorted(metrics_dir.iterdir()):
        if not mfile.is_file():
            continue
        key = mfile.name
        for raw in mfile.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split()
            if len(parts) < 2:
                continue
            timestamp = int(float(parts[0]))
            value = float(parts[1])
            step = int(float(parts[2])) if len(parts) >= 3 else 0
            client.log_metric(new_run_id, key, value, timestamp=timestamp, step=step)
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Push one local MLflow file-store run to remote MLflow tracking server.")
    ap.add_argument("--source-root", default="mlruns_recovered")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--tracking-uri", default="http://localhost:5006")
    ap.add_argument("--experiment-name", default="")
    ap.add_argument("--artifact-path", default="", help="Optional subpath for artifacts")
    args = ap.parse_args()

    source_root = Path(args.source_root).resolve()
    exp_id, run_dir = _locate_run(source_root, args.run_id)

    exp_name = args.experiment_name
    if not exp_name:
        exp_meta = source_root / exp_id / "meta.yaml"
        exp_name = _read_yaml_name(exp_meta) if exp_meta.is_file() else f"migrated-{exp_id}"

    run_name = args.run_id
    run_name_tag = run_dir / "tags" / "mlflow.runName"
    if run_name_tag.is_file():
        run_name = _read_text(run_name_tag) or run_name

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(exp_name)
    client = MlflowClient()

    with mlflow.start_run(run_name=run_name) as new_run:
        new_run_id = new_run.info.run_id
        _log_params(client, new_run_id, run_dir)
        _log_tags(client, new_run_id, run_dir, args.run_id)
        metric_points = _log_metrics(client, new_run_id, run_dir)

        artifacts_dir = run_dir / "artifacts"
        if artifacts_dir.is_dir():
            if args.artifact_path:
                mlflow.log_artifacts(str(artifacts_dir), artifact_path=args.artifact_path)
            else:
                mlflow.log_artifacts(str(artifacts_dir))

    print(f"source_run_id={args.run_id}")
    print(f"new_run_id={new_run_id}")
    print(f"experiment={exp_name}")
    print(f"metric_points_logged={metric_points}")
    print(f"model_uri=runs:/{new_run_id}/model")


if __name__ == "__main__":
    main()

