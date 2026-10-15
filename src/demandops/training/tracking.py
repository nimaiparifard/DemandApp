"""MLflow helpers: provenance logging, registry, champion/challenger aliases."""

from __future__ import annotations

import contextlib
import hashlib
import subprocess
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def data_fingerprint(panel: pd.DataFrame) -> str:
    """Cheap content hash of the training panel (DVC hash is logged separately when available)."""
    sample = panel[["id", "d", "sales"]].fillna(-1)
    h = hashlib.sha256(pd.util.hash_pandas_object(sample, index=False).to_numpy().tobytes())
    return h.hexdigest()[:16]


def dvc_hash(dvc_file: Path = Path("data/raw/m5.dvc")) -> str:
    try:
        for line in dvc_file.read_text().splitlines():
            if "md5:" in line:
                return line.split("md5:")[1].strip()
    except OSError:
        pass
    return "untracked"


def get_alias_version(client: MlflowClient, name: str, alias: str) -> ModelVersion | None:
    try:
        return client.get_model_version_by_alias(name, alias)
    except mlflow.exceptions.MlflowException:
        return None


def champion_metrics(
    client: MlflowClient, name: str, alias: str = "champion"
) -> dict[str, float] | None:
    mv = get_alias_version(client, name, alias)
    if mv is None or mv.run_id is None:
        return None
    run = client.get_run(mv.run_id)
    return {k: v for k, v in run.data.metrics.items() if k.startswith("cv_")}


def register_and_alias(
    client: MlflowClient, run_id: str, name: str, alias: str, artifact: str = "model"
) -> str:
    with contextlib.suppress(mlflow.exceptions.MlflowException):  # already exists
        client.create_registered_model(name)
    mv = client.create_model_version(name, f"runs:/{run_id}/{artifact}", run_id)
    client.set_registered_model_alias(name, alias, mv.version)
    return str(mv.version)


def promote(
    client: MlflowClient, name: str, from_alias: str = "challenger", to_alias: str = "champion"
) -> str | None:
    mv = get_alias_version(client, name, from_alias)
    if mv is None:
        return None
    client.set_registered_model_alias(name, to_alias, mv.version)
    return str(mv.version)
