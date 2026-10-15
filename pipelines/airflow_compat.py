"""One import surface for Airflow 3 (airflow.sdk) and Airflow 2.x (airflow.decorators)."""

from __future__ import annotations

try:  # Airflow 3
    from airflow.sdk import dag, get_current_context, task
except ImportError:  # Airflow 2.x
    from airflow.decorators import dag, task  # type: ignore[no-redef]
    from airflow.operators.python import get_current_context  # type: ignore[no-redef]

try:  # Airflow 3 (standard provider)
    from airflow.providers.standard.sensors.external_task import ExternalTaskSensor
except ImportError:  # Airflow 2.x
    from airflow.sensors.external_task import ExternalTaskSensor  # type: ignore[no-redef]

__all__ = ["ExternalTaskSensor", "dag", "get_current_context", "task"]
