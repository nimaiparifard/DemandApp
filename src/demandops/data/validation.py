"""Data-quality gate (DAG 1 / CI). Raises on failure so the pipeline STOPS."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from demandops.data.schemas import PanelSchema


class DataQualityError(RuntimeError):
    def __init__(self, failures: list[str]) -> None:
        super().__init__("data quality checks failed: " + "; ".join(failures))
        self.failures = failures


@dataclass
class ValidationReport:
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def validate_panel(panel: pd.DataFrame, expected_last_day: int | None = None) -> ValidationReport:
    """Checks: schema, no negative sales, prices > 0, every series has the latest day,
    calendar has no gaps, no duplicated (id, d)."""
    rep = ValidationReport()
    try:
        PanelSchema.validate(panel, lazy=True)
    except Exception as exc:  # pandera SchemaErrors
        rep.failures.append("schema: " + " ".join(str(exc).split())[:200])

    if (panel["sales"] < 0).any():
        rep.failures.append("negative sales found")
    if (panel["sell_price"] <= 0).any():
        rep.failures.append("non-positive prices found")
    if panel.duplicated(["id", "d"]).any():
        rep.failures.append("duplicate (id, d) rows")

    days = pd.Index(sorted(panel["d"].unique()))
    if len(days) and (days.max() - days.min() + 1) != len(days):
        rep.failures.append("calendar has gaps")

    observed = panel[panel["sales"].notna()]
    last = expected_last_day if expected_last_day is not None else int(observed["d"].max())
    have_last = set(observed.loc[observed["d"] == last, "id"])
    missing = set(panel["id"]) - have_last
    if missing:
        rep.failures.append(f"{len(missing)} series missing day d_{last}")
    return rep


def assert_valid(panel: pd.DataFrame, expected_last_day: int | None = None) -> None:
    rep = validate_panel(panel, expected_last_day)
    if not rep.ok:
        raise DataQualityError(rep.failures)
