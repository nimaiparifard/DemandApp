"""Promotion gates: a challenger replaces the champion ONLY if every gate passes.

Pure function, no I/O, so it is unit-tested exhaustively and reused by the Airflow DAG.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GateConfig:
    min_wrmsse_improvement: float = 0.01  # >= 1% relative improvement
    max_abs_bias: float = 0.05  # |bias| < 5%
    max_p95_latency_ms: float = 100.0  # staging p95 < 100 ms


@dataclass
class GateResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def decision(self) -> str:
        return "promote challenger" if self.passed else "kept champion"


def evaluate_gates(
    challenger: dict[str, float],
    champion: dict[str, float] | None,
    p95_latency_ms: float,
    dq_failures: int,
    cfg: GateConfig = GateConfig(),
) -> GateResult:
    """challenger / champion: dicts with at least `wrmsse` and `bias` (backtest means).

    With no champion yet (first model) the accuracy-improvement gate is skipped; the
    bias, latency and data-quality gates still apply.
    """
    reasons: list[str] = []

    if champion is not None:
        base = champion["wrmsse"]
        improvement = (base - challenger["wrmsse"]) / base if base > 0 else 0.0
        if improvement < cfg.min_wrmsse_improvement:
            reasons.append(
                f"WRMSSE improvement {improvement:.2%} < required {cfg.min_wrmsse_improvement:.2%}"
            )
    if abs(challenger["bias"]) >= cfg.max_abs_bias:
        reasons.append(f"|bias| {abs(challenger['bias']):.2%} >= {cfg.max_abs_bias:.2%}")
    if p95_latency_ms >= cfg.max_p95_latency_ms:
        reasons.append(f"p95 latency {p95_latency_ms:.1f} ms >= {cfg.max_p95_latency_ms:.1f} ms")
    if dq_failures > 0:
        reasons.append(f"{dq_failures} data-quality failure(s) in the training window")

    return GateResult(passed=not reasons, reasons=reasons)
