import pytest

from pipelines.gates import GateConfig, evaluate_gates

CHAMP = {"wrmsse": 0.80, "bias": 0.01}
GOOD = {"wrmsse": 0.70, "bias": 0.02}


def test_all_gates_pass():
    r = evaluate_gates(GOOD, CHAMP, p95_latency_ms=40, dq_failures=0)
    assert r.passed and r.decision == "promote challenger" and r.reasons == []


def test_improvement_below_one_percent_keeps_champion():
    r = evaluate_gates({"wrmsse": 0.795, "bias": 0.0}, CHAMP, 40, 0)  # 0.6% better
    assert not r.passed and r.decision == "kept champion"
    assert any("WRMSSE" in x for x in r.reasons)


def test_exactly_one_percent_passes():
    r = evaluate_gates({"wrmsse": 0.80 * 0.99, "bias": 0.0}, CHAMP, 40, 0)
    assert r.passed


def test_worse_model_fails():
    assert not evaluate_gates({"wrmsse": 0.9, "bias": 0.0}, CHAMP, 40, 0).passed


@pytest.mark.parametrize("bias", [0.05, -0.05, 0.2, -0.3])
def test_bias_gate_is_symmetric_and_strict(bias):
    r = evaluate_gates({"wrmsse": 0.5, "bias": bias}, CHAMP, 40, 0)
    assert not r.passed and any("bias" in x for x in r.reasons)


def test_latency_gate():
    r = evaluate_gates(GOOD, CHAMP, p95_latency_ms=100.0, dq_failures=0)
    assert not r.passed and any("latency" in x for x in r.reasons)


def test_data_quality_gate():
    r = evaluate_gates(GOOD, CHAMP, 40, dq_failures=2)
    assert not r.passed and any("data-quality" in x for x in r.reasons)


def test_first_model_without_champion_skips_accuracy_gate_only():
    assert evaluate_gates(GOOD, None, 40, 0).passed
    assert not evaluate_gates({"wrmsse": 0.7, "bias": 0.5}, None, 40, 0).passed


def test_multiple_failures_are_all_reported():
    r = evaluate_gates({"wrmsse": 0.9, "bias": 0.2}, CHAMP, 500, 1)
    assert len(r.reasons) == 4


def test_custom_thresholds():
    cfg = GateConfig(min_wrmsse_improvement=0.2)
    assert not evaluate_gates(GOOD, CHAMP, 40, 0, cfg).passed  # 12.5% < 20%
