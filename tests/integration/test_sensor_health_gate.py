from pathlib import Path

import pytest

from configuration.system import load_system_stack
from scripts.evaluate_sensor_health_gate import run_fault_matrix

pytestmark = pytest.mark.integration


_CONFIG = Path(__file__).parents[2] / "configs" / "system.yaml"


def test_1000_fault_cases_have_zero_false_accepts_and_maximum_braking():
    stack = load_system_stack(_CONFIG)

    metrics = run_fault_matrix(stack, samples=1000)

    assert metrics["total_faults"] == 1000
    assert metrics["rejected_faults"] == 1000
    assert metrics["false_accepts"] == 0
    assert metrics["detection_rate"] == 1.0
    assert metrics["maximum_brake_count"] == 1000
    assert metrics["fail_safe_rate"] == 1.0
    assert metrics["latency_p95_ms"] < metrics["control_budget_ms"]
    assert metrics["control_budget_pass"] is True
    assert all(count > 0 for count in metrics["injected_fault_counts"].values())
