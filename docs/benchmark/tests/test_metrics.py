# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
from skala_benchmark.metrics import (
    composition_times,
    metric_value,
    warmup_ratio,
)


def test_report_metrics_use_the_steady_state_and_partition_the_cycle() -> None:
    row = {
        "is_converged": True,
        "status": "ok",
        "kernel_time_ms": 1000.0,
        "setup_ms": 60.0,
        "num_scf_iterations": 4,
        "cycles": [
            {
                "cycle": 0,
                "wall_ms": 400.0,
                "veff_ms": 360.0,
                "numint_ms": 320.0,
                "xc_eval_ms": 160.0,
                "forward_ms": 64.0,
            },
            {
                "cycle": 1,
                "wall_ms": 100.0,
                "veff_ms": 90.0,
                "numint_ms": 80.0,
                "xc_eval_ms": 30.0,
                "forward_ms": 12.0,
            },
            {
                "cycle": 2,
                "wall_ms": 100.0,
                "veff_ms": 90.0,
                "numint_ms": 80.0,
                "xc_eval_ms": 30.0,
                "forward_ms": 12.0,
            },
        ],
    }

    assert metric_value(row, "cycle") == pytest.approx(100.0)
    assert metric_value(row, "numint") == pytest.approx(80.0)
    assert metric_value(row, "xc_eval") == pytest.approx(12.0)
    assert metric_value(row, "jk") == pytest.approx(10.0)
    assert warmup_ratio(row) == pytest.approx(4.0)

    composition = composition_times(row)
    assert composition == pytest.approx(
        {
            "xc_eval": 12.0,
            "numint_rest": 68.0,
            "jk": 10.0,
            "cycle_rest": 10.0,
        }
    )
    assert composition is not None
    assert sum(composition.values()) == pytest.approx(100.0)
