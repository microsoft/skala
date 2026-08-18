# SPDX-License-Identifier: MIT

import dataclasses
import json
import math
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy
import pytest

from skala.benchmark.models import Molecule
from skala.benchmark.protocol import Device, FunctionalKind, FunctionalSpec
from skala.benchmark.runner import (
    RunConfig,
    RunResult,
    _build_mf,
    _build_mol,
    _release_gpu4pyscf_global_cache,
    _release_warmup_memory,
    _use_torch_memory_pool_in_cupy,
    main,
    run_worker,
)


def test_run_worker_h2_cation() -> None:
    config = RunConfig(
        molecule=Molecule(
            atomic_numbers=[1, 1],
            geometry_bohr=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
            charge=1,
            multiplicity=2,
        ),
        basis="sto-3g",
        functional=FunctionalSpec("pbe", FunctionalKind.NATIVE),
        device=Device.CPU,
        density_fit=False,
    )

    outcome = run_worker(config)

    assert isinstance(outcome, RunResult)
    assert outcome.is_converged
    assert outcome.num_atoms == 2
    assert outcome.num_electrons == 1
    assert math.isfinite(outcome.total_energy)
    assert outcome.wall_time_ms > 0


def test_run_worker_records_one_entry_per_cycle() -> None:
    config = RunConfig(
        molecule=Molecule(
            atomic_numbers=[1, 1],
            geometry_bohr=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
        ),
        basis="sto-3g",
        functional=FunctionalSpec("pbe", FunctionalKind.NATIVE),
        device=Device.CPU,
        density_fit=False,
    )

    outcome = run_worker(config)

    assert isinstance(outcome, RunResult)
    assert outcome.num_scf_iterations == len(outcome.cycles)
    assert outcome.cycles
    for cycle in outcome.cycles:
        # The measured layers nest: the functional within the quadrature.
        assert cycle["xc_eval_ms"] <= cycle["numint_ms"] <= cycle["wall_ms"]
    # libxc is timed at the same layer as the network, so it is not zero.
    assert all(cycle["xc_eval_ms"] > 0.0 for cycle in outcome.cycles)
    assert all(cycle["xc_eval_calls"] > 0 for cycle in outcome.cycles)


def test_run_worker_phases_partition_the_worker() -> None:
    """load + warmup + build + kernel must account for the whole worker."""
    config = RunConfig(
        molecule=Molecule(
            atomic_numbers=[1, 1],
            geometry_bohr=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
        ),
        basis="sto-3g",
        functional=FunctionalSpec("pbe", FunctionalKind.NATIVE),
        device=Device.CPU,
        density_fit=False,
    )

    outcome = run_worker(config)

    assert isinstance(outcome, RunResult)
    phases = (
        outcome.load_ms + outcome.warmup_ms + outcome.build_ms + outcome.kernel_time_ms
    )
    assert phases == pytest.approx(outcome.worker_ms, abs=1e-6)
    # The warmup SCF is real work and must be recorded, not silently dropped.
    assert outcome.warmup_ms > 0.0
    assert outcome.wall_time_ms == pytest.approx(
        outcome.build_ms + outcome.kernel_time_ms, abs=1e-6
    )


def test_run_worker_phases_partition_the_kernel() -> None:
    config = RunConfig(
        molecule=Molecule(
            atomic_numbers=[1, 1],
            geometry_bohr=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
        ),
        basis="sto-3g",
        functional=FunctionalSpec("pbe", FunctionalKind.NATIVE),
        device=Device.CPU,
        density_fit=False,
    )

    outcome = run_worker(config)

    assert isinstance(outcome, RunResult)
    accounted = (
        outcome.setup_ms
        + outcome.finalize_ms
        + sum(cycle["wall_ms"] for cycle in outcome.cycles)
    )
    assert accounted == pytest.approx(outcome.kernel_time_ms, rel=0.02)


def test_run_worker_cycles_are_json_serializable() -> None:
    config = RunConfig(
        molecule=Molecule(
            atomic_numbers=[1, 1],
            geometry_bohr=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
        ),
        basis="sto-3g",
        functional=FunctionalSpec("pbe", FunctionalKind.NATIVE),
        device=Device.CPU,
        density_fit=False,
    )

    outcome = run_worker(config)

    assert isinstance(outcome, RunResult)
    # The worker hands its result to the orchestrator as JSON.
    assert json.loads(json.dumps(dataclasses.asdict(outcome)))["cycles"]


def test_gpu_runner_routes_cupy_through_torch_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    package = ModuleType("pytorch_pfn_extras")
    package.cuda = SimpleNamespace(
        use_torch_mempool_in_cupy=lambda: calls.append("configured")
    )
    monkeypatch.setitem(sys.modules, "pytorch_pfn_extras", package)

    _use_torch_memory_pool_in_cupy()

    assert calls == ["configured"]


def test_gpu_runner_releases_gpu4pyscf_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = {"cartesian-to-spherical": object()}
    module = ModuleType("gpu4pyscf.gto.mole")
    module._c2s = cache
    monkeypatch.setitem(sys.modules, "gpu4pyscf.gto.mole", module)
    collect = Mock()
    monkeypatch.setattr("skala.benchmark.runner.gc.collect", collect)

    _release_gpu4pyscf_global_cache(Device.GPU)

    assert cache == {}
    collect.assert_called_once_with()


def test_gpu_runner_releases_warmup_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collect = Mock()
    empty_cache = Mock()
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(empty_cache=empty_cache)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr("skala.benchmark.runner.gc.collect", collect)

    _release_warmup_memory(Device.GPU)

    collect.assert_called_once_with()
    empty_cache.assert_called_once_with()


def test_cpu_runner_releases_warmup_memory_without_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collect = Mock()
    monkeypatch.setattr("skala.benchmark.runner.gc.collect", collect)

    _release_warmup_memory(Device.CPU)

    collect.assert_called_once_with()


def test_run_config_json_round_trip(tmp_path: Path) -> None:
    config = RunConfig(
        molecule=Molecule(
            atomic_numbers=[1],
            geometry_bohr=[[0.0, 0.0, 0.0]],
            multiplicity=2,
        ),
        basis="sto-3g",
        functional=FunctionalSpec("r2scan", FunctionalKind.NATIVE),
        device=Device.GPU,
    )
    path = tmp_path / "run.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")

    assert RunConfig.from_json(path) == config


def test_worker_writes_an_error_result_for_a_failed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed calculation must still leave a result file for the orchestrator."""
    config = RunConfig(
        molecule=Molecule(
            atomic_numbers=[1],
            geometry_bohr=[[0.0, 0.0, 0.0]],
            multiplicity=2,
        ),
        basis="sto-3g",
        functional=FunctionalSpec("pbe", FunctionalKind.NATIVE),
        device=Device.CPU,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    result_path = tmp_path / "result.json"

    def _explode(*_args: object, **_kwargs: object) -> RunResult:
        raise RuntimeError("boom")

    monkeypatch.setattr("skala.benchmark.runner._build_mol", _explode)

    main(["--config", str(config_path), "--result", str(result_path)])

    assert "boom" in json.loads(result_path.read_text(encoding="utf-8"))["error"]


def test_classical_functionals_time_libxc_as_the_xc_eval_layer() -> None:
    config = RunConfig(
        molecule=Molecule(
            atomic_numbers=[1, 1],
            geometry_bohr=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
        ),
        basis="sto-3g",
        functional=FunctionalSpec("r2scan", FunctionalKind.NATIVE),
        device=Device.CPU,
        density_fit=False,
    )

    outcome = run_worker(config)

    assert isinstance(outcome, RunResult)
    assert outcome.is_converged
    # libxc returns energy and derivative together, so there is no separable
    # forward, but the combined call is a genuine fraction of the quadrature.
    for cycle in outcome.cycles:
        assert 0.0 < cycle["xc_eval_ms"] < cycle["numint_ms"]
        assert cycle["forward_ms"] == 0.0
        assert cycle["backward_ms"] == 0.0


@pytest.mark.parametrize("functional", ["r2scan", "b3lyp", "b3lyp5"])
def test_traditional_benchmarks_use_stock_pyscf(functional: str) -> None:
    molecule = Molecule(
        atomic_numbers=[1, 1],
        geometry_bohr=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
    )
    config = RunConfig(
        molecule=molecule,
        basis="sto-3g",
        functional=FunctionalSpec(functional, FunctionalKind.NATIVE),
        device=Device.CPU,
        density_fit=False,
    )

    mean_field = _build_mf(_build_mol(molecule, config.basis), config, Device.CPU)

    assert mean_field.__class__.__module__.startswith("pyscf.dft.")
    assert mean_field.xc == functional


def test_the_kernel_region_imports_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """No module may be imported between the kernel timestamps.

    A lazy import evaluated inside the measured region charges seconds of module
    loading to the calculation. The timeline selector is a plain string for this
    reason, so a classical functional on CPU never imports torch at all.
    """
    config = RunConfig(
        molecule=Molecule(
            atomic_numbers=[1, 1],
            geometry_bohr=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
        ),
        basis="sto-3g",
        functional=FunctionalSpec("pbe", FunctionalKind.NATIVE),
        device=Device.CPU,
        density_fit=False,
    )

    outcome = run_worker(config)

    assert isinstance(outcome, RunResult)
    phases = (
        outcome.setup_ms
        + outcome.finalize_ms
        + sum(cycle["wall_ms"] for cycle in outcome.cycles)
    )
    # A stray import shows up as time inside the kernel that no phase claims.
    assert phases == pytest.approx(outcome.kernel_time_ms, abs=5.0)


def test_boot_is_measured_against_the_parents_clock() -> None:
    from skala.benchmark.runner import _MODULE_READY_AT, _boot_ms

    # A launch timestamp from before this module loaded yields a positive boot.
    assert _boot_ms(_MODULE_READY_AT - 0.25) == pytest.approx(250.0, abs=1.0)
    # No timestamp, or clocks that are not comparable, yield nothing rather than
    # a nonsensical negative duration.
    assert _boot_ms(None) is None
    assert _boot_ms(_MODULE_READY_AT + 1.0) is None


def test_timeline_selector_needs_no_torch() -> None:
    from skala.benchmark.runner import _timeline_device

    assert _timeline_device(Device.CPU) == "cpu"
    assert _timeline_device(Device.GPU) == "cuda"


def _mean_field_with_costs(costs: list[float]) -> SimpleNamespace:
    """A mean field whose XC evaluation takes the given times, in order."""
    clock = iter(costs)

    def evaluate(mol: object, grids: object, xc: object, dm: object) -> None:
        time.sleep(next(clock))

    return SimpleNamespace(
        mol=object(),
        grids=object(),
        xc="custom",
        init_guess="minao",
        get_init_guess=lambda mol, guess: numpy.zeros((2, 2, 2)),
        _numint=SimpleNamespace(nr_uks=evaluate),
    )


def test_warmup_waits_for_a_late_compile() -> None:
    from skala.benchmark.runner import _settle_xc_evaluation

    # TorchScript compiles on a later evaluation, so the cost falls, spikes,
    # then settles. Stopping on the falling edge would leave the spike to land
    # in the measured run.
    # Settling needs two consecutive evaluations that agree, so the spike at
    # the third is followed by the fourth and fifth before it stops.
    mf = _mean_field_with_costs([0.04, 0.02, 0.08, 0.005, 0.005, 0.005])
    assert len(_settle_xc_evaluation(mf, Device.CPU)) == 5


def test_warmup_stops_once_the_cost_is_stable() -> None:
    from skala.benchmark.runner import _settle_xc_evaluation

    # A classical functional has no compile to wait for, so it stops as soon as
    # the minimum number of evaluations has confirmed a stable cost.
    mf = _mean_field_with_costs([0.01] * 6)
    assert len(_settle_xc_evaluation(mf, Device.CPU)) == 3


def test_warmup_gives_up_rather_than_looping_forever() -> None:
    from skala.benchmark.runner import _settle_xc_evaluation

    # A machine noisy enough that consecutive evaluations never agree must not
    # trap the worker in the warmup.
    mf = _mean_field_with_costs([0.001, 0.02] * 8)
    assert len(_settle_xc_evaluation(mf, Device.CPU, max_evaluations=4)) == 4


def test_warmup_outlasts_a_larger_profiling_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skala.benchmark import runner
    from skala.benchmark.runner import _settle_xc_evaluation

    # With num_profiled_runs = 4 the compiles land on the 5th and 9th
    # evaluations, and the ones in between run at full speed. Stopping at the
    # first agreeing pair would hand both compiles to the measured run.
    monkeypatch.setattr(runner, "_num_profiled_runs", lambda default=1: 4)
    costs = [0.05, 0.005, 0.005, 0.005, 0.02, 0.006, 0.006, 0.006, 0.09]
    mf = _mean_field_with_costs(costs + [0.005] * 4)
    assert len(_settle_xc_evaluation(mf, Device.CPU)) >= len(costs)
