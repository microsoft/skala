"""Benchmark dense and screened AO integration across CPU and GPU backends.

The module compares numerical agreement, runtime, and peak allocations on a small
acene ladder. Profiling workloads run each route in an isolated process so allocator
state and backend initialization do not contaminate the measurements.
"""

from __future__ import annotations

import multiprocessing as mp
import tempfile
import traceback
from collections.abc import Callable, Iterator
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import pytest
import torch
from pyscf import dft, gto, lib
from pytest_benchmark.fixture import BenchmarkFixture
from torch.utils.dlpack import from_dlpack
from utils import force_ao_screening

from skala.functional import load_functional
from skala.functional.base import ExcFunctionalBase
from skala.pyscf.grids import SkalaGrids
from skala.pyscf.numint import SkalaNumInt

THREAD_COUNT = 4
MAX_MEMORY_MB = 2000
MEMORY_WORKER_TIMEOUT_SECONDS = 240

NAPHTHALENE = """
C -1.2280  0.7090 0.0
C -1.2280 -0.7090 0.0
C  0.0000  1.4180 0.0
C  0.0000 -1.4180 0.0
C  1.2280  0.7090 0.0
C  1.2280 -0.7090 0.0
C  2.4560  1.4180 0.0
C  2.4560 -1.4180 0.0
C  3.6840  0.7090 0.0
C  3.6840 -0.7090 0.0
H -2.1700  1.2530 0.0
H -2.1700 -1.2530 0.0
H  0.0000  2.5060 0.0
H  0.0000 -2.5060 0.0
H  2.4560  2.5060 0.0
H  2.4560 -2.5060 0.0
H  4.6260  1.2530 0.0
H  4.6260 -1.2530 0.0
"""

ANTHRACENE = """
C -1.2280  0.7090 0.0
C -1.2280 -0.7090 0.0
C  0.0000  1.4180 0.0
C  0.0000 -1.4180 0.0
C  1.2280  0.7090 0.0
C  1.2280 -0.7090 0.0
C  2.4560  1.4180 0.0
C  2.4560 -1.4180 0.0
C  3.6840  0.7090 0.0
C  3.6840 -0.7090 0.0
C  4.9120  1.4180 0.0
C  4.9120 -1.4180 0.0
C  6.1400  0.7090 0.0
C  6.1400 -0.7090 0.0
H -2.1700  1.2530 0.0
H -2.1700 -1.2530 0.0
H  0.0000  2.5060 0.0
H  0.0000 -2.5060 0.0
H  2.4560  2.5060 0.0
H  2.4560 -2.5060 0.0
H  4.9120  2.5060 0.0
H  4.9120 -2.5060 0.0
H  7.0820  1.2530 0.0
H  7.0820 -1.2530 0.0
"""

TETRACENE = """
C -1.2280  0.7090 0.0
C -1.2280 -0.7090 0.0
C  0.0000  1.4180 0.0
C  0.0000 -1.4180 0.0
C  1.2280  0.7090 0.0
C  1.2280 -0.7090 0.0
C  2.4560  1.4180 0.0
C  2.4560 -1.4180 0.0
C  3.6840  0.7090 0.0
C  3.6840 -0.7090 0.0
C  4.9120  1.4180 0.0
C  4.9120 -1.4180 0.0
C  6.1400  0.7090 0.0
C  6.1400 -0.7090 0.0
C  7.3680  1.4180 0.0
C  7.3680 -1.4180 0.0
C  8.5960  0.7090 0.0
C  8.5960 -0.7090 0.0
H -2.1700  1.2530 0.0
H -2.1700 -1.2530 0.0
H  0.0000  2.5060 0.0
H  0.0000 -2.5060 0.0
H  2.4560  2.5060 0.0
H  2.4560 -2.5060 0.0
H  4.9120  2.5060 0.0
H  4.9120 -2.5060 0.0
H  7.3680  2.5060 0.0
H  7.3680 -2.5060 0.0
H  9.5380  1.2530 0.0
H  9.5380 -1.2530 0.0
"""


class BenchmarkSpec(NamedTuple):
    name: str
    atoms: str


DeviceResult = tuple[float, float, object]


class BenchmarkCase(NamedTuple):
    backend: str
    mol: gto.Mole
    run: Callable[[], DeviceResult]
    synchronize: Callable[[], None]


BENCHMARK_SPECS = [
    pytest.param(BenchmarkSpec("naphthalene", NAPHTHALENE), id="naphthalene"),
    pytest.param(BenchmarkSpec("anthracene", ANTHRACENE), id="anthracene"),
    pytest.param(BenchmarkSpec("tetracene", TETRACENE), id="tetracene"),
]


def _make_benchmark_case(
    spec: BenchmarkSpec, functional: ExcFunctionalBase, backend: str
) -> BenchmarkCase:
    mol = gto.M(atom=spec.atoms, basis="def2-qzvpp", verbose=0)
    initial_dm = dft.RKS(mol).get_init_guess()

    if backend == "cpu":
        grids = SkalaGrids(mol)
        grids.level = 1
        grids.alignment = 1
        grids.build(sort_grids=False)
        dm: Any = initial_dm
        numint: Any = SkalaNumInt(functional)
        synchronize: Callable[[], None] = lambda: None  # noqa: E731
    elif backend == "cuda":
        import cupy

        from skala.gpu4pyscf import SkalaKS

        ks = SkalaKS(mol, xc=functional, with_dftd3=False)
        ks.grids.level = 1
        ks.grids.alignment = 1
        ks.grids.build(sort_grids=False)
        grids = ks.grids
        dm = cupy.asarray(initial_dm)
        numint = ks._numint
        synchronize = torch.cuda.synchronize
    else:
        raise ValueError(f"Unknown benchmark backend: {backend}")

    def run() -> DeviceResult:
        result = numint.nr_rks(
            mol,
            grids,
            None,
            dm,
            max_memory=MAX_MEMORY_MB,
        )
        synchronize()
        return cast(DeviceResult, result)

    return BenchmarkCase(backend, mol, run, synchronize)


@pytest.fixture(scope="module")
def fixed_cpu_threads() -> Iterator[None]:
    previous_pyscf_threads = lib.num_threads()
    previous_torch_threads = torch.get_num_threads()
    lib.num_threads(THREAD_COUNT)
    torch.set_num_threads(THREAD_COUNT)
    try:
        yield
    finally:
        torch.set_num_threads(previous_torch_threads)
        lib.num_threads(previous_pyscf_threads)


@pytest.fixture(scope="module", params=BENCHMARK_SPECS)
def benchmark_spec(request: pytest.FixtureRequest) -> BenchmarkSpec:
    return cast(BenchmarkSpec, request.param)


@pytest.fixture(scope="module")
def benchmark_case(
    benchmark_spec: BenchmarkSpec,
    request: pytest.FixtureRequest,
    fixed_cpu_threads: None,
    load_functional_cached: Callable[..., ExcFunctionalBase | str],
) -> BenchmarkCase:
    functional = load_functional_cached("skala-1.1")
    assert isinstance(functional, ExcFunctionalBase)
    return _make_benchmark_case(benchmark_spec, functional, "cpu")


@pytest.fixture(
    scope="module",
    params=["cpu", pytest.param("cuda", marks=pytest.mark.gpu)],
)
def device_benchmark_case(
    request: pytest.FixtureRequest,
    benchmark_spec: BenchmarkSpec,
    fixed_cpu_threads: None,
    load_functional_cached: Callable[..., ExcFunctionalBase | str],
) -> Iterator[BenchmarkCase]:
    backend = cast(str, request.param)
    if backend == "cpu":
        functional = load_functional_cached("skala-1.1")
        assert isinstance(functional, ExcFunctionalBase)
    else:
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available")
        pytest.importorskip("cupy")
        pytest.importorskip("gpu4pyscf")
        functional = load_functional_cached("skala-1.1", device=torch.device("cuda:0"))
        assert isinstance(functional, ExcFunctionalBase)

    case = _make_benchmark_case(benchmark_spec, functional, backend)
    with force_ao_screening(True):
        yield case


@pytest.fixture
def screened_case(benchmark_case: BenchmarkCase) -> Iterator[BenchmarkCase]:
    with force_ao_screening(True):
        yield benchmark_case


@pytest.fixture
def dense_case(benchmark_case: BenchmarkCase) -> Iterator[BenchmarkCase]:
    with force_ao_screening(False):
        yield benchmark_case


def _benchmark_device_xc(benchmark: BenchmarkFixture, case: BenchmarkCase) -> None:
    case.synchronize()
    pedantic = cast(Callable[..., object], benchmark.pedantic)
    pedantic(
        case.run,
        rounds=1,
        iterations=2,
    )


def _run_gpu_xc(spec: BenchmarkSpec, screened: bool) -> int:
    functional = load_functional("skala-1.1", device=torch.device("cuda:0"))
    assert isinstance(functional, ExcFunctionalBase)
    case = _make_benchmark_case(spec, functional, "cuda")

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with force_ao_screening(screened):
        case.run()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - baseline_bytes


def _memory_worker(
    spec: BenchmarkSpec,
    screened: bool,
    backend: str,
    control: Connection,
) -> None:
    """Measure one route in an isolated process and return its allocation peak."""
    try:
        lib.num_threads(THREAD_COUNT)
        torch.set_num_threads(THREAD_COUNT)
        if backend == "cpu":
            import memray

            functional = load_functional("skala-1.1")
            assert isinstance(functional, ExcFunctionalBase)
            case = _make_benchmark_case(spec, functional, "cpu")

            with tempfile.TemporaryDirectory() as tmpdir:
                profile_path = Path(tmpdir) / "allocations.bin"
                with (
                    force_ao_screening(screened),
                    memray.Tracker(profile_path),
                ):
                    case.run()
                peak_bytes = memray.FileReader(profile_path).metadata.peak_memory
        elif backend == "cuda":
            peak_bytes = _run_gpu_xc(spec, screened)
        else:
            raise ValueError(f"Unknown memory benchmark backend: {backend}")
        control.send(("done", peak_bytes))
    except Exception:  # noqa: BLE001 - forward worker failures to the parent
        control.send(("error", traceback.format_exc()))
    finally:
        control.close()


def _measure_peak_memory(spec: BenchmarkSpec, screened: bool, backend: str) -> int:
    """Return peak allocations for one isolated CPU or CUDA evaluation."""
    context = mp.get_context("spawn")
    control, worker_control = context.Pipe()
    worker = context.Process(
        target=_memory_worker,
        args=(spec, screened, backend, worker_control),
    )
    worker.start()
    worker_control.close()

    try:
        if not control.poll(MEMORY_WORKER_TIMEOUT_SECONDS):
            raise TimeoutError("Memory benchmark worker timed out")
        status, detail = cast(tuple[str, int | str], control.recv())
        if status == "error":
            raise RuntimeError(f"Memory benchmark worker failed:\n{detail}")
        if status != "done":
            raise RuntimeError(f"Unexpected memory benchmark status: {status}")

        worker.join()
        if worker.exitcode != 0:
            raise RuntimeError(
                f"Memory benchmark worker exited with code {worker.exitcode}"
            )
        assert isinstance(detail, int)
        return detail
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join()
        control.close()


@pytest.mark.profiling
def test_screened_and_dense_values_agree(
    device_benchmark_case: BenchmarkCase,
    benchmark_spec: BenchmarkSpec,
    load_functional_cached: Callable[..., ExcFunctionalBase | str],
) -> None:
    case = device_benchmark_case
    screened = case.run()

    with force_ao_screening(False):
        if case.backend == "cpu":
            dense = case.run()
        else:
            cpu_functional = load_functional_cached(
                "skala-1.1", device=torch.device("cpu")
            )
            assert isinstance(cpu_functional, ExcFunctionalBase)
            dense = _make_benchmark_case(benchmark_spec, cpu_functional, "cpu").run()

    density_rtol = 2e-10 if case.backend == "cpu" else 1e-8
    energy_rtol = 5e-10 if case.backend == "cpu" else 1e-8
    density_close = np.allclose(dense[0], screened[0], rtol=density_rtol, atol=1e-11)
    energy_close = np.isclose(dense[1], screened[1], rtol=energy_rtol, atol=1e-10)
    dense_vxc = (
        dense[2]
        if isinstance(dense[2], np.ndarray)
        else from_dlpack(cast(Any, dense[2])).cpu().numpy()
    )
    screened_vxc = (
        screened[2]
        if isinstance(screened[2], np.ndarray)
        else from_dlpack(cast(Any, screened[2])).cpu().numpy()
    )
    vxc_difference = dense_vxc - screened_vxc
    vxc_max_abs_difference = np.max(np.abs(vxc_difference))
    vxc_relative_l2_difference = np.linalg.norm(vxc_difference) / np.linalg.norm(
        dense_vxc
    )
    vxc_max_atol = 5e-8 if case.backend == "cpu" else 2e-7
    vxc_relative_rtol = 1e-8 if case.backend == "cpu" else 1e-7
    assert (
        density_close
        and energy_close
        and vxc_max_abs_difference < vxc_max_atol
        and vxc_relative_l2_difference < vxc_relative_rtol
    ), (
        f"N: dense={dense[0]:.16g}, screened={screened[0]:.16g}, "
        f"abs_diff={abs(dense[0] - screened[0]):.3e}; "
        f"E_xc: dense={dense[1]:.16g}, screened={screened[1]:.16g}, "
        f"abs_diff={abs(dense[1] - screened[1]):.3e}; "
        f"V_xc: max_abs_diff={vxc_max_abs_difference:.3e}, "
        f"relative_l2_diff={vxc_relative_l2_difference:.3e}"
    )


@pytest.mark.benchmark(group="def2-qzvpp")
def test_with_ao_screening(
    benchmark: BenchmarkFixture, screened_case: BenchmarkCase
) -> None:
    _benchmark_device_xc(benchmark, screened_case)


@pytest.mark.benchmark(group="def2-qzvpp")
def test_without_ao_screening_by_patching_decision(
    benchmark: BenchmarkFixture, dense_case: BenchmarkCase
) -> None:
    _benchmark_device_xc(benchmark, dense_case)


@pytest.mark.benchmark(group="device-def2-qzvpp-screened")
def test_screened_runtime_by_device(
    benchmark: BenchmarkFixture,
    device_benchmark_case: BenchmarkCase,
) -> None:
    _benchmark_device_xc(benchmark, device_benchmark_case)


@pytest.mark.profiling
@pytest.mark.parametrize("spec", BENCHMARK_SPECS)
@pytest.mark.parametrize(
    "backend",
    ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)],
)
def test_screened_and_dense_peak_memory(
    spec: BenchmarkSpec,
    backend: str,
    record_property: Callable[[str, object], None],
) -> None:
    if backend == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available")
        pytest.importorskip("cupy")
        pytest.importorskip("gpu4pyscf")

    screened_peak_bytes = _measure_peak_memory(spec, screened=True, backend=backend)
    dense_peak_bytes = _measure_peak_memory(spec, screened=False, backend=backend)
    mib = 1024**2
    screened_peak_mib = screened_peak_bytes / mib
    dense_peak_mib = dense_peak_bytes / mib
    peak_ratio = screened_peak_bytes / dense_peak_bytes

    record_property("backend", backend)
    record_property("screened_peak_allocations_mib", screened_peak_mib)
    record_property("dense_peak_allocations_mib", dense_peak_mib)
    record_property("screened_to_dense_peak_ratio", peak_ratio)
    print(
        f"\n{spec.name} {backend} peak allocations: "
        f"screened={screened_peak_mib:.1f} MiB, dense={dense_peak_mib:.1f} MiB, "
        f"ratio={peak_ratio:.3f}"
    )


@pytest.mark.profiling
def test_profile_with_ao_screening(
    device_benchmark_case: BenchmarkCase,
) -> None:
    device_benchmark_case.run()


@pytest.mark.profiling
def test_profile_without_ao_screening_by_patching_decision(
    device_benchmark_case: BenchmarkCase,
) -> None:
    with force_ao_screening(False):
        device_benchmark_case.run()
