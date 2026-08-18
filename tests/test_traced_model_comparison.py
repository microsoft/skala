# SPDX-License-Identifier: MIT

"""Compare the current Skala model trace with the published Skala 1.1 trace."""

import argparse
import gc
import io
import multiprocessing as mp
import os
import statistics
import tempfile
import time
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Literal, cast

import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture

from skala.features import Feature, FeatureMap
from skala.functional import TracedFunctional, load_functional
from skala.functional.model import SkalaFunctional

NUM_ATOMS = 10
GRID_POINTS_PER_ATOM = 500
POINTS_PER_CHUNK = NUM_ATOMS * GRID_POINTS_PER_ATOM
NUM_CHUNKS = 40
TOTAL_GRID_POINTS = NUM_CHUNKS * POINTS_PER_CHUNK
THREAD_COUNT = 4

# Strict forward-value tolerances on both devices and backward tolerances on CPU.
DENSITY_RTOL = 1e-7
DENSITY_ATOL = 1e-8

# CUDA einsum backward uses nondeterministic cuBLAS reductions. Repeating the
# published trace against itself produced differences up to ~6.3e-6 relative
# and ~8.4e-7 absolute, so CUDA gradient comparisons use the next round bounds.
CUDA_GRADIENT_RTOL = 1e-5
CUDA_GRADIENT_ATOL = 1e-6
RUNTIME_WARMUP_ROUNDS = 4
RUNTIME_ROUNDS = 5
MAX_RUNTIME_RATIO = 1.05
MAX_MEMORY_RATIO = 1.01
MEMORY_WORKER_TIMEOUT_SECONDS = 240

pytestmark = pytest.mark.model_benchmark

Workload = Literal["inference", "backward"]
WORKLOADS: tuple[Workload, ...] = ("inference", "backward")
Implementation = Literal["published", "local"]
IMPLEMENTATIONS: tuple[Implementation, ...] = ("published", "local")
RuntimeResults = dict[tuple[str, Workload], dict[Implementation, float]]
VXC_FEATURES = (
    Feature.DENSITY,
    Feature.GRAD,
    Feature.KIN,
)
NUCLEAR_GRADIENT_FEATURES = VXC_FEATURES + (
    Feature.GRID_COORDS,
    Feature.GRID_WEIGHTS,
    Feature.COARSE_0_ATOMIC_COORDS,
)


@dataclass(frozen=True)
class TracedModelCase:
    """Published and locally traced models with identical inputs."""

    device: torch.device
    published: torch.jit.ScriptModule
    local: torch.jit.ScriptModule
    chunks: tuple[FeatureMap, ...]


@dataclass
class DifferenceStats:
    """Streaming local-versus-published difference statistics."""

    max_abs: float = 0.0
    max_rel: float = 0.0
    outside_tolerance: int = 0
    values: int = 0

    def update(
        self,
        local: torch.Tensor,
        published: torch.Tensor,
        *,
        rtol: float,
        atol: float,
    ) -> None:
        difference = (local - published).abs()
        relative = torch.where(
            published != 0,
            difference / published.abs(),
            torch.where(difference == 0, 0.0, torch.inf),
        )
        self.max_abs = max(self.max_abs, float(difference.max().item()))
        self.max_rel = max(self.max_rel, float(relative.max().item()))
        self.outside_tolerance += int(
            (~torch.isclose(local, published, rtol=rtol, atol=atol)).sum().item()
        )
        self.values += local.numel()

    @property
    def passed(self) -> bool:
        return self.outside_tolerance == 0


@dataclass(frozen=True)
class ForwardAccuracyReport:
    """Forward density and integrated-energy differences."""

    density: DifferenceStats
    energy: DifferenceStats

    @property
    def passed(self) -> bool:
        return self.density.passed and self.energy.passed


@dataclass(frozen=True)
class GradientAccuracyReport:
    """Vxc and nuclear-gradient model-derivative differences."""

    gradients: dict[Feature, DifferenceStats]

    @property
    def passed(self) -> bool:
        return all(stats.passed for stats in self.gradients.values())


@dataclass(frozen=True)
class RatioReport:
    """Local and published measurements with their regression ratio."""

    local: float
    published: float
    ratio: float
    limit: float

    @property
    def passed(self) -> bool:
        return self.ratio <= self.limit


@pytest.fixture(scope="module", autouse=True)
def fixed_cpu_threads() -> Iterator[None]:
    """Pin CPU model execution to a reproducible OpenMP thread count."""
    previous_threads = torch.get_num_threads()
    previous_omp_threads = os.environ.get("OMP_NUM_THREADS")
    previous_mkl_threads = os.environ.get("MKL_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = str(THREAD_COUNT)
    os.environ["MKL_NUM_THREADS"] = str(THREAD_COUNT)
    torch.set_num_threads(THREAD_COUNT)
    try:
        yield
    finally:
        torch.set_num_threads(previous_threads)
        if previous_omp_threads is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = previous_omp_threads
        if previous_mkl_threads is None:
            os.environ.pop("MKL_NUM_THREADS", None)
        else:
            os.environ["MKL_NUM_THREADS"] = previous_mkl_threads


def _make_features(
    case_index: int,
    device: torch.device,
    *,
    gradient_features: tuple[Feature, ...] = (),
) -> FeatureMap:
    """Create one structured analytic tensor-only model workload."""
    # These values are not molecular reference data and are not intended to satisfy a
    # self-consistent density calculation. They are analytic, physically shaped inputs chosen to
    # exercise reproducible model regimes without letting random sampling decide which regimes
    # appear in a benchmark run. Coordinates are expressed on the model's Bohr scale.
    #
    # The ten coarse centers follow a non-planar, helix-like path. The 1.7 and 1.4 Bohr transverse
    # amplitudes and 0.32 Bohr axial spacing keep the centers in a compact but non-symmetric region;
    # the smaller sine term prevents a regular lattice. Each case advances a 0.37-radian phase.
    # That phase and the different angular frequencies below do not repeat over the 42
    # trace/check/measurement cases, so chunks differ smoothly without using an RNG.
    #
    # Each center receives 500 atom-major points. Polar cosines are midpoint samples of [-1, 1],
    # while 2.399963229728653 is the golden angle, pi * (3 - sqrt(5)); together they give a simple
    # deterministic spherical covering without repeated azimuthal spokes or points at the poles.
    # Radii span approximately 0.08 to 3.0 Bohr. The power 1.35 places more samples near a center,
    # and the +/-10% atom/case modulation avoids giving every center an identical radial shell.
    #
    # Spin densities use exp(-2.5 r), providing order-one values near a center and tails near the
    # 1e-4 floor. The 1.6/1.2 amplitudes and small directional modulations exercise spin asymmetry
    # while keeping both channels positive. Gradients contain the radial derivative of that
    # exponential plus a small signed tangential component, so all Cartesian signs occur and the
    # gradient is not artificially parallel to the radius everywhere.
    #
    # The kinetic density is positive by construction: 0.01 + 0.35 rho supplies a smooth baseline,
    # and |grad rho|^2 / (8 rho) is the spin-channel von Weizsaecker form. The clamp is only a
    # defensive denominator bound below the explicit density floor. Atomic weights scale as 1/500,
    # vary quadratically with radius, and receive +/-10% center modulation. Grid weights apply an
    # additional +/-20% directional partition factor. Both remain strictly positive and cover
    # nonuniform quadrature behavior. None of these constants is a golden expected output: every
    # assertion still compares the current local trace directly with the published trace.
    atom_index = torch.arange(NUM_ATOMS, dtype=torch.float64)
    point_index = torch.arange(GRID_POINTS_PER_ATOM, dtype=torch.float64)
    phase = case_index * 0.37

    coarse_coords = torch.stack(
        (
            1.7 * torch.cos(0.71 * atom_index + 0.13 * phase),
            1.4 * torch.sin(0.93 * atom_index - 0.11 * phase),
            0.32 * (atom_index - (NUM_ATOMS - 1) / 2)
            + 0.25 * torch.sin(0.47 * atom_index + phase),
        ),
        dim=-1,
    )

    unit_interval = (point_index + 0.5) / GRID_POINTS_PER_ATOM
    cos_polar = 1 - 2 * unit_interval
    sin_polar = torch.sqrt(1 - cos_polar.square())
    azimuth = (
        2.399963229728653 * point_index[None, :] + 0.41 * atom_index[:, None] + phase
    )
    directions = torch.stack(
        (
            sin_polar[None, :] * torch.cos(azimuth),
            sin_polar[None, :] * torch.sin(azimuth),
            cos_polar.expand(NUM_ATOMS, -1),
        ),
        dim=-1,
    )
    radii = (0.08 + 2.92 * unit_interval.pow(1.35))[None, :] * (
        0.9 + 0.1 * torch.sin(0.53 * atom_index[:, None] + phase)
    )
    grid_coords = (coarse_coords[:, None, :] + radii[..., None] * directions).reshape(
        -1, 3
    )

    radial_profile = torch.exp(-2.5 * radii)
    density_up = 1e-4 + 1.6 * radial_profile * (1 + 0.12 * directions[..., 0])
    density_down = 1e-4 + 1.2 * radial_profile * (1 - 0.10 * directions[..., 1])
    density = torch.stack((density_up.reshape(-1), density_down.reshape(-1)))

    tangential = torch.stack(
        (directions[..., 1], -directions[..., 0], torch.zeros_like(radii)), dim=-1
    )
    gradient_up = (
        -2.5 * (density_up - 1e-4)[..., None] * directions
        + 0.08 * radial_profile[..., None] * tangential
    )
    gradient_down = (
        -2.5 * (density_down - 1e-4)[..., None] * directions
        - 0.06 * radial_profile[..., None] * tangential
    )
    gradient = torch.stack(
        (
            gradient_up.reshape(-1, 3).T,
            gradient_down.reshape(-1, 3).T,
        )
    )
    kinetic = (
        0.01
        + 0.35 * density
        + gradient.square().sum(dim=1) / (8 * density.clamp_min(1e-6))
    )

    atomic_grid_weights = (
        (0.15 + 1.85 * unit_interval.square())[None, :]
        * (0.9 + 0.1 * torch.cos(0.43 * atom_index[:, None] - phase))
        / GRID_POINTS_PER_ATOM
    )
    grid_weights = atomic_grid_weights * (1 + 0.2 * directions[..., 2])

    features: FeatureMap = {
        Feature.DENSITY: density,
        Feature.GRAD: gradient,
        Feature.KIN: kinetic,
        Feature.GRID_COORDS: grid_coords,
        Feature.GRID_WEIGHTS: grid_weights.reshape(-1),
        Feature.ATOMIC_GRID_WEIGHTS: atomic_grid_weights.reshape(-1),
        Feature.ATOMIC_GRID_SIZES: torch.full(
            (NUM_ATOMS,), GRID_POINTS_PER_ATOM, dtype=torch.int64
        ),
        Feature.COARSE_0_ATOMIC_COORDS: coarse_coords,
        Feature.ATOMIC_GRID_SIZE_BOUND_SHAPE: torch.zeros(
            GRID_POINTS_PER_ATOM, 0, dtype=torch.int64
        ),
    }
    device_features = {feature: value.to(device) for feature, value in features.items()}
    for feature in gradient_features:
        device_features[feature].requires_grad_()
    return device_features


def _trace_current_model(
    published: TracedFunctional, device: torch.device
) -> torch.jit.ScriptModule:
    """Load published weights into the current implementation and trace it."""
    prefix = "_traced_model."
    state_dict = {}
    for key, value in published.state_dict().items():
        if not key.startswith(prefix):
            raise AssertionError(f"Unexpected published state-dict key: {key}")
        state_dict[key.removeprefix(prefix)] = value

    model = SkalaFunctional(lmax=3, num_non_local_layers=3, num_mid_layers=4)
    model.load_state_dict(state_dict, strict=True)
    model.to(device=device)
    model.eval()

    example = _make_features(case_index=0, device=device)
    check_example = _make_features(case_index=1, device=device)
    with torch.no_grad():
        trace_module = cast(
            Callable[..., torch.jit.ScriptModule], torch.jit.trace_module
        )
        traced = trace_module(
            model,
            {"get_exc_density": (example,)},
            check_inputs=[{"get_exc_density": (check_example,)}],
            strict=True,
        )

    archive = io.BytesIO()
    save_module = cast(Callable[..., None], torch.jit.save)
    load_module = cast(Callable[..., torch.jit.ScriptModule], torch.jit.load)
    save_module(traced, archive)
    archive.seek(0)
    return load_module(archive, map_location=device)


def _build_traced_model_case(device: torch.device) -> TracedModelCase:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    published = load_functional("skala-1.1", device=device)
    if not isinstance(published, TracedFunctional):
        raise TypeError("Expected the published Skala 1.1 model to be traced")
    local = _trace_current_model(published, device)
    chunks = tuple(
        _make_features(
            case_index=chunk_index + 2,
            device=device,
            gradient_features=VXC_FEATURES,
        )
        for chunk_index in range(NUM_CHUNKS)
    )

    with torch.no_grad():
        published._traced_model.get_exc_density(chunks[0])
        local.get_exc_density(chunks[0])
    warmup_features = _with_gradient_features(chunks[0], NUCLEAR_GRADIENT_FEATURES)
    _evaluate_with_gradients(
        published._traced_model,
        warmup_features,
        NUCLEAR_GRADIENT_FEATURES,
    )
    _evaluate_with_gradients(
        local,
        warmup_features,
        NUCLEAR_GRADIENT_FEATURES,
    )
    _synchronize(device)

    return TracedModelCase(
        device=device,
        published=published._traced_model,
        local=local,
        chunks=chunks,
    )


@pytest.fixture(
    scope="module",
    params=[
        pytest.param("cpu", id="cpu"),
        pytest.param("cuda", id="cuda", marks=pytest.mark.gpu),
    ],
)
def traced_model_case(request: pytest.FixtureRequest) -> Iterator[TracedModelCase]:
    """Build both traces and all measured inputs on one target device."""
    device = torch.device(str(request.param))
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    case = _build_traced_model_case(device)
    yield case
    del case
    if device.type == "cuda":
        torch.cuda.empty_cache()


def test_traced_model_accuracy(traced_model_case: TracedModelCase) -> None:
    """Compare every density value and the accumulated energy over 200k points."""
    report = _measure_forward_accuracy(traced_model_case)
    assert report.density.values == TOTAL_GRID_POINTS
    _assert_difference("density", report.density)
    _assert_difference("energy", report.energy)


def _gradient_inputs(
    features: FeatureMap, gradient_features: tuple[Feature, ...]
) -> tuple[torch.Tensor, ...]:
    return tuple(features[feature] for feature in gradient_features)


def _with_gradient_features(
    features: FeatureMap, gradient_features: tuple[Feature, ...]
) -> FeatureMap:
    gradient_feature_set = set(gradient_features)
    result = {feature: value.detach() for feature, value in features.items()}
    for feature in gradient_feature_set:
        result[feature].requires_grad_()
    return result


def _evaluate_with_gradients(
    model: Any,
    features: FeatureMap,
    gradient_features: tuple[Feature, ...] = VXC_FEATURES,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    density = model.get_exc_density(features)
    energy = (density.double() * features[Feature.GRID_WEIGHTS]).sum()
    gradients = torch.autograd.grad(
        energy, _gradient_inputs(features, gradient_features)
    )
    return (
        density.detach(),
        energy.detach(),
        tuple(gradient.detach() for gradient in gradients),
    )


def test_traced_model_backward_accuracy(traced_model_case: TracedModelCase) -> None:
    """Compare Vxc and nuclear-gradient model derivatives over 200k points."""
    report = _measure_gradient_accuracy(traced_model_case)
    for feature, stats in report.gradients.items():
        _assert_difference(f"dE/d{feature.value}", stats)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_workload(
    model: Any, chunks: tuple[FeatureMap, ...], workload: Workload
) -> None:
    if workload == "inference":
        with torch.no_grad():
            for features in chunks:
                model.get_exc_density(features)
    elif workload == "backward":
        for features in chunks:
            _evaluate_with_gradients(model, features)
    else:
        raise ValueError(f"Unknown traced-model workload: {workload}")


@pytest.fixture(scope="module")
def runtime_results() -> Iterator[RuntimeResults]:
    """Collect pytest-benchmark medians and gate every completed model pair."""
    results: RuntimeResults = {}
    yield results

    failures = []
    for (device_name, workload), medians in results.items():
        if set(medians) != set(IMPLEMENTATIONS):
            continue
        report = _ratio_report(
            medians["local"], medians["published"], MAX_RUNTIME_RATIO
        )
        if not report.passed:
            failures.append(
                f"{device_name} {workload}: local={report.local:.6f}s, "
                f"published={report.published:.6f}s, ratio={report.ratio:.4f}, "
                f"limit={report.limit:.2f}"
            )
    assert not failures, "Runtime regressions:\n" + "\n".join(failures)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("workload", WORKLOADS)
def test_traced_model_runtime(
    traced_model_case: TracedModelCase,
    benchmark: BenchmarkFixture,
    record_property: Any,
    runtime_results: RuntimeResults,
    workload: Workload,
    implementation: Implementation,
) -> None:
    """Record symmetric pytest-benchmark medians for both traces."""
    model = getattr(traced_model_case, implementation)
    pedantic = cast(Callable[..., object], benchmark.pedantic)

    def setup() -> tuple[tuple[()], dict[str, object]]:
        _synchronize(traced_model_case.device)
        return (), {}

    def run_model() -> None:
        _run_workload(model, traced_model_case.chunks, workload)
        _synchronize(traced_model_case.device)

    pedantic(
        run_model,
        setup=setup,
        warmup_rounds=RUNTIME_WARMUP_ROUNDS,
        rounds=RUNTIME_ROUNDS,
        iterations=1,
    )

    assert benchmark.stats is not None
    median = cast(float, benchmark.stats.stats.median)
    key = (traced_model_case.device.type, workload)
    runtime_results.setdefault(key, {})[implementation] = median

    record_property("device", traced_model_case.device.type)
    record_property("workload", workload)
    record_property("implementation", implementation)
    record_property("median_seconds", median)


def _prepare_memory_workload(
    implementation: str, device: torch.device, workload: Workload
) -> tuple[Any, tuple[FeatureMap, ...]]:
    published = load_functional("skala-1.1", device=device)
    assert isinstance(published, TracedFunctional)

    if implementation == "local":
        model = _trace_current_model(published, device)
    elif implementation == "published":
        model = published._traced_model
    else:
        raise ValueError(f"Unknown traced-model implementation: {implementation}")

    del published
    chunks = tuple(
        _make_features(
            case_index=chunk_index + 2,
            device=device,
            gradient_features=VXC_FEATURES if workload == "backward" else (),
        )
        for chunk_index in range(NUM_CHUNKS)
    )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model, chunks


def _measure_cpu_peak(
    model: Any, chunks: tuple[FeatureMap, ...], workload: Workload
) -> int:
    import memray

    _run_workload(model, chunks, workload)
    with tempfile.TemporaryDirectory() as tmpdir:
        profile_path = Path(tmpdir) / "allocations.bin"
        with memray.Tracker(profile_path):
            _run_workload(model, chunks, workload)
        return memray.FileReader(profile_path).metadata.peak_memory


def _measure_cuda_peak(
    model: Any,
    chunks: tuple[FeatureMap, ...],
    device: torch.device,
    workload: Workload,
) -> int:
    _run_workload(model, chunks, workload)
    _synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()
    baseline_bytes = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    _run_workload(model, chunks, workload)
    _synchronize(device)
    return torch.cuda.max_memory_allocated(device) - baseline_bytes


def _memory_worker(
    implementation: str,
    device_name: str,
    workload: Workload,
    control: Connection,
) -> None:
    """Measure one trace in an isolated process and return its allocation peak."""
    try:
        os.environ["OMP_NUM_THREADS"] = str(THREAD_COUNT)
        os.environ["MKL_NUM_THREADS"] = str(THREAD_COUNT)
        torch.set_num_threads(THREAD_COUNT)
        device = torch.device(device_name)
        model, chunks = _prepare_memory_workload(implementation, device, workload)
        if device.type == "cpu":
            peak_bytes = _measure_cpu_peak(model, chunks, workload)
        elif device.type == "cuda":
            peak_bytes = _measure_cuda_peak(model, chunks, device, workload)
        else:
            raise ValueError(f"Unknown memory benchmark device: {device_name}")
        control.send(("done", peak_bytes))
    except Exception:  # noqa: BLE001 - forward worker failures to the parent
        control.send(("error", traceback.format_exc()))
    finally:
        control.close()


def _measure_peak_memory(
    implementation: str, device_name: str, workload: Workload
) -> int:
    """Return peak inference allocations from one isolated worker."""
    context = mp.get_context("spawn")
    control, worker_control = context.Pipe()
    worker = context.Process(
        target=_memory_worker,
        args=(implementation, device_name, workload, worker_control),
    )
    worker.start()
    worker_control.close()

    try:
        if not control.poll(MEMORY_WORKER_TIMEOUT_SECONDS):
            raise TimeoutError("Traced-model memory worker timed out")
        status, detail = cast(tuple[str, int | str], control.recv())
        if status == "error":
            raise RuntimeError(f"Traced-model memory worker failed:\n{detail}")
        if status != "done":
            raise RuntimeError(f"Unexpected memory worker status: {status}")

        worker.join()
        if worker.exitcode != 0:
            raise RuntimeError(
                f"Traced-model memory worker exited with code {worker.exitcode}"
            )
        assert isinstance(detail, int)
        return detail
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join()
        control.close()


@pytest.mark.parametrize(
    "device_name",
    [
        pytest.param("cpu", id="cpu"),
        pytest.param("cuda", id="cuda", marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize("workload", WORKLOADS)
def test_traced_model_peak_memory(
    device_name: str,
    record_property: Callable[[str, object], None],
    workload: Workload,
) -> None:
    """Require forward and backward peak allocations within 1% of published."""
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    report = _measure_memory(device_name, workload)
    mib = 1024**2
    record_property("device", device_name)
    record_property("workload", workload)
    record_property("local_peak_allocations_mib", report.local / mib)
    record_property("published_peak_allocations_mib", report.published / mib)
    record_property("local_to_published_peak_ratio", report.ratio)

    assert report.passed, (
        f"Local trace peak allocations {report.local / mib:.3f} MiB exceed "
        f"{report.limit:.2f}x the published trace peak "
        f"{report.published / mib:.3f} MiB (ratio={report.ratio:.4f})"
    )


def _gradient_tolerances(device: torch.device) -> tuple[float, float]:
    if device.type == "cuda":
        return CUDA_GRADIENT_RTOL, CUDA_GRADIENT_ATOL
    return DENSITY_RTOL, DENSITY_ATOL


def _measure_forward_accuracy(case: TracedModelCase) -> ForwardAccuracyReport:
    density_stats = DifferenceStats()
    local_energy = torch.zeros((), dtype=torch.float64, device=case.device)
    published_energy = torch.zeros((), dtype=torch.float64, device=case.device)

    for features in case.chunks:
        with torch.no_grad():
            local_density = case.local.get_exc_density(features)
            published_density = case.published.get_exc_density(features)
        density_stats.update(
            local_density,
            published_density,
            rtol=DENSITY_RTOL,
            atol=DENSITY_ATOL,
        )
        grid_weights = features[Feature.GRID_WEIGHTS]
        local_energy += (local_density.double() * grid_weights).sum()
        published_energy += (published_density.double() * grid_weights).sum()

    energy_stats = DifferenceStats()
    energy_stats.update(
        local_energy,
        published_energy,
        rtol=DENSITY_RTOL,
        atol=DENSITY_ATOL,
    )
    return ForwardAccuracyReport(density_stats, energy_stats)


def _measure_gradient_accuracy(case: TracedModelCase) -> GradientAccuracyReport:
    gradient_stats = {
        feature: DifferenceStats() for feature in NUCLEAR_GRADIENT_FEATURES
    }
    gradient_rtol, gradient_atol = _gradient_tolerances(case.device)

    for base_features in case.chunks:
        features = _with_gradient_features(base_features, NUCLEAR_GRADIENT_FEATURES)
        _, _, local_gradients = _evaluate_with_gradients(
            case.local, features, NUCLEAR_GRADIENT_FEATURES
        )
        _, _, published_gradients = _evaluate_with_gradients(
            case.published, features, NUCLEAR_GRADIENT_FEATURES
        )
        for feature, local_gradient, published_gradient in zip(
            NUCLEAR_GRADIENT_FEATURES,
            local_gradients,
            published_gradients,
            strict=True,
        ):
            gradient_stats[feature].update(
                local_gradient,
                published_gradient,
                rtol=gradient_rtol,
                atol=gradient_atol,
            )

    return GradientAccuracyReport(gradient_stats)


def _time_workload(
    model: Any,
    chunks: tuple[FeatureMap, ...],
    device: torch.device,
    workload: Workload,
) -> float:
    _synchronize(device)
    start = time.perf_counter()
    _run_workload(model, chunks, workload)
    _synchronize(device)
    return time.perf_counter() - start


def _ratio_report(local: float, published: float, limit: float) -> RatioReport:
    if published <= 0:
        raise ValueError("Published measurement must be positive")
    return RatioReport(local, published, local / published, limit)


def _measure_runtime(case: TracedModelCase, workload: Workload) -> RatioReport:
    local_durations: list[float] = []
    published_durations: list[float] = []
    total_rounds = RUNTIME_WARMUP_ROUNDS + RUNTIME_ROUNDS

    for round_index in range(total_rounds):
        published_duration = _time_workload(
            case.published, case.chunks, case.device, workload
        )
        local_duration = _time_workload(case.local, case.chunks, case.device, workload)

        if round_index >= RUNTIME_WARMUP_ROUNDS:
            published_durations.append(published_duration)
            local_durations.append(local_duration)

    local_median = statistics.median(local_durations)
    published_median = statistics.median(published_durations)
    return _ratio_report(local_median, published_median, MAX_RUNTIME_RATIO)


def _measure_memory(device_name: str, workload: Workload) -> RatioReport:
    published_bytes = _measure_peak_memory("published", device_name, workload)
    local_bytes = _measure_peak_memory("local", device_name, workload)
    return _ratio_report(float(local_bytes), float(published_bytes), MAX_MEMORY_RATIO)


def _assert_difference(name: str, stats: DifferenceStats) -> None:
    assert stats.passed, (
        f"{name}: max_abs={stats.max_abs:.3e}, max_rel={stats.max_rel:.3e}, "
        f"outside_tolerance={stats.outside_tolerance}/{stats.values}"
    )


def _status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _print_difference(name: str, stats: DifferenceStats) -> None:
    print(
        f"  {name:<24} max_abs={stats.max_abs:.3e}  "
        f"max_rel={stats.max_rel:.3e}  "
        f"outside={stats.outside_tolerance}/{stats.values}  "
        f"{_status(stats.passed)}"
    )


def _device_label(device: torch.device) -> str:
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else device.index
        return f"CUDA {index} ({torch.cuda.get_device_name(index)})"
    return "CPU"


def _run_script_device(device_name: str) -> bool:
    device = torch.device(device_name)
    print("\nSkala local vs published trace")
    print(f"Execution device: {_device_label(device)}")
    print(f"CPU threads: {torch.get_num_threads()}")
    print(f"Workload: {TOTAL_GRID_POINTS:,} points")
    case = _build_traced_model_case(device)

    print("\nAccuracy differences")
    forward_accuracy = _measure_forward_accuracy(case)
    gradient_accuracy = _measure_gradient_accuracy(case)
    _print_difference("density", forward_accuracy.density)
    _print_difference("energy", forward_accuracy.energy)
    for feature in NUCLEAR_GRADIENT_FEATURES:
        _print_difference(f"dE/d{feature.value}", gradient_accuracy.gradients[feature])

    print("\nRuntime medians (seconds)")
    runtime_reports: list[RatioReport] = []
    for workload in WORKLOADS:
        report = _measure_runtime(case, workload)
        runtime_reports.append(report)
        print(
            f"  {workload:<12} published={report.published:.6f}  "
            f"local={report.local:.6f}  ratio={report.ratio:.4f}  "
            f"limit={report.limit:.2f}  {_status(report.passed)}"
        )

    del case
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\nPeak allocations (MiB)")
    memory_reports: list[RatioReport] = []
    mib = 1024**2
    for workload in WORKLOADS:
        report = _measure_memory(device_name, workload)
        memory_reports.append(report)
        print(
            f"  {workload:<12} published={report.published / mib:.3f}  "
            f"local={report.local / mib:.3f}  ratio={report.ratio:.4f}  "
            f"limit={report.limit:.2f}  {_status(report.passed)}"
        )

    passed = (
        forward_accuracy.passed
        and gradient_accuracy.passed
        and all(report.passed for report in runtime_reports)
        and all(report.passed for report in memory_reports)
    )
    print(f"\nOverall: {_status(passed)}")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare the current Skala trace with the published Skala 1.1 model."
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "all"),
        default="cpu",
        help="device comparison to run (default: cpu)",
    )
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(THREAD_COUNT)
    os.environ["MKL_NUM_THREADS"] = str(THREAD_COUNT)
    torch.set_num_threads(THREAD_COUNT)

    device_names = ("cpu", "cuda") if args.device == "all" else (args.device,)
    if "cuda" in device_names and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    results = [_run_script_device(name) for name in device_names]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
