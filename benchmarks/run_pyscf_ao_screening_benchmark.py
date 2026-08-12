"""Run isolated Skala PySCF and GPU4PySCF AO-screening benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

FULL_CARBON_COUNTS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
EXPECTED_AO_COUNTS = (
    294,
    411,
    528,
    645,
    762,
    879,
    996,
    1113,
    1230,
    1347,
    1464,
)
MODES = ("cpu", "cpu_dense", "gpu")
MEASUREMENTS = ("runtime", "memory")
TERMINAL_STATUSES = {"ok", "timeout", "oom", "error", "skipped_after_resource_failure"}
WORKER_RESULT_PREFIX = "SKALA_BENCHMARK_RESULT="
THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

Vector = tuple[float, float, float]
Atom = tuple[str, float, float, float]

CARBON_CARBON_BOND_ANGSTROM = 1.54
CARBON_HYDROGEN_BOND_ANGSTROM = 1.09
CARBON_BOND_ANGLE_DEGREES = 112.0
COORDINATE_PRECISION = 12
GEOMETRY_PARAMETERS = {
    "version": "zigzag-alkane-v1",
    "carbon_carbon_bond_angstrom": CARBON_CARBON_BOND_ANGSTROM,
    "carbon_hydrogen_bond_angstrom": CARBON_HYDROGEN_BOND_ANGSTROM,
    "carbon_bond_angle_degrees": CARBON_BOND_ANGLE_DEGREES,
    "hydrogen_dot_product": -1.0 / 3.0,
    "coordinate_precision": COORDINATE_PRECISION,
}
EXPECTED_AOS_BY_CARBON: dict[int, int] = dict(
    zip(FULL_CARBON_COUNTS, EXPECTED_AO_COUNTS, strict=True)
)


def find_repository_root(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "LICENSE.txt"
        ).is_file():
            return candidate
    raise FileNotFoundError(f"Could not find the Skala repository above {start}")


RUNNER_ROOT = find_repository_root(Path(__file__).resolve())
DEFAULT_SOURCE_ROOT = RUNNER_ROOT / "skala"


@dataclass(frozen=True)
class BenchmarkConfig:
    source_root: Path
    results_dir: Path
    run_label: str
    functional: str = "skala-1.1"
    basis: str = "def2-qzvpp"
    grid_level: int = 1
    grid_alignment: int = 1
    max_memory_mb: int = 2000
    cpu_threads: int = 4
    runtime_repetitions: int = 3
    worker_timeout_seconds: int = 30 * 60
    smoke_run: bool = False

    @property
    def carbon_counts(self) -> tuple[int, ...]:
        return FULL_CARBON_COUNTS[:1] if self.smoke_run else FULL_CARBON_COUNTS

    @property
    def worker_thread_environment(self) -> dict[str, str]:
        thread_count = str(self.cpu_threads)
        return {name: thread_count for name in THREAD_ENVIRONMENT_VARIABLES}

    def as_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_root"] = str(self.source_root)
        data["results_dir"] = str(self.results_dir)
        data["carbon_counts"] = list(self.carbon_counts)
        data["full_carbon_counts"] = list(FULL_CARBON_COUNTS)
        data["expected_ao_counts"] = list(EXPECTED_AO_COUNTS)
        data["worker_thread_environment"] = self.worker_thread_environment
        return data


def vector_add(left: Vector, right: Vector) -> Vector:
    return tuple(a + b for a, b in zip(left, right, strict=True))  # type: ignore[return-value]


def vector_subtract(left: Vector, right: Vector) -> Vector:
    return tuple(a - b for a, b in zip(left, right, strict=True))  # type: ignore[return-value]


def vector_scale(scale: float, vector: Vector) -> Vector:
    return tuple(scale * value for value in vector)  # type: ignore[return-value]


def vector_dot(left: Vector, right: Vector) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def vector_cross(left: Vector, right: Vector) -> Vector:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def vector_normalize(vector: Vector) -> Vector:
    norm = math.sqrt(vector_dot(vector, vector))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero vector")
    return vector_scale(1.0 / norm, vector)


def carbon_backbone(carbon_count: int) -> tuple[Vector, ...]:
    if carbon_count < 2:
        raise ValueError("The benchmark requires at least two carbon atoms")
    half_turn = math.radians((180.0 - CARBON_BOND_ANGLE_DEGREES) / 2.0)
    positions: list[Vector] = [(0.0, 0.0, 0.0)]
    for bond_index in range(carbon_count - 1):
        angle = half_turn if bond_index % 2 == 0 else -half_turn
        direction = (math.cos(angle), math.sin(angle), 0.0)
        positions.append(
            vector_add(
                positions[-1], vector_scale(CARBON_CARBON_BOND_ANGSTROM, direction)
            )
        )

    center = tuple(
        sum(position[axis] for position in positions) / carbon_count
        for axis in range(3)
    )
    return tuple(vector_subtract(position, center) for position in positions)  # type: ignore[arg-type]


def terminal_hydrogen_directions(
    carbon: Vector, neighbor: Vector
) -> tuple[Vector, ...]:
    neighbor_direction = vector_normalize(vector_subtract(neighbor, carbon))
    perpendicular = (0.0, 0.0, 1.0)
    second_perpendicular = vector_normalize(
        vector_cross(neighbor_direction, perpendicular)
    )
    radial_scale = math.sqrt(8.0 / 9.0)
    directions = []
    for index in range(3):
        phase = 2.0 * math.pi * index / 3.0
        radial = vector_add(
            vector_scale(math.cos(phase), perpendicular),
            vector_scale(math.sin(phase), second_perpendicular),
        )
        directions.append(
            vector_add(
                vector_scale(-1.0 / 3.0, neighbor_direction),
                vector_scale(radial_scale, radial),
            )
        )
    return tuple(directions)


def internal_hydrogen_directions(
    carbon: Vector, previous_carbon: Vector, next_carbon: Vector
) -> tuple[Vector, Vector]:
    previous_direction = vector_normalize(vector_subtract(previous_carbon, carbon))
    next_direction = vector_normalize(vector_subtract(next_carbon, carbon))
    neighbor_dot = vector_dot(previous_direction, next_direction)
    in_plane_scale = (-1.0 / 3.0) / (1.0 + neighbor_dot)
    in_plane = vector_scale(
        in_plane_scale, vector_add(previous_direction, next_direction)
    )
    normal = vector_normalize(vector_cross(previous_direction, next_direction))
    normal_scale = math.sqrt(max(0.0, 1.0 - vector_dot(in_plane, in_plane)))
    return (
        vector_add(in_plane, vector_scale(normal_scale, normal)),
        vector_subtract(in_plane, vector_scale(normal_scale, normal)),
    )


def generate_alkane_atoms(carbon_count: int) -> tuple[Atom, ...]:
    carbons = carbon_backbone(carbon_count)
    atoms: list[Atom] = [("C", *position) for position in carbons]
    for index, carbon in enumerate(carbons):
        if index == 0:
            directions = terminal_hydrogen_directions(carbon, carbons[1])
        elif index == carbon_count - 1:
            directions = terminal_hydrogen_directions(carbon, carbons[-2])
        else:
            directions = internal_hydrogen_directions(
                carbon, carbons[index - 1], carbons[index + 1]
            )
        atoms.extend(
            (
                "H",
                *vector_add(
                    carbon, vector_scale(CARBON_HYDROGEN_BOND_ANGSTROM, direction)
                ),
            )
            for direction in directions
        )
    return tuple(atoms)


def atoms_to_pyscf(atoms: tuple[Atom, ...]) -> str:
    return "\n".join(
        f"{element} {x:.{COORDINATE_PRECISION}f} "
        f"{y:.{COORDINATE_PRECISION}f} {z:.{COORDINATE_PRECISION}f}"
        for element, x, y, z in atoms
    )


@dataclass(frozen=True)
class MoleculeSpec:
    carbon_count: int
    expected_aos: int
    formula: str
    atoms: tuple[Atom, ...]

    @property
    def atom_text(self) -> str:
        return atoms_to_pyscf(self.atoms)

    @property
    def coordinate_sha256(self) -> str:
        return hashlib.sha256(self.atom_text.encode()).hexdigest()

    def as_json(self) -> dict[str, Any]:
        return {
            "carbon_count": self.carbon_count,
            "expected_aos": self.expected_aos,
            "formula": self.formula,
            "atoms": [
                {"element": element, "xyz_angstrom": [x, y, z]}
                for element, x, y, z in self.atoms
            ],
            "coordinate_sha256": self.coordinate_sha256,
        }


def make_molecule_spec(carbon_count: int) -> MoleculeSpec:
    hydrogen_count = 2 * carbon_count + 2
    return MoleculeSpec(
        carbon_count=carbon_count,
        expected_aos=EXPECTED_AOS_BY_CARBON[carbon_count],
        formula=f"C{carbon_count}H{hydrogen_count}",
        atoms=generate_alkane_atoms(carbon_count),
    )


FULL_MOLECULE_LADDER = tuple(make_molecule_spec(count) for count in FULL_CARBON_COUNTS)


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def verify_skala_import(source_root: Path) -> str:
    import skala

    imported_path = Path(skala.__file__).resolve()
    expected_root = (source_root / "src").resolve()
    try:
        imported_path.relative_to(expected_root)
    except ValueError as error:
        raise RuntimeError(
            f"Imported Skala from {imported_path}, expected a module below {expected_root}"
        ) from error
    return str(imported_path)


def collect_environment(payload: dict[str, Any]) -> dict[str, Any]:
    import torch

    import pyscf

    source_root = Path(payload["source_root"]).resolve()
    imported_skala = verify_skala_import(source_root)
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    cupy_version = package_version("cupy-cuda12x") or package_version("cupy")
    torch_cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "source_root": str(source_root),
        "imported_skala": imported_skala,
        "packages": {
            "skala": package_version("skala"),
            "pyscf": pyscf.__version__,
            "gpu4pyscf": package_version("gpu4pyscf-cuda12x")
            or package_version("gpu4pyscf"),
            "torch": torch.__version__,
            "cupy": cupy_version,
            "memray": package_version("memray"),
        },
        "cuda": {
            "available": cuda_available,
            "torch_cuda_version": torch_cuda_version,
            "device_name": gpu_name,
            "device_count": torch.cuda.device_count() if cuda_available else 0,
        },
        "thread_environment": {
            name: os.environ.get(name) for name in THREAD_ENVIRONMENT_VARIABLES
        },
    }


def find_route_controller(numint: Any) -> tuple[Any, Any, Any]:
    candidates = (numint, getattr(numint, "integrator", None))
    control_symbols = {"_should_screen_aos", "_functional_supports_atom_chunking"}
    for candidate in candidates:
        if candidate is None:
            continue
        route_callable = inspect.unwrap(type(candidate).__call__)
        referenced_names = set(route_callable.__code__.co_names)
        if referenced_names & control_symbols:
            route_module = inspect.getmodule(route_callable)
            if route_module is None:
                raise RuntimeError(
                    f"Cannot identify the module defining {route_callable.__qualname__}"
                )
            return candidate, route_callable, route_module
    raise RuntimeError("Cannot find the Skala route-selection implementation")


def force_dense_route(numint: Any) -> AbstractContextManager[Any]:
    route_owner, route_callable, route_module = find_route_controller(numint)
    referenced_names = set(route_callable.__code__.co_names)
    if "_should_screen_aos" in referenced_names and hasattr(
        route_module, "_should_screen_aos"
    ):
        return patch.object(route_module, "_should_screen_aos", return_value=False)
    if "_functional_supports_atom_chunking" in referenced_names and hasattr(
        type(route_owner), "_functional_supports_atom_chunking"
    ):
        return patch.object(
            type(route_owner),
            "_functional_supports_atom_chunking",
            return_value=False,
        )
    raise RuntimeError(
        "Cannot force dense evaluation through "
        f"{route_module.__name__}.{route_callable.__qualname__}"
    )


def route_metadata(numint: Any, mol: Any, forced_dense: bool) -> dict[str, Any]:
    from pyscf.dft import numint as pyscf_numint

    route_owner, route_callable, route_module = find_route_controller(numint)
    routing_source = inspect.getsource(route_callable)
    routing_sha256 = hashlib.sha256(routing_source.encode()).hexdigest()
    referenced_names = set(route_callable.__code__.co_names)
    if "_should_screen_aos" in referenced_names and hasattr(
        route_module, "_should_screen_aos"
    ):
        route_decision_callable = route_module._should_screen_aos
        route_decision = bool(route_decision_callable(mol))
        supports_screened_evaluation = bool(
            route_owner.feature_spec.supports_spatial_decomposition
        )
        route_selector = "ao_threshold"
    elif "_functional_supports_atom_chunking" in referenced_names and hasattr(
        type(route_owner), "_functional_supports_atom_chunking"
    ):
        route_decision_callable = route_owner._functional_supports_atom_chunking
        route_decision = bool(route_decision_callable())
        supports_screened_evaluation = route_decision
        route_selector = "functional_capability"
    else:
        raise RuntimeError("Unrecognized Skala route-selection API")
    route_decision_source = inspect.getsource(route_decision_callable)
    if "_should_screen_aos" in routing_source and (
        "_global_screened_features" in routing_source
        or "_integrate_screened" in routing_source
    ):
        implementation = "threshold_gated_global_ao_screening"
    elif "chunked_features" in routing_source:
        implementation = "legacy_atom_chunking"
    else:
        implementation = "unclassified"

    switch_size = int(pyscf_numint.SWITCH_SIZE)
    if forced_dense or not supports_screened_evaluation:
        selected_route = "dense"
    elif implementation == "threshold_gated_global_ao_screening":
        selected_route = "global_ao_screening" if route_decision else "dense"
    elif implementation == "legacy_atom_chunking":
        selected_route = "atom_chunking"
    else:
        selected_route = "unknown"
    return {
        "request": "forced_dense" if forced_dense else "natural",
        "implementation": implementation,
        "implementation_sha256": routing_sha256,
        "implementation_target": (
            f"{route_module.__name__}.{route_callable.__qualname__}"
        ),
        "route_selector": route_selector,
        "route_decision": {
            "target": (
                f"{route_decision_callable.__module__}."
                f"{route_decision_callable.__qualname__}"
            ),
            "source": route_decision_source,
            "source_sha256": hashlib.sha256(route_decision_source.encode()).hexdigest(),
            "result": route_decision,
        },
        "functional_supports_screened_evaluation": supports_screened_evaluation,
        "pyscf_switch_size": switch_size,
        "selected_route": selected_route,
    }


def build_case(payload: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch

    from pyscf import dft, gto, lib

    source_root = Path(payload["source_root"]).resolve()
    imported_skala = verify_skala_import(source_root)
    thread_count = int(payload["cpu_threads"])
    lib.num_threads(thread_count)
    torch.set_num_threads(thread_count)

    molecule = payload["molecule"]
    mol = gto.M(
        atom=molecule["atom_text"],
        basis=payload["basis"],
        charge=0,
        spin=0,
        unit="Angstrom",
        cart=False,
        verbose=0,
    )
    initial_dm = dft.RKS(mol).get_init_guess()
    backend = payload["backend"]
    if backend == "cpu":
        from skala.pyscf import SkalaKS as CpuSkalaKS

        ks = CpuSkalaKS(mol, xc=payload["functional"], with_dftd3=False)
        dm = initial_dm

        def synchronize() -> None:
            return None

        to_numpy = np.asarray
    elif backend == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        import cupy
        from skala.gpu4pyscf import SkalaKS as GpuSkalaKS

        ks = GpuSkalaKS(mol, xc=payload["functional"], with_dftd3=False)
        dm = cupy.asarray(initial_dm)
        synchronize = torch.cuda.synchronize
        to_numpy = cupy.asnumpy
    else:
        raise ValueError(f"Unknown backend: {backend}")

    ks.grids.level = int(payload["grid_level"])
    ks.grids.alignment = int(payload["grid_alignment"])
    ks.grids.build(sort_grids=False)
    grid_weights = ks.grids.weights
    if grid_weights is None:
        raise RuntimeError("Grid construction did not produce weights")
    numint = ks._numint
    forced_dense = bool(payload["forced_dense"])
    route = route_metadata(numint, mol, forced_dense)
    system = {
        "formula": molecule["formula"],
        "carbon_count": int(molecule["carbon_count"]),
        "electron_count": int(mol.nelectron),
        "actual_aos": int(mol.nao_nr()),
        "grid_points": int(cast(Any, grid_weights).size),
        "coordinate_sha256": molecule["coordinate_sha256"],
        "imported_skala": imported_skala,
    }
    return {
        "mol": mol,
        "grids": ks.grids,
        "dm": dm,
        "numint": numint,
        "backend": backend,
        "synchronize": synchronize,
        "to_numpy": to_numpy,
        "route": route,
        "system": system,
    }


def fingerprint(result: tuple[Any, Any, Any], to_numpy: Any) -> dict[str, float]:
    import numpy as np

    electron_integral, xc_energy, vxc = result
    matrix = np.asarray(to_numpy(vxc), dtype=np.float64)
    return {
        "electron_integral": float(electron_integral),
        "xc_energy": float(xc_energy),
        "vxc_sum": float(matrix.sum()),
        "vxc_trace": float(np.trace(matrix)),
        "vxc_frobenius_norm": float(np.linalg.norm(matrix)),
        "vxc_max_abs": float(np.max(np.abs(matrix))),
    }


def run_measurement(payload: dict[str, Any]) -> dict[str, Any]:
    import torch

    case = build_case(payload)
    numint = case["numint"]
    dense_route_override: AbstractContextManager[Any]
    if not payload["forced_dense"]:
        dense_route_override = nullcontext()
    else:
        dense_route_override = force_dense_route(numint)

    def evaluate() -> tuple[Any, Any, Any]:
        return numint.nr_rks(
            case["mol"],
            case["grids"],
            None,
            case["dm"],
            max_memory=int(payload["max_memory_mb"]),
        )

    result: tuple[Any, Any, Any] | None = None
    with dense_route_override:
        measurement = payload["measurement"]
        if measurement == "runtime":
            case["synchronize"]()
            started = time.perf_counter()
            result = evaluate()
            case["synchronize"]()
            elapsed_seconds = time.perf_counter() - started
            measurement_data = {"runtime_seconds": elapsed_seconds}
        elif measurement == "memory" and case["backend"] == "cpu":
            import memray

            with tempfile.TemporaryDirectory() as temp_dir:
                profile_path = Path(temp_dir) / "allocations.bin"
                with memray.Tracker(profile_path):
                    result = evaluate()
                peak_bytes = int(memray.FileReader(profile_path).metadata.peak_memory)
            measurement_data = {"incremental_peak_bytes": peak_bytes}
        elif measurement == "memory" and case["backend"] == "gpu":
            case["synchronize"]()
            torch.cuda.empty_cache()
            case["synchronize"]()
            baseline_bytes = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            result = evaluate()
            case["synchronize"]()
            peak_bytes = max(0, torch.cuda.max_memory_allocated() - baseline_bytes)
            measurement_data = {
                "incremental_peak_bytes": int(peak_bytes),
                "allocator_baseline_bytes": int(baseline_bytes),
            }
        else:
            raise ValueError(f"Unknown measurement: {measurement}")

    assert result is not None
    return {
        "status": "ok",
        "measurement": payload["measurement"],
        "mode": payload["mode"],
        "route": case["route"],
        "system": case["system"],
        "fingerprint": fingerprint(result, case["to_numpy"]),
        **measurement_data,
    }


def classify_exception(error: Exception) -> str:
    message = f"{type(error).__name__}: {error}".lower()
    if (
        isinstance(error, MemoryError)
        or "out of memory" in message
        or "bad alloc" in message
    ):
        return "oom"
    return "error"


def emit_worker_record(record: dict[str, Any]) -> None:
    print(WORKER_RESULT_PREFIX + json.dumps(record, sort_keys=True), flush=True)


def worker_main() -> None:
    payload = json.load(sys.stdin)
    try:
        if payload["operation"] == "environment":
            emit_worker_record(
                {"status": "ok", "environment": collect_environment(payload)}
            )
        elif payload["operation"] == "measure":
            emit_worker_record(run_measurement(payload))
        else:
            raise ValueError(f"Unknown operation: {payload['operation']}")
    except Exception as error:  # noqa: BLE001 - serialize worker failures for the parent
        emit_worker_record(
            {
                "status": classify_exception(error),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc()[-12000:],
                "measurement": payload.get("measurement"),
                "mode": payload.get("mode"),
            }
        )


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def git_output(source_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def source_metadata(source_root: Path) -> dict[str, Any]:
    return {
        "root": str(source_root),
        "commit": git_output(source_root, "rev-parse", "HEAD"),
        "branch": git_output(source_root, "branch", "--show-current") or None,
        "dirty": bool(git_output(source_root, "status", "--porcelain")),
    }


def runner_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def worker_environment(config: BenchmarkConfig) -> dict[str, str]:
    environment = os.environ.copy()
    source_python_path = str(config.source_root / "src")
    existing_python_path = environment.get("PYTHONPATH")
    python_paths = [source_python_path, str(RUNNER_ROOT)]
    if existing_python_path:
        python_paths.append(existing_python_path)
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    environment.update(config.worker_thread_environment)
    return environment


def execute_worker(payload: dict[str, Any], config: BenchmarkConfig) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            input=json.dumps(payload),
            cwd=config.source_root,
            env=worker_environment(config),
            capture_output=True,
            text=True,
            timeout=config.worker_timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return {
            "status": "timeout",
            "measurement": payload.get("measurement"),
            "mode": payload.get("mode"),
            "error": f"Worker exceeded {config.worker_timeout_seconds} seconds",
            "stdout_tail": (error.stdout or "")[-4000:],
            "stderr_tail": (error.stderr or "")[-4000:],
        }

    marker_lines = [
        line.removeprefix(WORKER_RESULT_PREFIX)
        for line in completed.stdout.splitlines()
        if line.startswith(WORKER_RESULT_PREFIX)
    ]
    if marker_lines:
        record = json.loads(marker_lines[-1])
        if record["status"] != "ok":
            record["stderr_tail"] = completed.stderr[-4000:]
        return record

    combined_output = f"{completed.stdout}\n{completed.stderr}".lower()
    status = (
        "oom"
        if completed.returncode in {-9, 137} or "out of memory" in combined_output
        else "error"
    )
    return {
        "status": status,
        "measurement": payload.get("measurement"),
        "mode": payload.get("mode"),
        "error": f"Worker exited with code {completed.returncode} without a result record",
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def execute_measurement(
    payload: dict[str, Any], config: BenchmarkConfig
) -> dict[str, Any]:
    if payload["measurement"] != "runtime":
        return execute_worker(payload, config)

    runtime_samples: list[float] = []
    first_result: dict[str, Any] | None = None
    for sample_index in range(config.runtime_repetitions):
        result = execute_worker(payload, config)
        if result["status"] != "ok":
            result["runtime_samples_seconds"] = runtime_samples
            result["failed_runtime_sample_index"] = sample_index
            return result
        if first_result is None:
            first_result = result
        else:
            for key in ("route", "system"):
                if result.get(key) != first_result.get(key):
                    raise ValueError(
                        f"Runtime worker {key} changed between isolated samples"
                    )
        runtime_samples.append(float(result["runtime_seconds"]))

    assert first_result is not None
    first_result.pop("runtime_seconds")
    first_result["runtime_samples_seconds"] = runtime_samples
    return first_result


def atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary_path = Path(stream.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def worker_payload(
    config: BenchmarkConfig,
    molecule: MoleculeSpec,
    mode: str,
    measurement: str,
) -> dict[str, Any]:
    backend = "gpu" if mode.startswith("gpu") else "cpu"
    return {
        "operation": "measure",
        "source_root": str(config.source_root),
        "functional": config.functional,
        "basis": config.basis,
        "grid_level": config.grid_level,
        "grid_alignment": config.grid_alignment,
        "max_memory_mb": config.max_memory_mb,
        "cpu_threads": config.cpu_threads,
        "backend": backend,
        "forced_dense": mode.endswith("_dense"),
        "mode": mode,
        "measurement": measurement,
        "molecule": {
            "carbon_count": molecule.carbon_count,
            "formula": molecule.formula,
            "atom_text": molecule.atom_text,
            "coordinate_sha256": molecule.coordinate_sha256,
        },
    }


def new_result_document(
    config: BenchmarkConfig,
    molecules: tuple[MoleculeSpec, ...],
    source: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    created_at = utc_now()
    return {
        "schema_version": 1,
        "created_at": created_at,
        "updated_at": created_at,
        "run_label": config.run_label,
        "source": source,
        "environment": environment,
        "configuration": config.as_json(),
        "geometry": GEOMETRY_PARAMETERS,
        "worker_sha256": runner_sha256(),
        "molecules": {
            molecule.formula: {
                **molecule.as_json(),
                "observed": None,
                "modes": {mode: {} for mode in MODES},
            }
            for molecule in molecules
        },
    }


def validate_resume_document(
    document: dict[str, Any], config: BenchmarkConfig, source: dict[str, Any]
) -> None:
    if document.get("schema_version") != 1:
        raise ValueError("Cannot resume a result file with a different schema version")
    if document.get("worker_sha256") != runner_sha256():
        raise ValueError(
            "Cannot resume results created by a different runner implementation"
        )
    if document.get("source", {}).get("commit") != source["commit"]:
        raise ValueError("Cannot resume results from a different Git commit")
    if document.get("configuration") != config.as_json():
        raise ValueError(
            "Cannot resume results created with a different benchmark configuration"
        )


def merge_worker_result(
    molecule_record: dict[str, Any], mode: str, measurement: str, result: dict[str, Any]
) -> None:
    result = dict(result)
    system = result.pop("system", None)
    route = result.pop("route", None)
    if system is not None:
        observed = molecule_record.get("observed")
        if observed is not None and observed != system:
            raise ValueError(
                f"Worker system metadata changed for {molecule_record['formula']}"
            )
        molecule_record["observed"] = system
    mode_record = molecule_record["modes"][mode]
    if route is not None:
        existing_route = mode_record.get("route")
        if existing_route is not None and existing_route != route:
            raise ValueError(
                f"Worker route metadata changed for {molecule_record['formula']} {mode}"
            )
        mode_record["route"] = route
    if measurement == "runtime" and "runtime_seconds" in result:
        result["runtime_samples_seconds"] = [result.pop("runtime_seconds")]
    mode_record[measurement] = result


def result_path(config: BenchmarkConfig, source: dict[str, Any]) -> Path:
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", config.run_label).strip("-.")
    if not safe_label:
        raise ValueError("The run label must contain a filename-safe character")
    return (
        config.results_dir
        / f"skala-pyscf-ao-screening-{safe_label}-{source['commit'][:12]}.json"
    )


def run_worker_preflight(config: BenchmarkConfig) -> dict[str, Any]:
    result = execute_worker(
        {"operation": "environment", "source_root": str(config.source_root)}, config
    )
    if result["status"] != "ok":
        raise RuntimeError(f"Benchmark preflight failed: {result}")
    environment = result["environment"]
    imported_path = Path(environment["imported_skala"])
    imported_path.relative_to(config.source_root / "src")
    required_packages = ("skala", "pyscf", "torch", "memray")
    missing = [
        name for name in required_packages if not environment["packages"].get(name)
    ]
    if missing:
        raise RuntimeError(f"Worker environment is missing packages: {missing}")
    return environment


def atom_distance(left: Atom, right: Atom) -> float:
    return math.dist(left[1:], right[1:])


def validate_molecule_ladder(config: BenchmarkConfig) -> None:
    from pyscf import gto

    assert len(FULL_MOLECULE_LADDER) == len(FULL_CARBON_COUNTS)
    assert len(
        {molecule.coordinate_sha256 for molecule in FULL_MOLECULE_LADDER}
    ) == len(FULL_CARBON_COUNTS)
    for molecule, expected_aos in zip(
        FULL_MOLECULE_LADDER, EXPECTED_AO_COUNTS, strict=True
    ):
        carbon_count = molecule.carbon_count
        hydrogen_count = 2 * carbon_count + 2
        assert molecule.formula == f"C{carbon_count}H{hydrogen_count}"
        assert len(molecule.atoms) == carbon_count + hydrogen_count
        assert molecule.atoms == generate_alkane_atoms(carbon_count)

        carbons = molecule.atoms[:carbon_count]
        hydrogens = molecule.atoms[carbon_count:]
        for left, right in pairwise(carbons):
            assert math.isclose(
                atom_distance(left, right),
                CARBON_CARBON_BOND_ANGSTROM,
                abs_tol=1e-12,
            )
        for hydrogen in hydrogens:
            nearest_carbon = min(atom_distance(hydrogen, carbon) for carbon in carbons)
            assert math.isclose(
                nearest_carbon,
                CARBON_HYDROGEN_BOND_ANGSTROM,
                abs_tol=1e-12,
            )

        mol = gto.M(
            atom=molecule.atom_text,
            basis=config.basis,
            charge=0,
            spin=0,
            unit="Angstrom",
            cart=False,
            verbose=0,
        )
        assert mol.nao_nr() == expected_aos == molecule.expected_aos
        assert mol.nelectron % 2 == 0


def run_benchmark(
    config: BenchmarkConfig,
    molecules: tuple[MoleculeSpec, ...],
    environment: dict[str, Any],
) -> Path:
    source = source_metadata(config.source_root)
    output_path = result_path(config, source)
    if output_path.exists():
        document = json.loads(output_path.read_text(encoding="utf-8"))
        validate_resume_document(document, config, source)
    else:
        document = new_result_document(config, molecules, source, environment)
        atomic_write_json(output_path, document)

    cuda_available = bool(document["environment"]["cuda"]["available"])
    for mode in MODES:
        backend = "gpu" if mode.startswith("gpu") else "cpu"
        for measurement in MEASUREMENTS:
            blocked_by: dict[str, Any] | None = None
            for molecule in molecules:
                molecule_record = document["molecules"][molecule.formula]
                existing = molecule_record["modes"][mode].get(measurement)
                if existing and existing.get("status") in TERMINAL_STATUSES:
                    if existing["status"] in {"oom", "timeout"}:
                        blocked_by = {
                            "formula": molecule.formula,
                            "status": existing["status"],
                        }
                    continue

                result: dict[str, Any]
                if backend == "gpu" and not cuda_available:
                    result = {
                        "status": "error",
                        "mode": mode,
                        "measurement": measurement,
                        "error": "CUDA is not available in the worker environment",
                    }
                elif blocked_by is not None:
                    result = {
                        "status": "skipped_after_resource_failure",
                        "mode": mode,
                        "measurement": measurement,
                        "blocked_by": blocked_by,
                    }
                else:
                    result = execute_measurement(
                        worker_payload(config, molecule, mode, measurement), config
                    )

                merge_worker_result(molecule_record, mode, measurement, result)
                document["updated_at"] = utc_now()
                atomic_write_json(output_path, document)
                if result["status"] in {"oom", "timeout"}:
                    blocked_by = {
                        "formula": molecule.formula,
                        "status": result["status"],
                    }
                print(
                    f"{mode:9s} {measurement:7s} {molecule.formula:8s} "
                    f"{result['status']}"
                )
    return output_path


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark one Skala XC/Vxc evaluation on CPU and GPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Result label, normally the revision name such as 'mr' or 'main'.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Skala checkout whose src/skala package is benchmarked.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RUNNER_ROOT / "benchmarks" / "results",
        help="Directory for commit-labelled JSON output.",
    )
    parser.add_argument("--functional", default="skala-1.1")
    parser.add_argument("--basis", default="def2-qzvpp")
    parser.add_argument("--grid-level", type=int, default=1)
    parser.add_argument("--max-memory-mb", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--runtime-repetitions", type=int, default=3)
    parser.add_argument("--timeout-minutes", type=float, default=30.0)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only C2H6 instead of the full 11-molecule ladder.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate geometry, dependencies, source import, and CUDA without measurements.",
    )
    return parser.parse_args(argv)


def config_from_arguments(arguments: argparse.Namespace) -> BenchmarkConfig:
    if arguments.timeout_minutes <= 0:
        raise ValueError("--timeout-minutes must be positive")
    if arguments.threads <= 0:
        raise ValueError("--threads must be positive")
    if arguments.runtime_repetitions <= 0:
        raise ValueError("--runtime-repetitions must be positive")
    source_root = arguments.source_root.expanduser().resolve()
    if not (source_root / "src" / "skala").is_dir():
        raise FileNotFoundError(f"No src/skala package below {source_root}")
    return BenchmarkConfig(
        source_root=source_root,
        results_dir=arguments.results_dir.expanduser().resolve(),
        run_label=arguments.label,
        functional=arguments.functional,
        basis=arguments.basis,
        grid_level=arguments.grid_level,
        max_memory_mb=arguments.max_memory_mb,
        cpu_threads=arguments.threads,
        runtime_repetitions=arguments.runtime_repetitions,
        worker_timeout_seconds=round(arguments.timeout_minutes * 60),
        smoke_run=arguments.smoke,
    )


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    config = config_from_arguments(arguments)
    validate_molecule_ladder(config)
    environment = run_worker_preflight(config)
    print("Benchmark configuration:")
    print(json.dumps(config.as_json(), indent=2, sort_keys=True))
    print(f"Skala import: {environment['imported_skala']}")
    print(f"Python: {environment['python_executable']}")
    print(f"CUDA: {environment['cuda']}")
    if arguments.preflight_only:
        print("Preflight passed.")
        return 0

    molecules = FULL_MOLECULE_LADDER[:1] if config.smoke_run else FULL_MOLECULE_LADDER
    output_path = run_benchmark(config, molecules, environment)
    print(f"Results written to {output_path}")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        worker_main()
    else:
        raise SystemExit(main())
