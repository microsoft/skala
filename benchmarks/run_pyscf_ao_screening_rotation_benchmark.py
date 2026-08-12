"""Benchmark AO screening across rotations of one approximately 900-AO molecule.

The default grid applies ``Rz(azimuth) @ Ry(polar)`` to C7H16 (879 AOs with
def2-qzvpp). Azimuth runs from 0 through 330 degrees and polar angle runs from
0 through 150 degrees, both in 30-degree steps, for 72 orientations per mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import run_pyscf_ao_screening_benchmark as benchmark

MODES = ("gpu", "cpu_dense", "cpu_screened")
MEASUREMENTS = ("runtime", "memory")
BASE_MOLECULE = benchmark.make_molecule_spec(7)


@dataclass(frozen=True)
class Orientation:
    azimuth_degrees: int
    polar_degrees: int

    @property
    def key(self) -> str:
        return f"azimuth_{self.azimuth_degrees:03d}_polar_{self.polar_degrees:03d}"

    def as_json(self) -> dict[str, int]:
        return {
            "azimuth_degrees": self.azimuth_degrees,
            "polar_degrees": self.polar_degrees,
        }


@dataclass(frozen=True)
class RotationBenchmarkConfig(benchmark.BenchmarkConfig):
    azimuth_step_degrees: int = 30
    polar_step_degrees: int = 30

    @property
    def azimuth_angles(self) -> tuple[int, ...]:
        return tuple(range(0, 360, self.azimuth_step_degrees))

    @property
    def polar_angles(self) -> tuple[int, ...]:
        return tuple(range(0, 180, self.polar_step_degrees))

    @property
    def full_orientations(self) -> tuple[Orientation, ...]:
        return tuple(
            Orientation(azimuth, polar)
            for azimuth in self.azimuth_angles
            for polar in self.polar_angles
        )

    @property
    def orientations(self) -> tuple[Orientation, ...]:
        orientations = self.full_orientations
        return orientations[:1] if self.smoke_run else orientations

    def as_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_root"] = str(self.source_root)
        data["results_dir"] = str(self.results_dir)
        data["azimuth_angles_degrees"] = list(self.azimuth_angles)
        data["polar_angles_degrees"] = list(self.polar_angles)
        data["orientation_count"] = len(self.orientations)
        data["full_orientation_count"] = len(self.full_orientations)
        data["modes"] = list(MODES)
        data["measurements"] = list(MEASUREMENTS)
        data["worker_thread_environment"] = self.worker_thread_environment
        return data


def rotate_atoms(
    atoms: tuple[benchmark.Atom, ...], orientation: Orientation
) -> tuple[benchmark.Atom, ...]:
    azimuth = math.radians(orientation.azimuth_degrees)
    polar = math.radians(orientation.polar_degrees)
    cos_azimuth = math.cos(azimuth)
    sin_azimuth = math.sin(azimuth)
    cos_polar = math.cos(polar)
    sin_polar = math.sin(polar)

    rotated: list[benchmark.Atom] = []
    for element, x, y, z in atoms:
        polar_x = cos_polar * x + sin_polar * z
        polar_z = -sin_polar * x + cos_polar * z
        rotated.append(
            (
                element,
                cos_azimuth * polar_x - sin_azimuth * y,
                sin_azimuth * polar_x + cos_azimuth * y,
                polar_z,
            )
        )
    return tuple(rotated)


def rotated_molecule(orientation: Orientation) -> benchmark.MoleculeSpec:
    return benchmark.MoleculeSpec(
        carbon_count=BASE_MOLECULE.carbon_count,
        expected_aos=BASE_MOLECULE.expected_aos,
        formula=BASE_MOLECULE.formula,
        atoms=rotate_atoms(BASE_MOLECULE.atoms, orientation),
    )


def runner_hashes() -> dict[str, str]:
    return {
        "rotation_runner_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "worker_sha256": benchmark.runner_sha256(),
    }


def validate_rotation_grid(config: RotationBenchmarkConfig) -> None:
    from pyscf import gto

    molecules = tuple(rotated_molecule(item) for item in config.full_orientations)
    coordinate_hashes = {molecule.coordinate_sha256 for molecule in molecules}
    if len(coordinate_hashes) != len(molecules):
        raise ValueError("The rotation grid produced duplicate coordinate sets")

    for molecule in molecules:
        for original, rotated in zip(BASE_MOLECULE.atoms, molecule.atoms, strict=True):
            if original[0] != rotated[0] or not math.isclose(
                math.dist((0.0, 0.0, 0.0), original[1:]),
                math.dist((0.0, 0.0, 0.0), rotated[1:]),
                abs_tol=1e-12,
            ):
                raise ValueError("A rotation changed the molecular geometry")

    mol = gto.M(
        atom=BASE_MOLECULE.atom_text,
        basis=config.basis,
        charge=0,
        spin=0,
        unit="Angstrom",
        cart=False,
        verbose=0,
    )
    actual_aos = int(mol.nao_nr())
    if actual_aos != BASE_MOLECULE.expected_aos:
        raise ValueError(
            f"Expected {BASE_MOLECULE.expected_aos} AOs for {BASE_MOLECULE.formula} "
            f"with {config.basis}, got {actual_aos}"
        )


def run_worker_preflight(config: RotationBenchmarkConfig) -> dict[str, Any]:
    result = benchmark.execute_worker(
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


def result_path(config: RotationBenchmarkConfig, source: dict[str, Any]) -> Path:
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", config.run_label).strip("-.")
    if not safe_label:
        raise ValueError("The run label must contain a filename-safe character")
    return (
        config.results_dir
        / f"skala-pyscf-ao-screening-rotations-{safe_label}-{source['commit'][:12]}.json"
    )


def new_result_document(
    config: RotationBenchmarkConfig,
    source: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    created_at = benchmark.utc_now()
    return {
        "schema_version": 1,
        "benchmark": "pyscf_ao_screening_rotations",
        "created_at": created_at,
        "updated_at": created_at,
        "run_label": config.run_label,
        "source": source,
        "environment": environment,
        "configuration": config.as_json(),
        "geometry": {
            **benchmark.GEOMETRY_PARAMETERS,
            "base_molecule": BASE_MOLECULE.as_json(),
            "rotation_convention": "active Cartesian rotation Rz(azimuth) @ Ry(polar)",
        },
        "runner_hashes": runner_hashes(),
        "orientations": {
            orientation.key: {
                "index": index,
                **orientation.as_json(),
                "coordinate_sha256": rotated_molecule(orientation).coordinate_sha256,
                "observed": None,
                "modes": {mode: {} for mode in MODES},
            }
            for index, orientation in enumerate(config.orientations)
        },
    }


def validate_resume_document(
    document: dict[str, Any],
    config: RotationBenchmarkConfig,
    source: dict[str, Any],
) -> None:
    if document.get("schema_version") != 1:
        raise ValueError("Cannot resume a result file with a different schema version")
    if document.get("runner_hashes") != runner_hashes():
        raise ValueError(
            "Cannot resume results created by different runner implementations"
        )
    if document.get("source", {}).get("commit") != source["commit"]:
        raise ValueError("Cannot resume results from a different Git commit")
    if document.get("configuration") != config.as_json():
        raise ValueError(
            "Cannot resume results created with a different benchmark configuration"
        )


def run_benchmark(config: RotationBenchmarkConfig, environment: dict[str, Any]) -> Path:
    source = benchmark.source_metadata(config.source_root)
    output_path = result_path(config, source)
    if output_path.exists():
        document = json.loads(output_path.read_text(encoding="utf-8"))
        validate_resume_document(document, config, source)
    else:
        document = new_result_document(config, source, environment)
        benchmark.atomic_write_json(output_path, document)

    cuda_available = bool(document["environment"]["cuda"]["available"])
    for mode in MODES:
        for measurement in MEASUREMENTS:
            blocked_by: dict[str, Any] | None = None
            for orientation in config.orientations:
                orientation_record = document["orientations"][orientation.key]
                existing = orientation_record["modes"][mode].get(measurement)
                if existing and existing.get("status") in benchmark.TERMINAL_STATUSES:
                    if existing["status"] in {"oom", "timeout"}:
                        blocked_by = {
                            "orientation": orientation.key,
                            "status": existing["status"],
                        }
                    continue

                result: dict[str, Any]
                if mode == "gpu" and not cuda_available:
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
                    molecule = rotated_molecule(orientation)
                    payload = benchmark.worker_payload(
                        config, molecule, mode, measurement
                    )
                    payload["orientation"] = orientation.as_json()
                    result = benchmark.execute_measurement(payload, config)

                benchmark.merge_worker_result(
                    orientation_record, mode, measurement, result
                )
                document["updated_at"] = benchmark.utc_now()
                benchmark.atomic_write_json(output_path, document)
                if result["status"] in {"oom", "timeout"}:
                    blocked_by = {
                        "orientation": orientation.key,
                        "status": result["status"],
                    }
                print(
                    f"{mode:12s} {measurement:7s} {orientation.key} {result['status']}"
                )
    return output_path


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Skala XC/Vxc evaluation for 72 rotations of one 879-AO "
            "molecule on GPU, dense CPU, and screened CPU."
        ),
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
        default=benchmark.DEFAULT_SOURCE_ROOT,
        help="Skala checkout whose src/skala package is benchmarked.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=benchmark.RUNNER_ROOT / "benchmarks" / "results",
        help="Directory for commit-labelled JSON output.",
    )
    parser.add_argument("--functional", default="skala-1.1")
    parser.add_argument("--basis", default="def2-qzvpp")
    parser.add_argument("--grid-level", type=int, default=1)
    parser.add_argument("--max-memory-mb", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--runtime-repetitions", type=int, default=3)
    parser.add_argument("--timeout-minutes", type=float, default=30.0)
    parser.add_argument("--azimuth-step-degrees", type=int, default=30)
    parser.add_argument("--polar-step-degrees", type=int, default=30)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only the unrotated orientation for each mode.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate rotations, dependencies, source import, and CUDA without measurements.",
    )
    return parser.parse_args(argv)


def config_from_arguments(
    arguments: argparse.Namespace,
) -> RotationBenchmarkConfig:
    if arguments.timeout_minutes <= 0:
        raise ValueError("--timeout-minutes must be positive")
    if arguments.threads <= 0:
        raise ValueError("--threads must be positive")
    if arguments.runtime_repetitions <= 0:
        raise ValueError("--runtime-repetitions must be positive")
    if arguments.azimuth_step_degrees <= 0 or 360 % arguments.azimuth_step_degrees:
        raise ValueError("--azimuth-step-degrees must be a positive divisor of 360")
    if arguments.polar_step_degrees <= 0 or 180 % arguments.polar_step_degrees:
        raise ValueError("--polar-step-degrees must be a positive divisor of 180")
    source_root = arguments.source_root.expanduser().resolve()
    if not (source_root / "src" / "skala").is_dir():
        raise FileNotFoundError(f"No src/skala package below {source_root}")
    return RotationBenchmarkConfig(
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
        azimuth_step_degrees=arguments.azimuth_step_degrees,
        polar_step_degrees=arguments.polar_step_degrees,
    )


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    config = config_from_arguments(arguments)
    validate_rotation_grid(config)
    environment = run_worker_preflight(config)
    print("Benchmark configuration:")
    print(json.dumps(config.as_json(), indent=2, sort_keys=True))
    print(
        f"Molecule: {BASE_MOLECULE.formula}, "
        f"{BASE_MOLECULE.expected_aos} AOs with {config.basis}"
    )
    print(f"Skala import: {environment['imported_skala']}")
    print(f"Python: {environment['python_executable']}")
    print(f"CUDA: {environment['cuda']}")
    if arguments.preflight_only:
        print("Preflight passed.")
        return 0

    output_path = run_benchmark(config, environment)
    print(f"Results written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
