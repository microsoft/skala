# SPDX-License-Identifier: MIT

"""The benchmark molecule set: where it lives and how it is described.

The structures themselves are not part of this repository. They belong to the
datasets they were taken from and are downloaded from those sources by
:mod:`skala_benchmark.fetch`; what is kept here is a manifest naming each
molecule, the file it comes from, and enough metadata to verify the download.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from skala_benchmark.models import Molecule

#: Manifest of the benchmark set. Metadata only; it holds no coordinates.
MANIFEST_FILE = "dataset_sources.csv"

#: The assembled molecules, written into the dataset directory by the fetch.
DATASET_FILE = "molecules.json"

#: Overrides the default dataset directory.
DATASET_DIR_ENV = "SKALA_BENCHMARK_DATASET_DIR"


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One molecule of the benchmark set, and where its structure comes from."""

    source: str
    path: str
    name: str
    formula: str
    num_atoms: int
    charge: int
    multiplicity: int
    subtag: str
    mol_hash: str


@dataclass(frozen=True, slots=True)
class BenchmarkMolecule:
    mol_hash: str
    name: str
    molecule: Molecule
    num_atoms: int
    num_electrons: int


def default_dataset_dir() -> Path:
    """Return the directory the dataset is fetched into unless told otherwise.

    Honours ``SKALA_BENCHMARK_DATASET_DIR``, then ``XDG_CACHE_HOME``, and falls
    back to ``~/.cache``. A cache directory rather than the package itself,
    because the structures are third-party data that a user downloads.
    """
    override = os.environ.get(DATASET_DIR_ENV)
    if override:
        return Path(override).expanduser()
    cache_home = os.environ.get("XDG_CACHE_HOME")
    root = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"
    return root / "skala" / "benchmark-dataset"


def read_manifest() -> list[ManifestEntry]:
    """Return the packaged manifest of benchmark molecules."""
    resource = files("skala_benchmark") / MANIFEST_FILE
    with resource.open("r", encoding="utf-8") as stream:
        return [
            ManifestEntry(
                source=row["source"],
                path=row["path"],
                name=row["name"],
                formula=row["formula"],
                num_atoms=int(row["natoms"]),
                charge=int(row["charge"]),
                multiplicity=int(row["multiplicity"]),
                subtag=row["subtag"],
                mol_hash=row["livdft_hash"],
            )
            for row in csv.DictReader(stream)
        ]


def dataset_path(dataset_dir: str | Path | None = None) -> Path:
    """Return the path of the assembled dataset file."""
    directory = Path(dataset_dir) if dataset_dir is not None else default_dataset_dir()
    return directory / DATASET_FILE


def load_benchmark_molecules(
    dataset_dir: str | Path | None = None,
) -> list[BenchmarkMolecule]:
    """Load the benchmark molecule set from a fetched dataset directory.

    Args:
        dataset_dir: Directory the dataset was fetched into. Defaults to
            :func:`default_dataset_dir`.

    Returns:
        The benchmark molecules, in manifest order.

    Raises:
        FileNotFoundError: If the dataset has not been fetched yet.
        ValueError: If the fetched dataset does not match the manifest, which
            means it was written by a different version of the benchmark.
    """
    path = dataset_path(dataset_dir)
    if not path.exists():
        command = "python -m skala_benchmark fetch-dataset"
        if dataset_dir is not None:
            command += f" --dataset-dir {path.parent}"
        raise FileNotFoundError(
            f"the benchmark molecules are not in {path.parent}.\n"
            "They are downloaded from the datasets they belong to rather than "
            f"shipped with this package. Fetch them once with:\n    {command}"
        )
    records = json.loads(path.read_text(encoding="utf-8"))
    expected = {entry.mol_hash for entry in read_manifest()}
    if {str(record["hash"]) for record in records} != expected:
        raise ValueError(
            f"the dataset in {path.parent} does not match this version of the "
            "benchmark; re-run 'python -m skala_benchmark fetch-dataset'"
        )
    return [
        BenchmarkMolecule(
            mol_hash=record["hash"],
            name=record["name"],
            molecule=Molecule(
                atomic_numbers=record["atomic_numbers"],
                geometry_bohr=record["geometry_bohr"],
                charge=record["molecular_charge"],
                multiplicity=record["molecular_multiplicity"],
            ),
            num_atoms=record["num_atoms"],
            num_electrons=record["num_electrons"],
        )
        for record in records
    ]
