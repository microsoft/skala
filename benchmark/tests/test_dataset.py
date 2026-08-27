# SPDX-License-Identifier: MIT

"""Tests for the benchmark molecule set and the fetch that rebuilds it."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from skala_benchmark.dataset import (
    DATASET_DIR_ENV,
    dataset_path,
    default_dataset_dir,
    load_benchmark_molecules,
    read_manifest,
)
from skala_benchmark.fetch import CONFORMER_SI_SHA256, SOURCES, atomic_numbers


def test_manifest_is_packaged_and_complete() -> None:
    """The manifest ships with the package and describes the whole set."""
    entries = read_manifest()
    assert len(entries) == 58
    assert len({entry.mol_hash for entry in entries}) == 58
    assert {entry.source for entry in entries} == {*SOURCES, "conformer-benchmark"}
    for entry in entries:
        assert entry.num_atoms > 0
        assert entry.multiplicity >= 1
        assert entry.path


def test_manifest_holds_no_coordinates() -> None:
    """The structures belong to their source datasets and are not redistributed."""
    from importlib.resources import files

    from skala_benchmark.dataset import MANIFEST_FILE

    text = (files("skala_benchmark") / MANIFEST_FILE).read_text(encoding="utf-8")
    columns = text.splitlines()[0].split(",")
    assert columns == [
        "source",
        "path",
        "name",
        "formula",
        "natoms",
        "charge",
        "multiplicity",
        "subtag",
        "livdft_hash",
    ]


def test_atomic_numbers_ignore_the_case_of_the_source_file() -> None:
    """Turbomole writes symbols in lower case, some XYZ files in upper."""
    assert atomic_numbers(["H", "h"]) == [1, 1]
    assert atomic_numbers(["Cl", "CL", "cl"]) == [17, 17, 17]


def test_manifest_formulas_agree_with_their_atom_counts() -> None:
    """A formula and an atom count that disagree would misdescribe the set."""
    pattern = re.compile(r"([A-Z][a-z]?)(\d*)")
    for entry in read_manifest():
        counts = [int(count or 1) for _, count in pattern.findall(entry.formula)]
        assert sum(counts) == entry.num_atoms, entry.formula


def test_default_dataset_dir_follows_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(DATASET_DIR_ENV, str(tmp_path / "explicit"))
    assert default_dataset_dir() == tmp_path / "explicit"

    monkeypatch.delenv(DATASET_DIR_ENV)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    assert default_dataset_dir() == tmp_path / "cache" / "skala" / "benchmark-dataset"


def test_missing_dataset_names_the_command_that_fetches_it(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="fetch-dataset"):
        load_benchmark_molecules(tmp_path)


def test_dataset_from_another_version_is_rejected(tmp_path: Path) -> None:
    """A stale dataset is refused rather than silently changing the task grid."""
    entries = read_manifest()
    records = [
        {
            "hash": entry.mol_hash,
            "name": entry.formula,
            "atomic_numbers": [1] * entry.num_atoms,
            "geometry_bohr": [[0.0, 0.0, float(i)] for i in range(entry.num_atoms)],
            "num_atoms": entry.num_atoms,
            "molecular_charge": entry.charge,
            "molecular_multiplicity": entry.multiplicity,
            "num_electrons": entry.num_atoms,
        }
        for entry in entries[:-1]
    ]
    path = dataset_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        load_benchmark_molecules(tmp_path)


def test_sources_are_pinned_to_a_commit() -> None:
    """A commit pin is what makes a rebuild reproducible; HEAD would not be."""
    for repository, commit in SOURCES.values():
        assert "/" in repository
        assert len(commit) == 40
        assert set(commit) <= set("0123456789abcdef")


def test_the_one_unpinned_source_is_checksummed() -> None:
    """The conformer SI comes from a URL, so a digest stands in for a commit."""
    assert len(CONFORMER_SI_SHA256) == 64
    assert set(CONFORMER_SI_SHA256) <= set("0123456789abcdef")
    sources = {entry.source for entry in read_manifest()}
    assert sources - set(SOURCES) == {"conformer-benchmark"}
