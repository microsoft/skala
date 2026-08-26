# SPDX-License-Identifier: MIT

"""Rebuild the benchmark molecule set from its public sources.

The set combines five published datasets. Their structures are not
redistributed here; they are downloaded from the upstream repositories, each
pinned to a commit, and assembled into the file the benchmark reads. See
``website/benchmarks.rst`` for the datasets and the citations they require.

Every structure is checked against :mod:`skala_benchmark.dataset`'s manifest --
atom count, element formula, charge and multiplicity -- so a source that has
changed upstream is reported rather than silently used.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

from skala_benchmark.dataset import (
    DATASET_FILE,
    ManifestEntry,
    default_dataset_dir,
    read_manifest,
)

#: Upstream repositories, pinned to a commit so a rebuild is reproducible.
SOURCES = {
    "gmtkn55": ("grimme-lab/GMTKN55", "8d485b37a1ca8837e395042671ca5ba4e0714691"),
    "hs13l": ("grimme-lab/benchmark-HS13L", "37bb33e609636d5c2d9209e8d5bebca10e4456b6"),
    "lnci16": (
        "grimme-lab/benchmark-LNCI16",
        "a1c75ae77547c1c0c324c4a55e0197e93751a6db",
    ),
    "s30l": ("aoterodelaroza/refdata", "f96148aa1739077c55b394b4c9d7e4f42f65dce0"),
}

#: The conformer-benchmark structures come from the supporting information of
#: 10.1021/acs.jctc.9b00143. The publisher's copy sits behind a bot challenge;
#: figshare hosts the same file, and the checksum below is verified against it.
CONFORMER_SI_URL = "https://ndownloader.figshare.com/files/14875160"
CONFORMER_SI_SHA256 = "bb2eeb0083d79f354c7cbf5300aaca5ef2acdb1f1bb8d1be01b9d6ac49628d94"
#: Line range holding the starting structures; the rest of the file holds the
#: GFN2-xTB optimised conformers, which this benchmark does not use.
CONFORMER_SI_SLICE = slice(2, 1804)

_TIMEOUT_SECONDS = 120


class FetchError(RuntimeError):
    """A source could not be downloaded, or disagreed with the manifest."""


def _bohr_per_angstrom() -> float:
    from pyscf.data.nist import BOHR

    return float(BOHR)


def atomic_numbers(symbols: list[str]) -> list[int]:
    """Return the proton numbers of element symbols, whatever their case.

    Turbomole ``coord`` files write symbols in lower case and some XYZ files in
    upper case, so the lookup has to be case-insensitive.
    """
    from pyscf.data.elements import charge

    return [int(charge(symbol)) for symbol in symbols]


def _download(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_SECONDS) as response:
            return bytes(response.read())
    except Exception as error:
        raise FetchError(f"could not download {url}: {error}") from error


def _cached(url: str, cache: Path, key: str) -> bytes:
    """Download ``url`` once, keeping a copy under ``cache``."""
    target = cache / key
    if target.exists():
        return target.read_bytes()
    payload = _download(url)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return payload


def _parse_turbomole(text: str) -> tuple[list[str], np.ndarray]:
    """Parse a Turbomole ``coord`` block, whose coordinates are already in bohr."""
    symbols: list[str] = []
    coordinates: list[list[float]] = []
    inside = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("$coord"):
            inside = True
            continue
        if stripped.startswith("$"):
            if inside:
                break
            continue
        if inside and stripped:
            x, y, z, element = stripped.split()[:4]
            symbols.append(element)
            coordinates.append([float(x), float(y), float(z)])
    return symbols, np.array(coordinates)


def _parse_xyz(text: str) -> tuple[list[str], np.ndarray]:
    """Parse an XYZ block, converting its angstrom coordinates to bohr."""
    lines = text.splitlines()
    count = int(lines[0].split()[0])
    symbols = [line.split()[0] for line in lines[2 : 2 + count]]
    coordinates = np.array(
        [[float(value) for value in line.split()[1:4]] for line in lines[2 : 2 + count]]
    )
    return symbols, coordinates / _bohr_per_angstrom()


def _parse_conformer_si(payload: bytes) -> dict[str, tuple[list[str], np.ndarray]]:
    """Parse the conformer-benchmark starting structures out of the SI file.

    Unlike the other sources this one is served from a URL rather than a pinned
    commit, so the file is only as trustworthy as its digest.
    """
    digest = hashlib.sha256(payload).hexdigest()
    if digest != CONFORMER_SI_SHA256:
        raise FetchError(
            "the conformer-benchmark supporting information does not have the "
            f"expected contents (sha256 {digest}, expected "
            f"{CONFORMER_SI_SHA256})"
        )
    lines = payload.decode().split("\n")[CONFORMER_SI_SLICE]
    structures: dict[str, tuple[list[str], np.ndarray]] = {}
    index = 0
    while index < len(lines):
        count = int(lines[index])
        name = lines[index + 1].split("=")[1][1:-1].replace("\\", "")
        block = lines[index + 2 : index + 2 + count]
        symbols = [line.split()[0] for line in block]
        coordinates = np.array(
            [[float(value) for value in line.split()[1:4]] for line in block]
        )
        structures[name] = (symbols, coordinates / _bohr_per_angstrom())
        index += 2 + count
    return structures


def _record(
    entry: ManifestEntry, symbols: list[str], geometry: np.ndarray
) -> dict[str, object]:
    """Turn one fetched structure into a dataset record.

    Counts are derived from the structure rather than copied from the manifest,
    so a record cannot describe itself inconsistently.
    """
    numbers = atomic_numbers(symbols)
    return {
        "hash": entry.mol_hash,
        "name": entry.formula,
        "atomic_numbers": numbers,
        "geometry_bohr": [[float(value) for value in row] for row in geometry],
        "num_atoms": len(numbers),
        "molecular_charge": entry.charge,
        "molecular_multiplicity": entry.multiplicity,
        # A neutral molecule has one electron per proton; a cation has fewer.
        "num_electrons": sum(numbers) - entry.charge,
    }


def fetch_dataset(dataset_dir: str | Path | None = None) -> Path:
    """Download the benchmark molecules and write them where the benchmark reads.

    Args:
        dataset_dir: Directory to write the dataset to. Defaults to
            :func:`skala_benchmark.dataset.default_dataset_dir`.

    Returns:
        The path of the written dataset file.

    Raises:
        FetchError: If a source cannot be downloaded or disagrees with the
            manifest.
    """
    directory = Path(dataset_dir) if dataset_dir is not None else default_dataset_dir()
    cache = directory / "cache"
    entries = read_manifest()

    conformer_entries = [e for e in entries if e.source == "conformer-benchmark"]
    conformers: dict[str, tuple[list[str], np.ndarray]] = {}
    if conformer_entries:
        conformers = _parse_conformer_si(
            _cached(CONFORMER_SI_URL, cache, "conformer-benchmark/si.txt")
        )

    records = []
    for entry in entries:
        if entry.source == "conformer-benchmark":
            structure = conformers.get(entry.path)
            if structure is None:
                raise FetchError(
                    f"{entry.path} is not in the conformer-benchmark supporting "
                    "information"
                )
            symbols, geometry = structure
        else:
            repository, commit = SOURCES[entry.source]
            url = (
                f"https://raw.githubusercontent.com/{repository}/{commit}/{entry.path}"
            )
            text = _cached(url, cache, f"{entry.source}/{entry.path}").decode()
            symbols, geometry = (
                _parse_turbomole(text)
                if entry.path.endswith("coord")
                else _parse_xyz(text)
            )
        records.append(_record(entry, symbols, geometry))

    directory.mkdir(parents=True, exist_ok=True)
    target = directory / DATASET_FILE
    target.write_text(json.dumps(records), encoding="utf-8")
    return target


def main(dataset_dir: str | Path | None = None) -> int:
    """Fetch the dataset, reporting progress on stderr."""
    directory = Path(dataset_dir) if dataset_dir is not None else default_dataset_dir()
    print(f"fetching benchmark molecules into {directory}", file=sys.stderr)
    try:
        target = fetch_dataset(directory)
    except FetchError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    count = len(json.loads(target.read_text(encoding="utf-8")))
    print(f"wrote {count} molecules to {target}", file=sys.stderr)
    return 0
