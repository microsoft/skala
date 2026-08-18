# SPDX-License-Identifier: MIT

import gzip
import json
from dataclasses import dataclass
from importlib.resources import files

from skala.benchmark.models import Molecule

_DATA_FILE = "dataset.json.gz"


@dataclass(frozen=True, slots=True)
class BenchmarkMolecule:
    mol_hash: str
    name: str
    molecule: Molecule
    num_atoms: int
    num_electrons: int


def load_benchmark_molecules() -> list[BenchmarkMolecule]:
    """Load the bundled benchmark molecule dataset."""
    resource = files("skala.benchmark") / _DATA_FILE
    with gzip.open(resource.open("rb"), "rt", encoding="utf-8") as fh:
        records = json.load(fh)
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
