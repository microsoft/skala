# SPDX-License-Identifier: MIT

"""Domain records shared by benchmark planning and worker execution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Molecule:
    """Geometry and electronic state needed to build a PySCF molecule."""

    atomic_numbers: list[int]
    geometry_bohr: list[list[float]]
    charge: int = 0
    multiplicity: int = 1
