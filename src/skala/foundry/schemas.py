# SPDX-License-Identifier: MIT

from typing import Literal, TypeAlias

import numpy as np
import qcelemental as qcel
from pydantic import BaseModel, Field

GridLevelOptions: TypeAlias = Literal[
    "ultrafine",
    "superfine",
    "fine",
]

BasisOptions: TypeAlias = Literal["def2-svp", "def2-tzvp", "def2-qzvp", "ma-def2-qzvp"]


TaskState: TypeAlias = Literal["succeeded", "failed", "running", "queued", "canceled"]


class SkalaConfig(BaseModel):  # type: ignore[misc]
    basis: BasisOptions = "def2-qzvp"
    grid_level: GridLevelOptions = "ultrafine"
    max_num_scf_steps: int = 100


class Molecule(BaseModel):  # type: ignore[misc]
    """Molecule representation based on qcelemental's Molecule model."""

    geometry: list[float] = Field(
        description="Flat list of atom positions [x1, y1, z1, x2, y2, z2, ...] in Bohr."
    )
    symbols: list[str] = Field(
        description="List of atomic symbols, e.g. ['H', 'O', 'H']"
    )
    molecular_charge: float = Field(
        description="The net electrostatic charge of the molecule."
    )
    molecular_multiplicity: int = Field(
        description="The total multiplicity of the molecule."
    )

    @classmethod
    def from_qcel(cls, molecule: qcel.models.Molecule) -> "Molecule":
        geometry = molecule.geometry
        if isinstance(geometry, np.ndarray):
            geometry = geometry.flatten().tolist()
        return cls(
            geometry=geometry,
            symbols=molecule.symbols,
            molecular_charge=molecule.molecular_charge,
            molecular_multiplicity=molecule.molecular_multiplicity,
        )

    def to_qcel(self) -> qcel.models.Molecule:
        return qcel.models.Molecule(
            geometry=self.geometry,
            symbols=self.symbols,
            molecular_charge=self.molecular_charge,
            molecular_multiplicity=self.molecular_multiplicity,
        )


class SkalaInput(BaseModel):  # type: ignore[misc]
    molecule: Molecule
    input_config: SkalaConfig = Field(default_factory=SkalaConfig)


class SkalaOutput(BaseModel):  # type: ignore[misc]
    total_energy: float
    energy_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of energy components. Values should sum to total_energy.",
    )
    num_scf_iterations: int = Field(description="Number of SCF iterations performed.")
    dipole_moment: list[float] = Field(
        description="Dipole moment vector in electron Bohr (x, y, z)."
    )


class TaskStatus(BaseModel):  # type: ignore[misc]
    status: TaskState
    num_tasks_ahead: int
    exception: str | None = None
    output: SkalaOutput | None = None
