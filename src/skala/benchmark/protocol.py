# SPDX-License-Identifier: MIT

"""The fixed scientific protocol used by the Skala benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

Ansatz = Literal["RKS", "UKS"]


class Device(str, Enum):  # noqa: UP042 - Python 3.10 does not provide StrEnum
    """Compute target requested for a benchmark run."""

    CPU = "cpu"
    GPU = "gpu"


class FunctionalKind(str, Enum):  # noqa: UP042 - Python 3.10 does not provide StrEnum
    """Implementation used to evaluate an exchange-correlation functional."""

    SKALA = "skala"
    NATIVE = "native"


@dataclass(frozen=True, slots=True)
class FunctionalSpec:
    """One functional in the benchmark comparison."""

    name: str
    kind: FunctionalKind


@dataclass(frozen=True, slots=True)
class BenchmarkProtocol:
    """Scientific settings held constant across a benchmark sweep."""

    bases: tuple[str, ...]
    functionals: tuple[FunctionalSpec, ...]
    ansatz: Ansatz = "UKS"
    density_fit: bool = True
    auxbasis: str = "def2-universal-jkfit"
    grid_level: int = 3
    conv_tol: float = 5e-6


DEFAULT_PROTOCOL = BenchmarkProtocol(
    bases=("def2-svp", "def2-tzvp", "def2-qzvp"),
    functionals=(
        FunctionalSpec("skala-1.1", FunctionalKind.SKALA),
        FunctionalSpec("r2scan", FunctionalKind.NATIVE),
        FunctionalSpec("b3lyp5", FunctionalKind.NATIVE),
        FunctionalSpec("m06-2x", FunctionalKind.NATIVE),
    ),
)
