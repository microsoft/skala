"""Cross-platform DFT-D3 backend compatibility."""

from __future__ import annotations

from typing import Self

import numpy as np
from pyscf import gto

try:
    from pyscf.dispersion.dftd3 import (  # type: ignore[import-not-found, unused-ignore]
        DFTD3Dispersion as _PySCFDFTD3Dispersion,
    )

    class DFTD3Dispersion:  # pyright: ignore[reportRedeclaration]
        """Normalize the PySCF DFT-D3 API."""

        def __init__(self, mol: gto.Mole, xc: str):
            self._xc = xc
            self._backend = _PySCFDFTD3Dispersion(mol, xc)

        def get_energy(self) -> float:
            """Return the dispersion energy as a Python scalar."""
            energy = self._backend.get_dispersion()["energy"]
            return float(np.asarray(energy).item())

        def get_gradient(self) -> np.ndarray:
            """Return the dispersion nuclear gradient."""
            gradient = self._backend.get_dispersion(grad=True)["gradient"]
            return np.asarray(gradient)

        def reset(self, mol: gto.Mole) -> Self:
            """Reset the backend for a new molecular geometry."""
            self._backend = _PySCFDFTD3Dispersion(mol, self._xc)
            return self
except ModuleNotFoundError as error:
    if error.name not in {"pyscf.dispersion", "pyscf.dispersion.dftd3"}:
        raise

    from dftd3.pyscf import (  # type: ignore[import-not-found, unused-ignore]
        DFTD3Dispersion as _StandaloneDFTD3Dispersion,
    )

    class DFTD3Dispersion:  # type: ignore[no-redef]
        """Normalize the standalone DFT-D3 API."""

        def __init__(self, mol: gto.Mole, xc: str):
            self._xc = xc
            self._backend = _StandaloneDFTD3Dispersion(mol, xc)

        def get_energy(self) -> float:
            """Return the dispersion energy as a Python scalar."""
            energy = self._backend.kernel()[0]
            return float(np.asarray(energy).item())

        def get_gradient(self) -> np.ndarray:
            """Return the dispersion nuclear gradient."""
            gradient = self._backend.kernel()[1]
            return np.asarray(gradient)

        def reset(self, mol: gto.Mole) -> Self:
            """Reset the backend for a new molecular geometry."""
            self._backend = _StandaloneDFTD3Dispersion(mol, self._xc)
            return self
