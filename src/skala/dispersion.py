"""Cross-platform DFT-D3 backend compatibility."""

import sys
from importlib import import_module
from typing import Any

import numpy as np
from pyscf import gto

_USE_PYSCF_DISPERSION = sys.platform == "linux"
_backend_module = import_module(
    "pyscf.dispersion.dftd3" if _USE_PYSCF_DISPERSION else "dftd3.pyscf"
)


class DFTD3Dispersion:
    """Normalize the DFT-D3 APIs available on supported platforms."""

    def __init__(self, mol: gto.Mole, xc: str):
        self._xc = xc
        self._backend: Any = _backend_module.DFTD3Dispersion(mol, xc)

    def get_energy(self) -> float:
        """Return the dispersion energy as a Python scalar."""
        if _USE_PYSCF_DISPERSION:
            energy = self._backend.get_dispersion()["energy"]
        else:
            energy = self._backend.kernel()[0]
        return float(np.asarray(energy).item())

    def get_gradient(self) -> np.ndarray:
        """Return the dispersion nuclear gradient."""
        if _USE_PYSCF_DISPERSION:
            gradient = self._backend.get_dispersion(grad=True)["gradient"]
        else:
            gradient = self._backend.kernel()[1]
        return np.asarray(gradient)

    def reset(self, mol: gto.Mole) -> "DFTD3Dispersion":
        """Reset the backend for a new molecular geometry."""
        self._backend = _backend_module.DFTD3Dispersion(mol, self._xc)
        return self
