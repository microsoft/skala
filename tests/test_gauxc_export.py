from tempfile import NamedTemporaryFile

import h5py
import numpy as np
import pytest

from pyscf import dft, gto
from skala.gauxc.export import write_gauxc_h5_from_pyscf


@pytest.fixture(params=["He", "Li"])
def mol_name(request) -> str:
    return request.param


@pytest.fixture
def basis() -> str:
    return "def2-svp"


@pytest.fixture(params=["cart", "sph"])
def cartesian(request) -> bool:
    return request.param == "cart"


@pytest.fixture
def mol(mol_name: str, basis: str, cartesian: bool) -> gto.Mole:
    match mol_name:
        case "He":
            return gto.M(atom="He 0 0 0", basis=basis, cart=cartesian, unit="Bohr", spin=0)
        case "Li":
            return gto.M(atom="Li 0 0 0", basis=basis, cart=cartesian, unit="Bohr", spin=1)
        case _:
            raise ValueError(f"Unknown molecule name: {mol_name}")


@pytest.fixture
def dm(mol: gto.Mole) -> np.ndarray:
    ks = dft.KS(mol, xc="pbe")
    ks.kernel()
    return ks.make_rdm1()


def test_write_pyscf(mol: gto.Mole, dm: np.ndarray) -> None:
    with NamedTemporaryFile(suffix=".h5") as tmp:
        write_gauxc_h5_from_pyscf(tmp.name, mol, dm)

        with h5py.File(tmp.name, "r") as h5:
            assert "MOLECULE" in h5
            assert "BASIS" in h5
            assert "DENSITY_SCALAR" in h5
            assert "DENSITY_Z" in h5