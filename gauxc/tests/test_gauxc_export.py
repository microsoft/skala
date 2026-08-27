from tempfile import NamedTemporaryFile
from typing import cast

import h5py
import numpy as np
import pytest
from skala_gauxc import write_gauxc_h5_from_pyscf

from pyscf import dft, gto


@pytest.fixture(params=["He", "Li"])
def mol_name(request: pytest.FixtureRequest) -> str:
    return cast(str, request.param)


@pytest.fixture
def basis() -> str:
    return "def2-svp"


@pytest.fixture(params=["cart", "sph"])
def cartesian(request: pytest.FixtureRequest) -> bool:
    return cast(str, request.param) == "cart"


@pytest.fixture
def mol(mol_name: str, basis: str, cartesian: bool) -> gto.Mole:
    match mol_name:
        case "He":
            return gto.M(
                atom="He 0 0 0", basis=basis, cart=cartesian, unit="Bohr", spin=0
            )
        case "Li":
            return gto.M(
                atom="Li 0 0 0", basis=basis, cart=cartesian, unit="Bohr", spin=1
            )
        case _:
            raise ValueError(f"Unknown molecule name: {mol_name}")


@pytest.fixture
def ks(mol: gto.Mole) -> dft.rks.RKS:
    ks = dft.KS(mol, xc="pbe")
    ks.kernel()
    return ks


@pytest.fixture
def dm(ks: dft.rks.RKS) -> np.ndarray:
    result = ks.make_rdm1()
    assert isinstance(result, np.ndarray)
    return result


@pytest.fixture
def exc(ks: dft.rks.RKS) -> float:
    result = ks.scf_summary["exc"]
    assert isinstance(result, float)
    return result


@pytest.fixture
def vxc(ks: dft.rks.RKS, dm: np.ndarray) -> np.ndarray:
    if dm.ndim == 2:
        _, _, vxc = ks._numint.nr_rks(ks.mol, ks.grids, ks.xc, dm)
    else:
        _, _, vxc = ks._numint.nr_uks(ks.mol, ks.grids, ks.xc, dm)
    assert isinstance(vxc, np.ndarray)
    return vxc


def test_write_pyscf(
    mol: gto.Mole, dm: np.ndarray, exc: float, vxc: np.ndarray
) -> None:
    with NamedTemporaryFile(suffix=".h5") as tmp:
        write_gauxc_h5_from_pyscf(tmp.name, mol, dm, exc, vxc)

        with h5py.File(tmp.name, "r") as h5:
            assert "MOLECULE" in h5, "Molecule is missing in h5 export"
            assert "BASIS" in h5, "Basis is missing in h5 export"
            assert "DENSITY_SCALAR" in h5, "Density (a+b) is missing in h5 export"
            assert "DENSITY_Z" in h5, "Density (a-b) is missing in h5 export"
            assert "EXC" in h5, "Exchange-correlation energy is missing in h5 export"
            assert "VXC_SCALAR" in h5, (
                "Exchange-correlation potential (a+b) is missing in h5 export"
            )
            assert "VXC_Z" in h5, (
                "Exchange-correlation potential (a-b) is missing in h5 export"
            )
