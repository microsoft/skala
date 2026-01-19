import numpy as np
import pytest

try:
    import ase
    from ase.build import molecule
    from ase.calculators import calculator

    from skala.ase import Skala
except ModuleNotFoundError:
    ase = None


@pytest.fixture(params=["pbe", "tpss", "skala"])
def xc(request) -> str:
    return request.param


@pytest.mark.skipif(ase is None, reason="ASE is not installed")
def test_calc(xc: str) -> None:
    atoms = molecule("H2O")
    atoms.calc = Skala(
        xc=xc, basis="def2-svp", with_density_fit=True, auxbasis="def2-svp-jkfit"
    )

    energy = atoms.get_potential_energy()

    reference_energy, reference_fnorm, reference_dipm = {
        "pbe": (-2075.4896490374904, 0.6395142802693002, 0.40519674886465107),
        "tpss": (-2077.88636677525, 0.5863078815838786, 0.40534133865824284),
        "skala": (-2076.4586374337177, 1.127975901679744, 0.4173008295594236),
    }[xc]

    assert (
        pytest.approx(energy, rel=1e-3) == reference_energy
    ), f"Energy mismatch for {xc}: {energy} vs {reference_energy}"
    assert (
        pytest.approx(np.linalg.norm(np.abs(atoms.get_forces())), rel=1e-3)
        == reference_fnorm
    ), f"Forces norm mismatch for {xc}: {np.linalg.norm(np.abs(atoms.get_forces()))} vs {reference_fnorm}"
    assert (
        pytest.approx(np.linalg.norm(atoms.get_dipole_moment()), rel=1e-3)
        == reference_dipm
    ), f"Dipole moment mismatch for {xc}: {np.linalg.norm(atoms.get_dipole_moment())} vs {reference_dipm}"


@pytest.mark.skipif(ase is None, reason="ASE is not installed")
def test_missing_basis() -> None:
    atoms = molecule("H2O")
    atoms.calc = Skala(xc="pbe", with_density_fit=True, auxbasis="def2-svp-jkfit")

    with pytest.raises(
        calculator.InputError, match="Basis set must be specified in the parameters."
    ):
        atoms.get_potential_energy()
