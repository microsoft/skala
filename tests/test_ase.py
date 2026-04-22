import numpy as np
import pytest

pytest.importorskip("ase")

from ase.build import molecule
from ase.calculators import calculator

from skala.ase import Skala


@pytest.mark.parametrize("xc", ["pbe", "tpss", "skala-1.0", "skala-1.1"])
def test_calc(xc: str) -> None:
    atoms = molecule("H2O")  # type: ignore[no-untyped-call]
    atoms.calc = Skala(
        xc=xc,
        basis="def2-svp",
        with_density_fit=True,
        auxbasis="def2-svp-jkfit",
    )

    energy = atoms.get_potential_energy()

    reference_energy, reference_fnorm, reference_dipole_moment = {
        "pbe": (-2075.4896490374904, 0.6395142802693002, 0.40519674886465107),
        "tpss": (-2077.88636677525, 0.5863078815838786, 0.40534133865824284),
        "skala-1.0": (-2076.4586374337177, 1.127975901679744, 0.4173008295594236),
        "skala-1.1": (-2076.839069353949, 0.5614649968829959, 0.41354587147386074),
    }[xc]

    assert pytest.approx(energy, rel=1e-3) == reference_energy, (
        f"Energy mismatch for {xc}: {energy} vs {reference_energy}"
    )
    assert (
        pytest.approx(np.linalg.norm(np.abs(atoms.get_forces())), rel=1e-3)
        == reference_fnorm
    ), (
        f"Forces norm mismatch for {xc}: {np.linalg.norm(np.abs(atoms.get_forces()))} vs {reference_fnorm}"
    )
    assert (
        pytest.approx(np.linalg.norm(atoms.get_dipole_moment()), rel=1e-3)
        == reference_dipole_moment
    ), (
        f"Dipole moment mismatch for {xc}: {np.linalg.norm(atoms.get_dipole_moment())} vs {reference_dipole_moment}"
    )


def test_missing_basis() -> None:
    atoms = molecule("H2O")  # type: ignore[no-untyped-call]
    atoms.calc = Skala(xc="pbe", with_density_fit=True, auxbasis="def2-svp-jkfit")

    with pytest.raises(
        calculator.InputError, match="Basis set must be specified in the parameters."
    ):
        atoms.get_potential_energy()


def test_ks_config() -> None:
    atoms = molecule("H2O")  # type: ignore[no-untyped-call]
    atoms.calc = Skala(
        xc="pbe",
        basis="def2-svp",
        with_density_fit=True,
        auxbasis="def2-svp-jkfit",
        ks_config={"conv_tol": 1e-6},
    )

    energy = atoms.get_potential_energy()

    assert atoms.calc._ks.base.conv_tol == 1e-6, (
        "KS solver convergence tolerance not set correctly"
    )

    reference_energy = -2075.4896490374904
    assert pytest.approx(energy, rel=1e-3) == reference_energy, (
        f"Energy mismatch with custom KS config: {energy} vs {reference_energy}"
    )
