# SPDX-License-Identifier: MIT

"""Fixed-density parity tests between Python Skala/PySCF and SkalaXC."""

import hashlib
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest
import skalaxc
import torch
from skala.functional import FunctionalArtifact, load_functional
from skala.functional.base import ExcFunctionalBase
from skala.pyscf.gradients import veff_and_expl_nuc_grad
from skala.pyscf.grids import SkalaGrids
from skala.pyscf.xc_integrator import XCIntegrator

from pyscf import dft, gto, scf
from tests.skalaxc_test_utils import (
    SKALA_1_1_REV1_SHA256,
    ParityTolerances,
    make_skalaxc_integrator,
    pyscf_to_skalaxc,
    uks_density_channels,
)


@dataclass(frozen=True)
class FunctionalCase:
    name: str
    functional: ExcFunctionalBase
    integrator: skalaxc.XCIntegrator
    tolerances: ParityTolerances


CPU_TOLERANCES = {
    "lda": ParityTolerances(1e-7, 1.5e-8, 2e-9, 7.5e-6),
    "pbe": ParityTolerances(1e-7, 1.5e-8, 3e-9, 7.5e-6),
    "tpss": ParityTolerances(1e-7, 4e-8, 1.5e-8, 7.5e-6),
    "skala-1.1": ParityTolerances(7.5e-5, 2.5e-4, 2.5e-4, 1e-4),
}


@pytest.fixture(scope="module")
def molecule() -> gto.Mole:
    return gto.M(
        atom="O 0 0 0; H 0 0 1.1",
        basis="def2-svp",
        spin=1,
        cart=False,
        unit="Angstrom",
        verbose=0,
    )


@pytest.fixture(scope="module")
def density(molecule: gto.Mole) -> npt.NDArray[np.float64]:
    mean_field = scf.UHF(molecule)
    mean_field.chkfile = None
    mean_field.conv_tol = 1e-12
    mean_field.kernel()
    assert mean_field.converged
    result = np.asarray(mean_field.make_rdm1(), dtype=np.float64)
    assert result.shape == (2, molecule.nao_nr(), molecule.nao_nr())
    return result


@pytest.fixture(scope="module")
def pyscf_grid(molecule: gto.Mole) -> SkalaGrids:
    grid = SkalaGrids(molecule)
    grid.level = 5
    grid.prune = None
    grid.radi_method = dft.radi.mura_knowles
    grid.build()
    return grid


def _structural_density(molecule: gto.Mole) -> npt.NDArray[np.float64]:
    basis_size = molecule.nao_nr()
    alpha = np.eye(basis_size, dtype=np.float64) * 0.5
    beta = np.eye(basis_size, dtype=np.float64) * 0.25
    return np.stack((alpha, beta))


def _matrix_error(
    actual: npt.NDArray[np.float64], expected: npt.NDArray[np.float64]
) -> float:
    return float(np.linalg.norm(actual - expected) / actual.shape[0])


@pytest.fixture(scope="module", params=("lda", "pbe", "tpss", "skala-1.1"))
def functional_case(
    request: pytest.FixtureRequest,
    molecule: gto.Mole,
) -> FunctionalCase:
    name = str(request.param)
    functional: ExcFunctionalBase
    if name == "skala-1.1":
        model_path = skalaxc.MODEL_DIR / "skala-1.1.fun"
        with model_path.open("rb") as model_file:
            digest = hashlib.file_digest(model_file, "sha256").hexdigest()
        assert digest == SKALA_1_1_REV1_SHA256
        functional = FunctionalArtifact(model_path, SKALA_1_1_REV1_SHA256).load()
        model = str(model_path)
    else:
        loaded_functional = load_functional(name)
        assert isinstance(loaded_functional, ExcFunctionalBase)
        functional = loaded_functional
        model = name.upper()
    return FunctionalCase(
        name,
        functional,
        make_skalaxc_integrator(molecule, model),
        CPU_TOLERANCES[name],
    )


def _fixed_density_xc_gradient(
    functional: ExcFunctionalBase,
    molecule: gto.Mole,
    grid: SkalaGrids,
    density: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    density_tensor = torch.from_numpy(density.copy())
    effective_potential, explicit_gradient = veff_and_expl_nuc_grad(
        functional,
        molecule,
        grid,
        density_tensor,
    )
    contracted_gradient = torch.empty(
        (molecule.natm, 3), dtype=effective_potential.dtype
    )
    for atom_index, (_, _, ao_start, ao_end) in enumerate(molecule.aoslice_by_atom()):
        contracted_gradient[atom_index] = (
            torch.einsum(
                "...xij,...ij->x",
                effective_potential[..., ao_start:ao_end, :],
                density_tensor[..., ao_start:ao_end, :],
            )
            * 2
        )
    return (contracted_gradient + explicit_gradient).numpy()


def test_pyscf_conversion_matches_skalaxc_layout(molecule: gto.Mole) -> None:
    density = _structural_density(molecule)
    skalaxc_molecule, skalaxc_basis = pyscf_to_skalaxc(molecule)
    assert skalaxc_molecule.natoms == molecule.natm
    assert skalaxc_basis.nbf == molecule.nao_nr()

    scalar_density, spin_density = uks_density_channels(density)
    assert scalar_density.flags.f_contiguous
    assert spin_density.flags.f_contiguous
    np.testing.assert_allclose((scalar_density + spin_density) / 2, density[0])
    np.testing.assert_allclose((scalar_density - spin_density) / 2, density[1])


def test_exc_vxc_parity(
    functional_case: FunctionalCase,
    molecule: gto.Mole,
    density: npt.NDArray[np.float64],
    pyscf_grid: SkalaGrids,
) -> None:
    pyscf_result = XCIntegrator(functional_case.functional)(
        molecule,
        pyscf_grid,
        torch.from_numpy(density.copy()),
    )

    scalar_density, spin_density = uks_density_channels(density)
    energy, scalar_potential, spin_potential = functional_case.integrator.eval_exc_vxc(
        scalar_density, spin_density
    )

    pyscf_potential = pyscf_result.potential.detach().numpy()
    # SkalaXC returns derivatives with respect to Ds = Da + Db and Dz = Da - Db.
    expected_scalar = (pyscf_potential[0] + pyscf_potential[1]) / 2
    expected_spin = (pyscf_potential[0] - pyscf_potential[1]) / 2
    energy_error = abs(energy - pyscf_result.energy.item())
    scalar_error = _matrix_error(scalar_potential, expected_scalar)
    spin_error = _matrix_error(spin_potential, expected_spin)

    assert energy_error < functional_case.tolerances.energy, (
        f"{functional_case.name} XC energy differs by {energy_error:.3e}: "
        f"PySCF={pyscf_result.energy.item():.15g}, SkalaXC={energy:.15g}"
    )
    assert scalar_error < functional_case.tolerances.scalar_potential, (
        f"{functional_case.name} scalar VXC error per basis function: {scalar_error:.3e}"
    )
    assert spin_error < functional_case.tolerances.spin_potential, (
        f"{functional_case.name} spin-z VXC error per basis function: {spin_error:.3e}"
    )


def test_exc_gradient_parity(
    functional_case: FunctionalCase,
    molecule: gto.Mole,
    density: npt.NDArray[np.float64],
    pyscf_grid: SkalaGrids,
) -> None:
    pyscf_gradient = _fixed_density_xc_gradient(
        functional_case.functional,
        molecule,
        pyscf_grid,
        density,
    )

    scalar_density, spin_density = uks_density_channels(density)
    skalaxc_gradient = functional_case.integrator.eval_exc_grad(
        scalar_density, spin_density
    )
    max_error = float(np.max(np.abs(skalaxc_gradient - pyscf_gradient)))

    assert max_error < functional_case.tolerances.gradient, (
        f"{functional_case.name} XC gradient max error: {max_error:.3e}\n"
        f"PySCF:\n{pyscf_gradient}\nSkalaXC:\n{skalaxc_gradient}"
    )
