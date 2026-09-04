# SPDX-License-Identifier: MIT

"""Fixed-density parity tests between GPU4PySCF Skala and SkalaXC CUDA."""

import hashlib
from collections.abc import Generator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest
import skalaxc
import torch
from skala.functional import load_functional
from skala.functional.base import ExcFunctionalBase

from pyscf import dft, gto, scf
from tests.skalaxc_test_utils import (
    SKALA_1_1_REV1_CUDA_SHA256,
    ParityTolerances,
    make_skalaxc_integrator,
    uks_density_channels,
)

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("CUDA is not available", allow_module_level=True)
if not skalaxc.CUDA_ENABLED:
    pytest.skip("SkalaXC was built without CUDA", allow_module_level=True)
pytest.importorskip("cupy", reason="CuPy is not available")
pytest.importorskip("gpu4pyscf", reason="GPU4PySCF is not available")
DEVICE_EXECUTION_SPACE = skalaxc.ExecutionSpace.DEVICE

from skala.gpu4pyscf.gradients import (  # noqa: E402
    nuc_grad_from_veff,
    veff_and_expl_nuc_grad,
)
from skala.gpu4pyscf.grids import SkalaGrids  # noqa: E402
from skala.pyscf.xc_integrator import XCIntegrator  # noqa: E402


@dataclass(frozen=True)
class FunctionalCase:
    name: str
    functional: ExcFunctionalBase
    model: str
    tolerances: ParityTolerances


GPU_TOLERANCES = {
    "lda": ParityTolerances(1e-7, 1.5e-8, 2e-9, 7.5e-6),
    "pbe": ParityTolerances(1e-7, 1.5e-8, 3e-9, 7.5e-6),
    "tpss": ParityTolerances(1e-7, 4e-8, 1.5e-8, 7.5e-6),
    "skala-1.1": ParityTolerances(7.5e-5, 2.5e-4, 2.5e-4, 1e-4),
}
FUNCTIONAL_NAMES = ("lda", "pbe", "tpss", "skala-1.1")
# The current TPSS TorchScript trace can exceed sm_120 launch resources during
# backward. Keep its EXC/VXC coverage and use neural Skala for the primary
# kinetic-density gradient path until TPSS is retraced with a smaller kernel.
GRADIENT_FUNCTIONAL_NAMES = ("lda", "pbe", "skala-1.1")


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
def gpu4pyscf_grid(molecule: gto.Mole) -> SkalaGrids:
    grid = SkalaGrids(molecule)
    grid.level = 5
    grid.prune = None
    grid.radi_method = dft.radi.mura_knowles
    grid.build()
    return grid


@pytest.fixture
def functional_case(
    request: pytest.FixtureRequest,
) -> FunctionalCase:
    name = str(request.param)
    device = torch.device("cuda:0")
    loaded_functional = load_functional(name, device=device)
    assert isinstance(loaded_functional, ExcFunctionalBase)
    if name == "skala-1.1":
        model_path = skalaxc.MODEL_DIR / "skala-1.1-cuda.fun"
        with model_path.open("rb") as model_file:
            digest = hashlib.file_digest(model_file, "sha256").hexdigest()
        assert digest == SKALA_1_1_REV1_CUDA_SHA256
        model = str(model_path)
    else:
        model = name.upper()
    return FunctionalCase(
        name,
        loaded_functional,
        model,
        GPU_TOLERANCES[name],
    )


def _matrix_error(
    actual: npt.NDArray[np.float64], expected: npt.NDArray[np.float64]
) -> float:
    return float(np.linalg.norm(actual - expected) / actual.shape[0])


def _fixed_density_xc_gradient(
    functional: ExcFunctionalBase,
    molecule: gto.Mole,
    grid: SkalaGrids,
    density: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    density_tensor = torch.as_tensor(density, device="cuda:0")
    effective_potential, explicit_gradient = veff_and_expl_nuc_grad(
        functional,
        molecule,
        grid,
        density_tensor,
    )
    contracted_gradient = 2 * nuc_grad_from_veff(
        molecule,
        effective_potential,
        density_tensor,
    )
    return (contracted_gradient + explicit_gradient).detach().cpu().numpy()


def _release_torch_cache() -> None:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def release_torch_cache_after_test() -> Generator[None, None, None]:
    """Release unused CUDA allocator blocks between parameterized cases."""
    yield
    _release_torch_cache()


@pytest.mark.parametrize("functional_case", FUNCTIONAL_NAMES, indirect=True)
def test_gpu_exc_vxc_parity(
    functional_case: FunctionalCase,
    molecule: gto.Mole,
    density: npt.NDArray[np.float64],
    gpu4pyscf_grid: SkalaGrids,
) -> None:
    density_tensor = torch.as_tensor(density, device="cuda:0")
    gpu4pyscf_result = XCIntegrator(
        functional_case.functional,
        device=torch.device("cuda:0"),
    )(molecule, gpu4pyscf_grid, density_tensor)
    gpu4pyscf_energy = gpu4pyscf_result.energy.item()
    gpu4pyscf_potential = gpu4pyscf_result.potential.detach().cpu().numpy()
    del gpu4pyscf_result, density_tensor
    _release_torch_cache()

    scalar_density, spin_density = uks_density_channels(density)
    integrator = make_skalaxc_integrator(
        molecule,
        functional_case.model,
        execution_space=DEVICE_EXECUTION_SPACE,
    )
    energy, scalar_potential, spin_potential = integrator.eval_exc_vxc(
        scalar_density,
        spin_density,
    )

    expected_scalar = (gpu4pyscf_potential[0] + gpu4pyscf_potential[1]) / 2
    expected_spin = (gpu4pyscf_potential[0] - gpu4pyscf_potential[1]) / 2
    energy_error = abs(energy - gpu4pyscf_energy)
    scalar_error = _matrix_error(scalar_potential, expected_scalar)
    spin_error = _matrix_error(spin_potential, expected_spin)

    assert energy_error < functional_case.tolerances.energy, (
        f"{functional_case.name} GPU XC energy error: {energy_error:.3e}"
    )
    assert scalar_error < functional_case.tolerances.scalar_potential, (
        f"{functional_case.name} GPU scalar VXC error per basis function: "
        f"{scalar_error:.3e}"
    )
    assert spin_error < functional_case.tolerances.spin_potential, (
        f"{functional_case.name} GPU spin-z VXC error per basis function: "
        f"{spin_error:.3e}"
    )


@pytest.mark.parametrize("functional_case", GRADIENT_FUNCTIONAL_NAMES, indirect=True)
def test_gpu_exc_gradient_parity(
    functional_case: FunctionalCase,
    molecule: gto.Mole,
    density: npt.NDArray[np.float64],
    gpu4pyscf_grid: SkalaGrids,
) -> None:
    gpu4pyscf_gradient = _fixed_density_xc_gradient(
        functional_case.functional,
        molecule,
        gpu4pyscf_grid,
        density,
    )
    _release_torch_cache()

    scalar_density, spin_density = uks_density_channels(density)
    integrator = make_skalaxc_integrator(
        molecule,
        functional_case.model,
        execution_space=DEVICE_EXECUTION_SPACE,
    )
    skalaxc_gradient = integrator.eval_exc_grad(scalar_density, spin_density)
    max_error = float(np.max(np.abs(skalaxc_gradient - gpu4pyscf_gradient)))

    assert max_error < functional_case.tolerances.gradient, (
        f"{functional_case.name} GPU XC gradient max error: {max_error:.3e}\n"
        f"GPU4PySCF:\n{gpu4pyscf_gradient}\nSkalaXC:\n{skalaxc_gradient}"
    )
