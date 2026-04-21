from collections.abc import Callable

import pytest
import torch
from pyscf import gto

if not torch.cuda.is_available():
    pytest.skip(
        "Skipping gpu4pyscf gradients tests, because CUDA is not available.",
        allow_module_level=True,
    )

from gpu4pyscf import dft, scf

from skala.functional import load_functional
from skala.functional.base import ExcFunctionalBase
from skala.gpu4pyscf import SkalaKS
from skala.gpu4pyscf.gradients import (
    SkalaRKSGradient,
    SkalaUKSGradient,
    nuc_grad_from_veff,
    veff_and_expl_nuc_grad,
)
from skala.pyscf.features import generate_features


def num_dif_ridders(
    func: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    initial_step: float = 0.01,
    step_div: float = 1.414,
    max_tab: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Numerical derivative via extrapolation, so this is expensive, but will be accurate, so good for testing.
    The second return value is the estimate error. If this is large, try reducing the initial_step.

    func:         function to differentiate
    x:            position where to evaluate the derivative
    initial_step: initial step size
    max_tab:      amount of different steps tried
    step_div:     amount by which the step is divided
    """
    d_estimate = torch.empty((max_tab, max_tab), dtype=x.dtype, device=x.device)

    step = initial_step
    step_div_2 = step_div**2
    err = torch.tensor(torch.finfo(x.dtype).max, dtype=x.dtype, device=x.device)
    prev_err = err

    d_estimate[0, 0] = (func(x + step) - func(x - step)) / (2 * step)
    prev_deriv = d_estimate[0, 0]
    num_deriv = prev_deriv
    for iter in range(1, max_tab):
        step /= step_div
        d_estimate[iter, 0] = (func(x + step) - func(x - step)) / (2 * step)
        # use this new central difference estimate to eliminate next leading errors from previous estimates
        factor = step_div_2
        for order in range(iter):
            # each step in order eliminates the term of order ~ step**(2order)
            factor *= step_div_2
            d_estimate[iter, order + 1] = (
                factor * d_estimate[iter, order] - d_estimate[iter - 1, order]
            ) / (factor - 1.0)
            # estimate error as the max difference w.r.t. the two lower order options
            err_est = torch.max(
                torch.abs(d_estimate[iter, order + 1] - d_estimate[iter, order]),
                torch.abs(d_estimate[iter, order + 1] - d_estimate[iter - 1, order]),
            )
            if err_est <= err:
                err = err_est
                num_deriv = d_estimate[iter, order + 1]

        if (
            torch.abs(d_estimate[iter, iter] - d_estimate[iter - 1, iter - 1])
            >= 2 * err
            and iter > 1
        ):
            # subtracting different step sizes does not work anymore to reduce error
            # suspect last step-size is too small, so don't trust -> stop and return previous best
            return prev_deriv, prev_err

        prev_deriv = num_deriv
        prev_err = err

    return num_deriv, err


def num_grad_ridders(
    func: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    initial_step: float = 0.01,
    step_div: float = 1.414,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recursively calculates the partial derivative w.r.t. all elements of x over all dimensions."""

    def func_1d_red(xi: torch.Tensor) -> torch.Tensor:
        x_ = x.clone()
        x_[i] = xi
        return func(x_)

    grad = torch.empty_like(x)
    err = torch.empty_like(x)

    if len(x.size()) == 1:
        for i, xi in enumerate(x):
            grad[i], err[i] = num_dif_ridders(
                func_1d_red, xi, initial_step=initial_step, step_div=step_div
            )
    else:
        for i, xi in enumerate(x):
            grad[i], err[i] = num_grad_ridders(
                func_1d_red, xi, initial_step=initial_step, step_div=step_div
            )

    return grad, err


@pytest.fixture(params=["HF", "H2O", "H2O+"])
def mol_name(request: pytest.FixtureRequest) -> str:
    return request.param


def get_mol(molname: str) -> gto.Mole:
    if molname == "HF":
        mol = gto.M(atom="H 0 0 0; F 0 0 1.1", basis="def2-qzvp", cart=True)
    elif molname == "H2O":
        mol = gto.M(
            atom="O 0 0 0; H 0.758602  0.000000  0.504284; H 0.758602  0.000000  -0.504284",
            basis="def2-qzvp",
        )
    elif molname == "H2O+":
        mol = gto.M(
            atom="O 0 0 0; H 0.758602  0.000000  0.504284; H 0.758602  0.000000  -0.504284",
            basis="def2-tzvp",
            charge=1,
            spin=1,
        )
    else:
        raise ValueError(f"Unknown molecule {molname}")

    return mol


def minimal_grid(mol: gto.Mole) -> dft.Grids:
    return dft.Grids(mol)(level=1, radi_method=dft.radi.treutler).build()


def get_grid_and_rdm1(mol: gto.Mole) -> tuple[dft.Grids, torch.Tensor]:
    mf = dft.KS(
        mol,
        xc="pbe",
    )(
        grids=minimal_grid(mol),
    )
    mf.kernel()
    rdm1 = torch.from_dlpack(mf.make_rdm1())  # type: ignore[attr-defined]
    return mf.grids, rdm1  # maybe_expand_and_divide(rdm1, len(rdm1.shape) == 2, 2)


def test_grid_coords_gradient(mol_name: str) -> None:
    class TestFunc(ExcFunctionalBase):
        def __init__(self) -> None:
            super().__init__()
            self.features = ["grid_coords"]

        def get_exc(self, mol: dict[str, torch.Tensor]) -> torch.Tensor:
            """This actually calculates the total electron number"""
            return mol["grid_coords"].sum()

    mol = get_mol(mol_name)
    grid, rdm1 = get_grid_and_rdm1(mol)
    exc_test = TestFunc()
    ana_grad = veff_and_expl_nuc_grad(exc_test, mol, grid, rdm1)[1]

    # calculate exact result
    atom_grids_tab = grid.gen_atomic_grids(
        mol, grid.atom_grid, grid.radi_method, grid.level, grid.prune
    )
    exact_grad = torch.empty_like(ana_grad)
    for iatm in range(mol.natm):
        n_atom_grid_points = atom_grids_tab[mol.atom_symbol(iatm)][0].shape[0]
        exact_grad[iatm] = n_atom_grid_points * torch.ones(3)

    assert torch.allclose(ana_grad, exact_grad, rtol=1e-15, atol=0.0)


def test_coarse_0_atomic_coords_gradient(mol_name: str) -> None:
    class TestFunc(ExcFunctionalBase):
        def __init__(self) -> None:
            super().__init__()
            self.features = ["coarse_0_atomic_coords"]

        def get_exc(self, mol: dict[str, torch.Tensor]) -> torch.Tensor:
            """This actually calculates the total electron number"""
            return torch.einsum("nx->", mol["coarse_0_atomic_coords"])

    mol = get_mol(mol_name)
    grid, rdm1 = get_grid_and_rdm1(mol)
    exc_test = TestFunc()
    ana_grad = veff_and_expl_nuc_grad(exc_test, mol, grid, rdm1)[1]

    # calculate exact result
    exact_grad = torch.ones_like(ana_grad)

    assert torch.allclose(ana_grad, exact_grad, rtol=1e-15, atol=0.0)


def test_grid_weights_gradient(mol_name: str) -> None:
    mol = get_mol(mol_name)

    class TestFunc(ExcFunctionalBase):
        def __init__(self) -> None:
            super().__init__()
            self.features = ["grid_weights"]

        def get_exc(self, mol: dict[str, torch.Tensor]) -> torch.Tensor:
            """This actually calculates the total electron number"""
            return mol["grid_weights"].sum()

    def finite_difference_nuc_grad(
        weight_sum: ExcFunctionalBase, mol: gto.Mole, rdm1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculates the gradient in Exc w.r.t. nuclear coordinates numerically"""
        # mol_.verbose = 2
        mol_feats = generate_features(
            mol, rdm1, minimal_grid(mol), set(weight_sum.features), gpu=True
        )

        def weight_sum_as_nuc_coords_func(nuc_coords: torch.Tensor) -> torch.Tensor:
            """Exc wrapper for the finite difference"""
            mol.set_geom_(nuc_coords.cpu().numpy(), "bohr", symmetry=None)
            mol_feats["grid_weights"] = torch.from_dlpack(minimal_grid(mol).weights)  # type: ignore[attr-defined]

            return weight_sum.get_exc(mol_feats)

        nuc_coords = torch.tensor(mol.atom_coords(), device=rdm1.device)
        return num_grad_ridders(weight_sum_as_nuc_coords_func, nuc_coords)

    grid, rdm1 = get_grid_and_rdm1(mol)
    exc_test = TestFunc()
    ana_grad = veff_and_expl_nuc_grad(exc_test, mol, grid, rdm1)[1]

    # calculate numerical derivative as accurate as possible
    num_grad, num_err = finite_difference_nuc_grad(exc_test, mol, rdm1)
    # estimate the minimum expected absolute error
    eps = (
        exc_test.get_exc({"grid_weights": torch.from_dlpack(grid.weights)})  # type: ignore[attr-defined]
        * torch.finfo(num_grad.dtype).eps
    )

    check_mat = (ana_grad - num_grad).abs() <= torch.max(128 * num_err, 128 * eps)

    print(f"{num_err = }")
    print(f"{ana_grad - num_grad = }")
    print(f"{check_mat = }")

    assert torch.all(check_mat)


def test_density_veff(mol_name: str) -> None:
    mol = get_mol(mol_name)

    class TestFunc(ExcFunctionalBase):
        def __init__(self) -> None:
            super().__init__()
            self.features = ["density", "grid_weights"]

        def get_exc(self, mol: dict[str, torch.Tensor]) -> torch.Tensor:
            """This actually calculates the total electron number"""
            return (mol["density"] @ mol["grid_weights"]).sum()

    def finite_difference_nuc_grad(
        dens_sum: ExcFunctionalBase, mol: gto.Mole, rdm1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculates the gradient in Exc w.r.t. nuclear coordinates numerically"""

        grid = minimal_grid(mol)
        mol_ = mol.copy()

        def dens_sum_as_nuc_coords_func(nuc_coords: torch.Tensor) -> torch.Tensor:
            """Exc wrapper for the finite difference"""
            mol_.set_geom_(nuc_coords.cpu().numpy(), "bohr", symmetry=None)
            mol_feats = generate_features(
                mol_, rdm1, grid, set(dens_sum.features), gpu=True
            )

            return dens_sum.get_exc(mol_feats)

        nuc_coords = torch.tensor(mol.atom_coords(), device=rdm1.device)
        return num_grad_ridders(dens_sum_as_nuc_coords_func, nuc_coords)

    grid, rdm1 = get_grid_and_rdm1(mol)
    exc_test = TestFunc()

    # calculate numerical gradient via density dependence
    num_grad, num_err = finite_difference_nuc_grad(exc_test, mol, rdm1)

    # calculate analytic result
    veff = veff_and_expl_nuc_grad(
        exc_test, mol, grid, rdm1, nuc_grad_feats={"density"}
    )[0]
    ana_grad = 2 * nuc_grad_from_veff(mol, veff, rdm1)

    check_mat = (ana_grad - num_grad).abs() <= torch.max(
        2**12 * num_err, torch.tensor(torch.finfo(num_grad.dtype).eps * 2**11)
    )

    print(f"{num_err = }")
    print(f"{ana_grad - num_grad = }")
    print(f"{check_mat = }")

    assert torch.all(check_mat)


def test_grad_veff(mol_name: str) -> None:
    mol = get_mol(mol_name)

    class TestFunc(ExcFunctionalBase):
        def __init__(self) -> None:
            super().__init__()
            self.features = ["grad", "grid_weights"]

        def get_exc(self, mol: dict[str, torch.Tensor]) -> torch.Tensor:
            return (
                (mol["grad"] ** 2 @ mol["grid_weights"])
                @ torch.tensor(
                    [1.0, 2.0, 3.0],
                    dtype=torch.float64,
                    device=mol["grad"].device,
                )
            ).sum()

    def finite_difference_nuc_grad(
        grad_func: ExcFunctionalBase, mol: gto.Mole, rdm1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculates the gradient in Exc w.r.t. nuclear coordinates numerically"""

        grid = minimal_grid(mol)
        mol_ = mol.copy()

        def grad_func_as_nuc_coords_func(nuc_coords: torch.Tensor) -> torch.Tensor:
            """Exc wrapper for the finite difference"""
            mol_.set_geom_(nuc_coords.cpu().numpy(), "bohr", symmetry=None)
            mol_feats = generate_features(
                mol_, rdm1, grid, set(grad_func.features), gpu=True
            )

            return grad_func.get_exc(mol_feats)

        nuc_coords = torch.tensor(mol.atom_coords(), device=rdm1.device)
        print(f"{grad_func_as_nuc_coords_func(nuc_coords).item() = :.15e}")
        return num_grad_ridders(grad_func_as_nuc_coords_func, nuc_coords)

    grid, rdm1 = get_grid_and_rdm1(mol)
    exc_test = TestFunc()

    # calculate numerical result
    num_grad, num_err = finite_difference_nuc_grad(exc_test, mol, rdm1)

    # calculate analytic result
    veff = veff_and_expl_nuc_grad(exc_test, mol, grid, rdm1, nuc_grad_feats={"grad"})[0]
    ana_grad = 2 * nuc_grad_from_veff(mol, veff, rdm1)

    check_mat = (ana_grad - num_grad).abs() <= torch.max(
        2**11 * num_err, torch.tensor(torch.finfo(num_grad.dtype).eps * 2**21)
    )

    print(f"{num_err = }")
    print(f"{ana_grad - num_grad = }")
    print(f"{check_mat = }")

    assert torch.all(check_mat)


def test_kin_veff(mol_name: str) -> None:
    mol = get_mol(mol_name)

    class TestFunc(ExcFunctionalBase):
        def __init__(self) -> None:
            super().__init__()
            self.features = ["kin", "grid_weights"]

        def get_exc(self, mol: dict[str, torch.Tensor]) -> torch.Tensor:
            """This actually calculates the total kinetic energy number"""
            return (mol["kin"] @ mol["grid_weights"]).sum()

    def finite_difference_nuc_grad(
        kin_func: ExcFunctionalBase, mol: gto.Mole, rdm1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculates the gradient in Exc w.r.t. nuclear coordinates numerically"""

        grid = minimal_grid(mol)
        mol_ = mol.copy()

        def kin_func_as_nuc_coords_func(nuc_coords: torch.Tensor) -> torch.Tensor:
            """Exc wrapper for the finite difference"""
            mol_.set_geom_(nuc_coords.cpu().numpy(), "bohr", symmetry=None)
            mol_feats = generate_features(
                mol_, rdm1, grid, set(kin_func.features), gpu=True
            )

            return kin_func.get_exc(mol_feats)

        nuc_coords = torch.tensor(mol.atom_coords(), device=rdm1.device)
        print(f"{kin_func_as_nuc_coords_func(nuc_coords).item() = :.15e}")
        return num_grad_ridders(kin_func_as_nuc_coords_func, nuc_coords)

    grid, rdm1 = get_grid_and_rdm1(mol)
    exc_test = TestFunc()

    # calculate numerical result
    num_grad, num_err = finite_difference_nuc_grad(exc_test, mol, rdm1)

    # calculate analytic result
    veff = veff_and_expl_nuc_grad(exc_test, mol, grid, rdm1, nuc_grad_feats={"kin"})[0]
    ana_grad = 2 * nuc_grad_from_veff(mol, veff, rdm1)

    check_mat = (ana_grad - num_grad).abs() <= torch.max(
        32 * num_err, torch.tensor(torch.finfo(num_grad.dtype).eps * 10**14)
    )

    print(f"{num_err = }")
    print(f"{ana_grad - num_grad = }")
    print(f"{check_mat = }")

    assert torch.all(check_mat)


def run_scf(
    mol: gto.Mole,
    functional: ExcFunctionalBase,
    with_dftd3: bool,
    *,
    grid_level: int = 1,
) -> scf.hf.SCF:
    print(f"{mol.basis = }")
    scf = SkalaKS(mol, xc=functional, with_dftd3=with_dftd3)
    scf.grids.level = grid_level
    scf.conv_tol = 1e-14
    scf.kernel()

    return scf


@pytest.fixture(
    params=[
        "pbe",
        "skala-1.0",
        "skala-1.1",
    ]
)
def xc_name(request: pytest.FixtureRequest) -> str:
    return request.param


def mol_min_bas(molname: str) -> gto.Mole:
    molecule = get_mol(molname)
    molecule.basis = "sto-3g"

    return molecule


FULL_GRAD_REF = {
    "HF:pbe": torch.tensor(
        [[0.0, 0.0, -1.0283181338840031e-01], [0.0, 0.0, 1.0283181338840475e-01]],
        dtype=torch.float64,
    ),
    "H2O:pbe": torch.tensor(
        [
            [7.3868922411540083e-02, 0.0, 0.0],
            [-3.6934461205758495e-02, 0.0, -1.3005275018782658e-01],
            [-3.6934461205764268e-02, 0.0, 1.3005275018783147e-01],
        ],
        dtype=torch.float64,
    ),
    "H2O+:pbe": torch.tensor(
        [
            [1.3766133501961964e-01, 0.0, 0.0],
            [-6.8830667509800936e-02, 0.0, -1.6302458647600626e-01],
            [-6.8830667509806709e-02, 0.0, 1.6302458647600737e-01],
        ],
        dtype=torch.float64,
    ),
    "HF:skala-1.0": torch.tensor(
        [
            [-6.613324869518045e-11, 8.252792463549002e-11, -0.11766455093793571],
            [6.61332489653549e-11, -8.252792460791792e-11, 0.11766455093858674],
        ],
        dtype=torch.float64,
    ),
    "H2O:skala-1.0": torch.tensor(
        [
            [0.047614265311160864, -1.531395405982722e-10, -3.8752502138778113e-10],
            [-0.023807132833466138, 1.4666252996608134e-10, -0.1265627683625854],
            [-0.02380713247717514, 6.477008760972603e-12, 0.12656276875009254],
        ],
        dtype=torch.float64,
    ),
    "H2O+:skala-1.0": torch.tensor(
        [
            [0.11016449231802916, -1.4692817466244152e-10, 8.447353527447336e-10],
            [-0.05508224759855285, 4.2209029890734686e-10, -0.15564538047513543],
            [-0.05508224472116563, -2.751621283378772e-10, 0.15564537963040848],
        ],
        dtype=torch.float64,
    ),
    "HF:skala-1.1": torch.tensor(
        [
            [-9.093651147681359e-11, 1.8436550945505342e-10, -0.11922130029704636],
            [9.093651147684337e-11, -1.8436550945509882e-10, 0.11922130029705125],
        ],
        dtype=torch.float64,
    ),
    "H2O:skala-1.1": torch.tensor(
        [
            [0.05518685428627901, 1.0112539782960166e-09, 5.727682266064312e-10],
            [-0.027593427632945478, -4.887476070628282e-06, -0.12591870031741337],
            [-0.027593426653364173, 4.886464816658505e-06, 0.12591869974465286],
        ],
        dtype=torch.float64,
    ),
    "H2O+:skala-1.1": torch.tensor(
        [
            [0.11201511304824052, -2.33353601498583e-10, 4.162082128268623e-10],
            [-0.05600755684684611, -1.4123079854089175e-06, -0.15729176960843216],
            [-0.056007556201392195, 1.4125413389947007e-06, 0.15729176919222287],
        ],
        dtype=torch.float64,
    ),
}


def test_full_grad(mol_name: str, xc_name: str) -> None:
    # analytical result
    mol = get_mol(mol_name)
    func = load_functional(xc_name, device=torch.device("cuda:0"))
    assert isinstance(func, ExcFunctionalBase)
    # skala-1.1 uses per-atom packed grids (unsorted) and needs a denser grid
    # to avoid NaNs in the SCF.
    scf = run_scf(mol, func, with_dftd3=False)

    if mol.spin == 0:
        grad = SkalaRKSGradient(scf).kernel()
    else:
        grad = SkalaUKSGradient(scf).kernel()
    ana_grad = torch.from_numpy(grad)

    # get reference result
    ref_grad = FULL_GRAD_REF[mol_name + ":" + xc_name]

    assert torch.allclose(ana_grad, ref_grad, atol=1e-4), (
        f"Gradients for {mol_name} with {xc_name} do not match reference.\n"
        f"Analytic: {ana_grad.tolist()!r}\n"
        f"Reference: {ref_grad}\n"
        f"Difference: {ana_grad - ref_grad}"
    )
