from collections.abc import Callable

import pytest
import torch

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip(
        "Skipping gpu4pyscf gradients tests, because CUDA is not available.",
        allow_module_level=True,
    )
try:
    import cupy
except ModuleNotFoundError:
    pytest.skip(
        "Skipping gpu4pyscf gradients tests, because CuPy is not available.",
        allow_module_level=True,
    )

from _ridders import num_grad_ridders  # noqa: E402
from gpu4pyscf import dft, scf  # noqa: E402
from pyscf import gto  # noqa: E402
from pyscf.dft import numint as pyscf_numint  # noqa: E402
from test_pyscf_gradients import FULL_GRAD_REF  # noqa: E402

from skala.functional.base import ExcFunctionalBase  # noqa: E402
from skala.gpu4pyscf import SkalaKS  # noqa: E402
from skala.gpu4pyscf.gradients import (  # noqa: E402
    SkalaRKSGradient,
    SkalaUKSGradient,
    nuc_grad_from_veff,
    veff_and_expl_nuc_grad,
)
from skala.pyscf import SkalaKS as CpuSkalaKS  # noqa: E402
from skala.pyscf.features import generate_features  # noqa: E402
from skala.pyscf.gradients import SkalaRKSGradient as CpuSkalaRKSGradient  # noqa: E402
from skala.pyscf.numint import _should_screen_aos  # noqa: E402
from skala.utils import torch_allocator  # noqa: E402

H2_SKALA_1_1_GRAD_REF = torch.tensor(
    [
        [
            2.6170957571276746e-10,
            2.1217813541405875e-10,
            -1.345246115431109e-02,
        ],
        [
            -2.6170957571276746e-10,
            -2.1217813541405844e-10,
            1.3452461154311535e-02,
        ],
    ],
    dtype=torch.float64,
)


def test_torch_allocator_is_active_after_import() -> None:
    assert torch_allocator._allocator is not None
    assert (
        getattr(cupy.cuda.get_allocator(), "__self__", None)
        is torch_allocator._allocator
    )


def test_torch_allocator_uses_shared_non_default_stream() -> None:
    cupy_stream = cupy.cuda.Stream(non_blocking=True)
    torch_stream = torch.cuda.ExternalStream(  # type: ignore[no-untyped-call]
        cupy_stream.ptr,
        device=cupy_stream.device_id,
    )

    with torch.cuda.stream(torch_stream), cupy_stream:
        torch_allocator.use_torch_mempool_in_cupy()
        array = cupy.empty(1)

    assert array.device.id == torch_stream.device.index


def test_torch_allocator_rejects_mismatched_streams() -> None:
    torch_stream = torch.cuda.Stream()  # type: ignore[no-untyped-call]

    with torch.cuda.stream(torch_stream), cupy.cuda.Stream.null:
        torch_allocator.use_torch_mempool_in_cupy()
        with pytest.raises(RuntimeError, match="must be same"):
            cupy.empty(1)


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


def test_full_grad(
    mol_name: str,
    xc_name: str,
    load_functional_cached: Callable[..., ExcFunctionalBase | str],
) -> None:
    # analytical result
    mol = get_mol(mol_name)
    func = load_functional_cached(xc_name, device=torch.device("cuda:0"))
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


def test_nuclear_gradient_cpu_gpu_dense_screened_agree(
    monkeypatch: pytest.MonkeyPatch,
    load_functional_cached: Callable[..., ExcFunctionalBase | str],
) -> None:
    """Compare complete nuclear gradients across backend and SCF screening routes.

    AO screening controls the SCF feature/Vxc path that produces the converged
    density. The analytic nuclear-gradient contraction then uses the same atom-major
    implementation for the dense and screened densities on each backend.
    """
    cpu_functional = load_functional_cached("skala-1.1")
    gpu_functional = load_functional_cached("skala-1.1", device=torch.device("cuda:0"))
    assert isinstance(cpu_functional, ExcFunctionalBase)
    assert isinstance(gpu_functional, ExcFunctionalBase)

    gradients: dict[str, torch.Tensor] = {}
    for backend, functional in (
        ("cpu", cpu_functional),
        ("gpu", gpu_functional),
    ):
        for screened in (False, True):
            mol = gto.M(
                atom="H 0 0 0; H 0 0 0.74",
                basis="sto-3g",
                verbose=0,
            )
            monkeypatch.setattr(
                pyscf_numint,
                "SWITCH_SIZE",
                mol.nao_nr() - int(screened),
            )
            assert _should_screen_aos(mol) is screened
            if backend == "cpu":
                mean_field = CpuSkalaKS(mol, xc=functional, with_dftd3=False)
                gradient_type = CpuSkalaRKSGradient
            else:
                mean_field = SkalaKS(mol, xc=functional, with_dftd3=False)
                gradient_type = SkalaRKSGradient
            mean_field.grids.level = 0
            mean_field.grids.build(mol, sort_grids=False)
            mean_field.conv_tol = 1e-10
            mean_field.kernel()
            assert mean_field.converged
            gradient = gradient_type(mean_field).kernel()
            route = f"{backend}-{'screened' if screened else 'dense'}"
            gradients[route] = torch.from_numpy(gradient)

    for route, gradient in gradients.items():
        torch.testing.assert_close(
            gradient,
            H2_SKALA_1_1_GRAD_REF,
            rtol=1e-7,
            atol=1e-8,
            msg=f"{route} does not match the stored nuclear-gradient reference",
        )

    cpu_dense = gradients["cpu-dense"]
    for route, gradient in gradients.items():
        torch.testing.assert_close(
            gradient,
            cpu_dense,
            rtol=1e-7,
            atol=1e-8,
            msg=f"{route} does not match cpu-dense",
        )


def test_cuda_kernel_memory_stability() -> None:
    """Checks that repeated calls do not increase Torch's allocated CUDA memory."""

    mol = mol_min_bas("HF")
    grid, rdm1 = get_grid_and_rdm1(mol)

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

    exc_test = TestFunc()

    # Warmup to avoid counting one-time allocations from CUDA runtime/libraries.
    for _ in range(2):
        veff = veff_and_expl_nuc_grad(
            exc_test, mol, grid, rdm1, nuc_grad_feats={"grad"}
        )[0]
        _ = 2 * nuc_grad_from_veff(mol, veff, rdm1)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    allocations: list[int] = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(5):
        veff = veff_and_expl_nuc_grad(
            exc_test, mol, grid, rdm1, nuc_grad_feats={"grad"}
        )[0]
        _ = 2 * nuc_grad_from_veff(mol, veff, rdm1)
        torch.cuda.synchronize()
        allocations.append(torch.cuda.memory_allocated())

    max_growth_bytes = max(allocations) - min(allocations)
    assert max_growth_bytes < 16 * 1024**2, (
        "CUDA kernel memory use appears unstable across repeated calls. "
        f"Observed growth: {max_growth_bytes / 1024**2:.2f} MiB"
    )
