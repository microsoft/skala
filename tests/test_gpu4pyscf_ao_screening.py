from collections.abc import Callable, Iterator
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pyscf import dft, gto
from pyscf.dft import numint as pyscf_numint
from torch.utils.dlpack import from_dlpack

if not torch.cuda.is_available():
    pytest.skip(
        "Skipping gpu4pyscf AO screening tests, because CUDA is not available.",
        allow_module_level=True,
    )

try:
    import cupy
except ModuleNotFoundError:
    pytest.skip(
        "Skipping gpu4pyscf AO screening tests, because CuPy is not available.",
        allow_module_level=True,
    )

from skala.functional.base import ExcFunctionalBase
from skala.gpu4pyscf import SkalaKS
from skala.pyscf.backend import dft_gpu
from skala.pyscf.features import ChunkEvalForward, MGGAFeatureFunction
from skala.pyscf.numint import SkalaNumInt

CARBON_CHAIN = """
C 0.0 0.0 0.0
C 1.4 0.0 0.0
C 2.8 0.0 0.0
C 4.2 0.0 0.0
"""


class QuadraticDensityFunctional(ExcFunctionalBase):
    def __init__(self) -> None:
        super().__init__()
        self.features = ["atomic_grid_sizes", "density", "grid_weights"]

    def get_exc(self, mol: dict[str, torch.Tensor]) -> torch.Tensor:
        return (mol["density"].square() * mol["grid_weights"]).sum()


def _to_numpy(value: object) -> np.ndarray:
    return cupy.asnumpy(value) if isinstance(value, cupy.ndarray) else np.asarray(value)


@pytest.mark.parametrize("unrestricted", [False, True])
def test_gpu_rks_uks_dense_screened_equivalence(
    monkeypatch: pytest.MonkeyPatch,
    load_functional_cached: Callable[..., ExcFunctionalBase | str],
    unrestricted: bool,
) -> None:
    if unrestricted:
        mol = gto.M(atom="H 0 0 0", basis="sto-3g", spin=1, verbose=0)
    else:
        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", spin=0, verbose=0)

    functional = load_functional_cached("skala-1.1", device=torch.device("cuda:0"))
    assert isinstance(functional, ExcFunctionalBase)
    ks = SkalaKS(mol, xc=functional, with_dftd3=False)
    ks.grids.level = 0
    ks.grids.alignment = 1
    ks.grids.build(sort_grids=False)
    dm = ks.get_init_guess()

    monkeypatch.setattr(pyscf_numint, "SWITCH_SIZE", mol.nao_nr())
    dense = (
        ks._numint.nr_uks(mol, ks.grids, None, dm)
        if unrestricted
        else ks._numint.nr_rks(mol, ks.grids, None, dm)
    )

    monkeypatch.setattr(pyscf_numint, "SWITCH_SIZE", mol.nao_nr() - 1)
    screened = (
        ks._numint.nr_uks(mol, ks.grids, None, dm)
        if unrestricted
        else ks._numint.nr_rks(mol, ks.grids, None, dm)
    )

    assert np.allclose(_to_numpy(dense[0]), _to_numpy(screened[0]), rtol=1e-9)
    assert np.isclose(dense[1], screened[1], rtol=1e-9)
    assert np.allclose(
        _to_numpy(dense[2]), _to_numpy(screened[2]), rtol=1e-8, atol=2e-9
    )


def test_gpu_response_dense_screened_equivalence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", spin=0, verbose=0)
    ks = SkalaKS(mol, xc=QuadraticDensityFunctional(), with_dftd3=False)
    ks.grids.level = 0
    ks.grids.alignment = 1
    ks.grids.build(sort_grids=False)
    mo_coeff = cupy.eye(mol.nao_nr())
    mo_occ = cupy.ones(mol.nao_nr())
    dm1 = cupy.arange(mol.nao_nr() ** 2, dtype=cupy.float64).reshape(
        mol.nao_nr(), mol.nao_nr()
    )
    dm1 += dm1.T

    monkeypatch.setattr(pyscf_numint, "SWITCH_SIZE", mol.nao_nr())
    dense_response = ks._numint.gen_response(mo_coeff, mo_occ, ks=ks)

    monkeypatch.setattr(pyscf_numint, "SWITCH_SIZE", mol.nao_nr() - 1)
    screened_response = ks._numint.gen_response(mo_coeff, mo_occ, ks=ks)

    assert np.allclose(
        _to_numpy(dense_response(dm1)),
        _to_numpy(screened_response(dm1)),
        rtol=1e-9,
        atol=1e-10,
    )


def test_gpu_screened_skala_matches_cpu_on_carbon_chain(
    monkeypatch: pytest.MonkeyPatch,
    load_functional_cached: Callable[..., ExcFunctionalBase | str],
) -> None:
    mol = gto.M(atom=CARBON_CHAIN, basis="def2-qzvpp", verbose=0)
    cpu_grids = dft.Grids(mol)
    cpu_grids.level = 1
    cpu_grids.alignment = 1
    cpu_grids.build(sort_grids=False)
    gpu_grids = dft_gpu.Grids(mol)
    gpu_grids.level = 1
    gpu_grids.alignment = 1
    gpu_grids.build(sort_grids=False)
    dm = dft.RKS(mol).get_init_guess()

    np.testing.assert_allclose(
        cpu_grids.coords,
        cupy.asnumpy(gpu_grids.coords),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        cpu_grids.weights,
        cupy.asnumpy(gpu_grids.weights),
        rtol=1e-12,
        atol=1e-12,
    )
    monkeypatch.setattr(pyscf_numint, "SWITCH_SIZE", mol.nao_nr() - 1)

    cpu_functional = load_functional_cached("skala-1.1", device=torch.device("cpu"))
    gpu_functional = load_functional_cached("skala-1.1", device=torch.device("cuda:0"))
    assert isinstance(cpu_functional, ExcFunctionalBase)
    assert isinstance(gpu_functional, ExcFunctionalBase)
    cpu_result = SkalaNumInt(cpu_functional, device=torch.device("cpu")).nr_rks(
        mol, cpu_grids, None, dm
    )
    gpu_result = SkalaNumInt(gpu_functional, device=torch.device("cuda:0")).nr_rks(
        mol, gpu_grids, None, cupy.asarray(dm)
    )

    gpu_vxc = cupy.asnumpy(gpu_result[2])
    vxc_difference = cpu_result[2] - gpu_vxc
    vxc_max_abs_difference = np.max(np.abs(vxc_difference))
    vxc_relative_l2_difference = np.linalg.norm(vxc_difference) / np.linalg.norm(
        cpu_result[2]
    )
    assert (
        np.isclose(cpu_result[0], gpu_result[0], rtol=1e-10, atol=1e-11)
        and np.isclose(cpu_result[1], gpu_result[1], rtol=1e-10, atol=1e-11)
        and vxc_max_abs_difference < 2e-9
        and vxc_relative_l2_difference < 1e-8
    ), (
        f"N: cpu={cpu_result[0]:.16g}, gpu={gpu_result[0]:.16g}, "
        f"abs_diff={abs(cpu_result[0] - gpu_result[0]):.3e}; "
        f"E_xc: cpu={cpu_result[1]:.16g}, gpu={gpu_result[1]:.16g}, "
        f"abs_diff={abs(cpu_result[1] - gpu_result[1]):.3e}; "
        f"V_xc: max_abs_diff={vxc_max_abs_difference:.3e}, "
        f"relative_l2_diff={vxc_relative_l2_difference:.3e}"
    )


def test_gpu_sparse_mask_sorts_scatters_and_unsorts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mol = gto.M(atom="C 0 0 0", basis="sto-3g", spin=2, verbose=0)
    ngrids = 32
    sort_idx = np.array([2, 0, 4, 1, 3])
    active_sorted_aos = np.array([0, 2, 4])
    ao = cupy.arange(active_sorted_aos.size * ngrids, dtype=cupy.float64).reshape(
        active_sorted_aos.size, ngrids
    )
    weights = cupy.ones(ngrids)
    coords = cupy.zeros((ngrids, 3))
    grids = SimpleNamespace(weights=weights, coords=coords)

    class FakeGpuNumInt:
        def build(self, mol: gto.Mole, coords: cupy.ndarray) -> "FakeGpuNumInt":
            self.gdftopt = SimpleNamespace(_ao_idx=sort_idx)
            return self

        def block_loop(
            self, *args: object, **kwargs: object
        ) -> Iterator[tuple[object, object, object, object]]:
            yield ao, cupy.asarray(active_sorted_aos), weights, coords

    assert dft_gpu is not None
    monkeypatch.setattr(dft_gpu.numint, "NumInt", FakeGpuNumInt)

    feature_function = MGGAFeatureFunction(
        with_density=True, with_grad=False, with_kin=False
    )
    dm = torch.diag(
        torch.arange(1, mol.nao_nr() + 1, dtype=torch.float64, device="cuda")
    ).requires_grad_()
    features = ChunkEvalForward.apply(  # type: ignore[no-untyped-call]
        dm, mol, grids, feature_function, None, False, True
    )

    sort_idx_t = torch.as_tensor(sort_idx, device="cuda")
    active_t = torch.as_tensor(active_sorted_aos, device="cuda")
    dm_sorted = dm[..., sort_idx_t, :][..., sort_idx_t]
    dm_active = dm_sorted[..., active_t[:, None], active_t[None, :]]
    ao_t = from_dlpack(ao)
    expected = torch.sum((dm_active @ ao_t) * ao_t, dim=0).unsqueeze(0)
    assert torch.allclose(features, expected)

    energy = features.square().sum()
    (vxc,) = torch.autograd.grad(energy, dm, create_graph=True)
    (hvp,) = torch.autograd.grad(vxc, dm, torch.ones_like(dm))

    inactive_original_aos = sort_idx[
        np.setdiff1d(np.arange(mol.nao_nr()), active_sorted_aos)
    ]
    assert vxc.shape == dm.shape
    assert hvp.shape == dm.shape
    assert torch.count_nonzero(vxc[inactive_original_aos]) == 0
    assert torch.count_nonzero(vxc[:, inactive_original_aos]) == 0
    assert torch.count_nonzero(hvp[inactive_original_aos]) == 0
    assert torch.count_nonzero(hvp[:, inactive_original_aos]) == 0
