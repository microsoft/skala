from __future__ import annotations

from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from pyscf import dft, gto
from torch.utils.dlpack import from_dlpack

from tests.utils import QuadraticFunctional, force_ao_screening, require_gpu

pytestmark = pytest.mark.gpu

cupy = require_gpu()

from skala.features import Feature  # noqa: E402
from skala.functional.base import ExcFunctionalBase  # noqa: E402
from skala.gpu4pyscf import SkalaKS  # noqa: E402
from skala.gpu4pyscf.grids import SkalaGrids as GPU4PySCFSkalaGrids  # noqa: E402
from skala.pyscf.ao_evaluation import (  # noqa: E402
    evaluate_ao_features_blockwise,
    evaluate_full_grid,
)
from skala.pyscf.backend import Array, dft_gpu  # noqa: E402
from skala.pyscf.evaluation import FeatureSpec  # noqa: E402
from skala.pyscf.feature_math import MGGAFeatureFunction  # noqa: E402
from skala.pyscf.grids import SkalaGrids as PySCFSkalaGrids  # noqa: E402
from skala.pyscf.numint import SkalaNumInt  # noqa: E402
from skala.pyscf.spatial_grid_layout import prepare_spatial_grid_layout  # noqa: E402
from skala.pyscf.xc_integrator import XCIntegrator  # noqa: E402
from skala.typing import D2, F64  # noqa: E402

CARBON_CHAIN = """
C 0.0 0.0 0.0
C 1.4 0.0 0.0
C 2.8 0.0 0.0
C 4.2 0.0 0.0
"""


def test_prepare_spatially_sorted_gpu_grids() -> None:
    mol = gto.M(atom="H 0 0 0", basis="sto-3g", spin=1, verbose=0)
    coords = cupy.asarray(
        [[20.0, 0.0, 0.0], [0.0, 0.0, 0.0], [21.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    weights = cupy.arange(coords.shape[0], dtype=cupy.float64)
    grids = dft_gpu.Grids(mol)
    grids.coords = coords
    grids.weights = weights
    original_screening_cache = cupy.arange(1)
    grids._non0ao_idx = original_screening_cache

    device = torch.device("cuda")
    layout = prepare_spatial_grid_layout(mol, grids, block_size=2, device=device)
    sorted_grids = layout.sorted_grids
    forward = layout.forward_permutation.cpu().numpy()
    inverse = layout.inverse_permutation.cpu().numpy()

    assert sorted_grids is not grids
    assert grids.coords is coords
    assert grids.weights is weights
    assert grids._non0ao_idx is original_screening_cache
    assert isinstance(sorted_grids.coords, cupy.ndarray)
    assert isinstance(sorted_grids.weights, cupy.ndarray)
    assert sorted_grids._non0ao_idx is None
    assert np.array_equal(
        cupy.asnumpy(sorted_grids.coords), cupy.asnumpy(coords)[forward]
    )
    assert np.array_equal(
        cupy.asnumpy(sorted_grids.weights), cupy.asnumpy(weights)[forward]
    )
    assert np.array_equal(
        cupy.asnumpy(sorted_grids.coords)[inverse], cupy.asnumpy(coords)
    )
    assert layout.forward_permutation.device.type == "cuda"
    assert layout.inverse_permutation.device.type == "cuda"


def test_gpu_atom_major_features_require_skala_grids() -> None:
    mol = gto.M(atom="H 0 0 0", basis="sto-3g", spin=1, verbose=0)
    grids = dft_gpu.Grids(mol)
    integrator = XCIntegrator(QuadraticFunctional(), device=torch.device("cuda:0"))
    dm = torch.eye(mol.nao_nr(), dtype=torch.float64, device="cuda:0")

    with pytest.raises(TypeError, match=r"requires .*\.SkalaGrids"):
        integrator(mol, grids, dm)
    with pytest.raises(TypeError, match=r"requires .*\.SkalaGrids"):
        integrator.gen_response(mol, grids, dm)


def test_gpu_skala_grids_invalidate_spatial_layout() -> None:
    mol = gto.M(atom="H 0 0 0", basis="sto-3g", spin=1, verbose=0)
    grids = GPU4PySCFSkalaGrids(mol)
    grids.level = 0
    grids.alignment = 1
    grids.build()
    integrator = XCIntegrator(QuadraticFunctional(), device=torch.device("cuda:0"))

    layout = integrator._get_spatial_grid_layout(mol, grids)
    assert integrator._get_spatial_grid_layout(mol, grids) is layout

    grids.reset()
    assert grids._spatial_grid_layout is None
    grids.level = 0
    grids.alignment = 1
    grids.build()
    assert integrator._get_spatial_grid_layout(mol, grids) is not layout


@pytest.mark.parametrize(
    ("atom", "spin", "integration_method_name"),
    [
        pytest.param("H 0 0 0; H 0 0 0.74", 0, "nr_rks", id="rks"),
        pytest.param("H 0 0 0", 1, "nr_uks", id="uks"),
    ],
)
def test_gpu_rks_uks_dense_screened_equivalence(
    load_functional_cached: Callable[..., ExcFunctionalBase | str],
    atom: str,
    spin: int,
    integration_method_name: str,
) -> None:
    mol = gto.M(atom=atom, basis="sto-3g", spin=spin, verbose=0)

    functional = load_functional_cached("skala-1.1", device=torch.device("cuda:0"))
    assert isinstance(functional, ExcFunctionalBase)
    ks = SkalaKS(mol, xc=functional, with_dftd3=False)
    ks.grids.level = 0
    ks.grids.alignment = 1
    ks.grids.build(sort_grids=False)
    dm = ks.get_init_guess()
    integrate = getattr(ks._numint, integration_method_name)

    with force_ao_screening(False):
        dense = integrate(mol, ks.grids, None, dm)

    with force_ao_screening(True):
        screened = integrate(mol, ks.grids, None, dm)

    cupy.testing.assert_allclose(dense[0], screened[0], rtol=1e-9)
    assert np.isclose(dense[1], screened[1], rtol=1e-9)
    # The trained functional amplifies the rounding differences between the
    # dense and screened AO contractions, so the potential needs a looser
    # tolerance than the integrated density and the energy.
    cupy.testing.assert_allclose(dense[2], screened[2], rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize(
    ("atom", "spin", "spin_shape"),
    [
        pytest.param("H 0 0 0; H 0 0 0.74", 0, (), id="rks"),
        pytest.param("H 0 0 0", 1, (2,), id="uks"),
    ],
)
def test_gpu_response_dense_screened_equivalence(
    atom: str, spin: int, spin_shape: tuple[int, ...]
) -> None:
    mol = gto.M(atom=atom, basis="sto-3g", spin=spin, verbose=0)
    ks = SkalaKS(mol, xc=QuadraticFunctional(), with_dftd3=False)
    ks.grids.level = 0
    ks.grids.alignment = 1
    ks.grids.build(sort_grids=False)
    matrix_shape = spin_shape + (mol.nao_nr(), mol.nao_nr())
    mo_coeff = cupy.broadcast_to(cupy.eye(mol.nao_nr()), matrix_shape).copy()
    mo_occ = cupy.ones(spin_shape + (mol.nao_nr(),))
    dm1 = cupy.arange(mol.nao_nr() ** 2, dtype=cupy.float64).reshape(
        mol.nao_nr(), mol.nao_nr()
    )
    dm1 += dm1.T
    dm1 = cupy.broadcast_to(dm1, matrix_shape).copy()

    with force_ao_screening(False):
        dense_response = ks._numint.gen_response(mo_coeff, mo_occ, ks=ks)

    with force_ao_screening(True):
        screened_response = ks._numint.gen_response(mo_coeff, mo_occ, ks=ks)

    cupy.testing.assert_allclose(
        screened_response(dm1),
        dense_response(dm1),
        rtol=1e-9,
        atol=1e-10,
    )


def test_gpu_multiblock_mgga_response_dense_screened_equivalence() -> None:
    """Exercise screened MGGA Hessian-vector products across multiple GPU blocks.

    Density-only and single-block cases cannot expose errors in block-local JVP
    assembly, spatial permutation, or reduction of vector gradient features. The
    large basis and grid force multiple GPU AO blocks, while the quadratic density,
    gradient, and kinetic terms give a nonzero response for every MGGA feature path.
    """
    mol = gto.M(atom=CARBON_CHAIN, basis="def2-qzvpp", verbose=0)
    ks = SkalaKS(
        mol,
        xc=QuadraticFunctional(
            [
                Feature.ATOMIC_GRID_SIZES,
                Feature.DENSITY,
                Feature.GRAD,
                Feature.KIN,
                Feature.GRID_WEIGHTS,
            ]
        ),
        with_dftd3=False,
    )
    ks.grids.level = 1
    ks.grids.alignment = 1
    ks.grids.build(sort_grids=False)
    assert ks.grids.weights.size > dft_gpu.numint.MIN_BLK_SIZE
    mo_coeff = cupy.eye(mol.nao_nr())
    mo_occ = cupy.ones(mol.nao_nr())
    dm1 = cupy.eye(mol.nao_nr())

    with force_ao_screening(False):
        dense_response = ks._numint.gen_response(mo_coeff, mo_occ, ks=ks)

    with force_ao_screening(True):
        screened_response = ks._numint.gen_response(mo_coeff, mo_occ, ks=ks)

    cupy.testing.assert_allclose(
        screened_response(dm1),
        dense_response(dm1),
        rtol=1e-9,
        atol=1e-9,
    )


def test_gpu_screened_skala_matches_cpu_on_carbon_chain(
    load_functional_cached: Callable[..., ExcFunctionalBase | str],
) -> None:
    """Prevent inaccurate GPU AO screening on spatially diffuse grid blocks.

    GPU4PySCF builds one active-shell mask for each fixed-size coordinate block. That
    screening is reliable only when the points in a block are spatially local enough
    for the sampled AO values to represent the whole block. Passing Skala's unsorted,
    atom-major grid directly would allow one GPU block to span a large region around
    an atom. This is especially problematic for the AO derivatives used by Skala: an
    AO value can be small at the sampled points even though its gradient still makes
    a significant contribution. The implementation therefore partitions the whole
    molecular grid into exact-size spatial blocks for AO evaluation, then restores
    atom-major feature order before evaluating the model. The linear carbon chain and
    large def2-QZVPP basis expose regressions in that ordering on a reasonably small
    system.

    The CPU and GPU calculations use identical coordinates, weights, density matrix,
    and Skala 1.1 model. CPU AO evaluation is deliberately forced dense to provide an
    independent reference, while GPU AO evaluation is deliberately forced through
    screening. Comparing the particle count, XC energy, and complete XC potential
    matrix verifies the full feature and VJP path; the potential is particularly
    sensitive to omitted derivative contributions.
    """
    mol = gto.M(atom=CARBON_CHAIN, basis="def2-qzvpp", verbose=0)
    cpu_grids = PySCFSkalaGrids(mol)
    cpu_grids.level = 1
    cpu_grids.alignment = 1
    cpu_grids.build(sort_grids=False)
    gpu_grids = GPU4PySCFSkalaGrids(mol)
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
    cpu_functional = load_functional_cached("skala-1.1", device=torch.device("cpu"))
    gpu_functional = load_functional_cached("skala-1.1", device=torch.device("cuda:0"))
    assert isinstance(cpu_functional, ExcFunctionalBase)
    assert isinstance(gpu_functional, ExcFunctionalBase)

    with force_ao_screening(False):
        cpu_result: tuple[float, float, np.ndarray[Any, F64]] = SkalaNumInt(
            cpu_functional, device=torch.device("cpu")
        ).nr_rks(mol, cpu_grids, None, dm)

    with force_ao_screening(True):
        gpu_result: tuple[float, float, Array[Any, F64]] = SkalaNumInt(
            gpu_functional, device=torch.device("cuda:0")
        ).nr_rks(mol, gpu_grids, None, cupy.asarray(dm))

    gpu_vxc = cupy.asnumpy(gpu_result[2])
    vxc_difference = cpu_result[2] - gpu_vxc
    vxc_max_abs_difference = np.max(np.abs(vxc_difference))
    vxc_relative_l2_difference = np.linalg.norm(vxc_difference) / np.linalg.norm(
        cpu_result[2]
    )
    assert (
        np.isclose(cpu_result[0], gpu_result[0], rtol=1e-10, atol=1e-11)
        and np.isclose(cpu_result[1], gpu_result[1], rtol=1e-8, atol=1e-9)
        and vxc_max_abs_difference < 2e-7
        and vxc_relative_l2_difference < 1e-7
    ), (
        f"N: cpu={cpu_result[0]:.16g}, gpu={gpu_result[0]:.16g}, "
        f"abs_diff={abs(cpu_result[0] - gpu_result[0]):.3e}; "
        f"E_xc: cpu={cpu_result[1]:.16g}, gpu={gpu_result[1]:.16g}, "
        f"abs_diff={abs(cpu_result[1] - gpu_result[1]):.3e}; "
        f"V_xc: max_abs_diff={vxc_max_abs_difference:.3e}, "
        f"relative_l2_diff={vxc_relative_l2_difference:.3e}"
    )


def test_gpu_empty_ao_block_matches_dense_reference() -> None:
    """Preserve grid alignment when GPU4PySCF finds no AOs in a block.

    GPU4PySCF normally omits fixed-size grid blocks whose screening mask contains
    no active atomic orbitals. Skala assigns each yielded result to a cumulative
    grid slice, so omitting an empty block would shift every later result into the
    wrong positions. ``strict_grid_order=True`` makes the backend yield the empty
    block and allows Skala to advance that slice before processing active blocks.

    The first block is placed far from the molecule to make its active-AO set
    empty, while the second block samples the molecular region. Comparing MGGA
    features and their density-matrix VJP with dense AO evaluation verifies both
    forward placement and backward slicing through the real GPU backend.
    """
    mol = gto.M(atom="C 0 0 0", basis="sto-3g", spin=2, verbose=0)
    assert dft_gpu is not None
    block_size = int(dft_gpu.numint.MIN_BLK_SIZE)
    far_coords = cupy.full((block_size, 3), 100.0, dtype=cupy.float64)
    near_coords = cupy.linspace(-0.5, 0.5, block_size * 3, dtype=cupy.float64).reshape(
        block_size, 3
    )
    coords = cupy.concatenate((far_coords, near_coords))
    grids = dft_gpu.Grids(mol)
    grids.coords = coords
    grids.weights = cupy.ones(coords.shape[0], dtype=cupy.float64)

    screening_numint = dft_gpu.numint.NumInt().build(mol, coords)
    active_ao_counts = [
        len(block[1]) for block in grids.get_non0ao_idx(screening_numint.gdftopt)
    ]
    assert active_ao_counts[0] == 0
    assert active_ao_counts[1] > 0
    grids._non0ao_idx = None

    feature_function = MGGAFeatureFunction(
        FeatureSpec([Feature.DENSITY, Feature.GRAD, Feature.KIN])
    )
    dm = torch.eye(mol.nao_nr(), dtype=torch.float64, device="cuda", requires_grad=True)
    screened = evaluate_ao_features_blockwise(
        dm, mol, grids, feature_function, block_size, False
    )
    dense = evaluate_full_grid(dm, mol, coords, feature_function, gpu=True)

    torch.testing.assert_close(screened, dense, rtol=1e-12, atol=1e-12)
    (screened_vjp,) = torch.autograd.grad(screened.square().sum(), dm)
    (dense_vjp,) = torch.autograd.grad(dense.square().sum(), dm)
    torch.testing.assert_close(screened_vjp, dense_vjp, rtol=1e-12, atol=5e-8)


def test_gpu_sparse_mask_sorts_scatters_and_unsorts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mol = gto.M(atom="C 0 0 0", basis="sto-3g", spin=2, verbose=0)
    assert dft_gpu is not None
    block_size = int(dft_gpu.numint.MIN_BLK_SIZE)
    ngrids = 2 * block_size
    sort_idx = np.array([2, 0, 4, 1, 3])
    active_sorted_aos = np.array([0, 2, 4])
    ao = cupy.arange(active_sorted_aos.size * block_size, dtype=cupy.float64).reshape(
        active_sorted_aos.size, block_size
    )
    weights = cupy.ones(ngrids)
    coords = cupy.zeros((ngrids, 3))
    grids = SimpleNamespace(weights=weights, coords=coords)

    class FakeGpuNumInt:
        def build(self, mol: gto.Mole, coords: Array[D2, F64]) -> FakeGpuNumInt:
            self.gdftopt = SimpleNamespace(_ao_idx=sort_idx)
            return self

        def block_loop(
            self, *args: object, **kwargs: object
        ) -> Iterator[tuple[object, object, object, object]]:
            assert kwargs["strict_grid_order"] is True
            yield (
                cupy.empty((0, block_size), dtype=cupy.float64),
                cupy.empty(0, dtype=cupy.int64),
                weights[:block_size],
                coords[:block_size],
            )
            yield (
                ao,
                cupy.asarray(active_sorted_aos),
                weights[block_size:],
                coords[block_size:],
            )

    monkeypatch.setattr(dft_gpu.numint, "NumInt", FakeGpuNumInt)

    feature_function = MGGAFeatureFunction(FeatureSpec([Feature.DENSITY]))
    dm = torch.diag(
        torch.arange(1, mol.nao_nr() + 1, dtype=torch.float64, device="cuda")
    ).requires_grad_()
    features = evaluate_ao_features_blockwise(
        dm, mol, grids, feature_function, None, False
    )

    sort_idx_t = torch.as_tensor(sort_idx, device="cuda")
    active_t = torch.as_tensor(active_sorted_aos, device="cuda")
    dm_sorted = dm[..., sort_idx_t, :][..., sort_idx_t]
    dm_active = dm_sorted[..., active_t[:, None], active_t[None, :]]
    ao_t = from_dlpack(ao)
    expected = torch.zeros_like(features)
    expected[..., block_size:] = torch.sum((dm_active @ ao_t) * ao_t, dim=0).unsqueeze(
        0
    )
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
