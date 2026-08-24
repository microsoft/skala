from collections.abc import Callable, Iterator
from itertools import combinations
from typing import Any, TypeAlias, cast

import numpy as np
import pytest
import torch
from pyscf import dft, gto
from skala.features import Feature, FeatureMap
from skala.functional.base import ExcFunctionalBase
from skala.pyscf import ao_evaluation as ao_evaluation_module
from skala.pyscf import grids as grids_module
from skala.pyscf import model_chunking as model_chunking_module
from skala.pyscf import spatial_grid_layout as spatial_grid_layout_module
from skala.pyscf import xc_integrator as xc_integrator_module
from skala.pyscf.ao_evaluation import (
    _active_cpu_ao_indices,
    _AOBlock,
    _CPUAOBlockLoop,
    _evaluate_feature_block,
    _resolve_ao_block_size,
    evaluate_ao_features_blockwise,
)
from skala.pyscf.evaluation import FeatureSpec
from skala.pyscf.feature_math import MGGAFeatureFunction
from skala.pyscf.grids import SkalaGrids
from skala.pyscf.model_chunking import ModelFeatureChunk
from skala.pyscf.numint import SkalaNumInt
from skala.pyscf.spatial_grid_layout import (
    SpatialGridLayout,
    _decompose_grid_into_spatial_blocks,
    prepare_spatial_grid_layout,
)
from skala.pyscf.xc_integrator import XCIntegrator
from skala.typing import F64

from tests.utils import QuadraticFunctional, force_ao_screening

_NumPyNumInt: TypeAlias = SkalaNumInt[np.ndarray[Any, F64]]

_MGGA_FEATURES = (Feature.DENSITY, Feature.GRAD, Feature.KIN, Feature.LAPL)
_MGGA_FEATURE_COMBINATIONS = [
    combination
    for size in range(1, len(_MGGA_FEATURES) + 1)
    for combination in combinations(_MGGA_FEATURES, size)
]


@pytest.fixture
def carbon() -> gto.Mole:
    return gto.M(atom="C 0 0 0", basis="sto-3g", spin=2, verbose=0)


@pytest.mark.parametrize(
    ("feature_names", "expected_deriv", "expected_nfeats"),
    [
        ({Feature.DENSITY}, 0, 1),
        ({Feature.GRAD}, 1, 3),
        ({Feature.KIN}, 1, 1),
        ({Feature.LAPL}, 2, 1),
        (
            {
                Feature.DENSITY,
                Feature.GRAD,
                Feature.KIN,
                Feature.LAPL,
            },
            2,
            6,
        ),
    ],
)
def test_mgga_supported_features_are_linear_in_density_matrix(
    feature_names: set[Feature],
    expected_deriv: int,
    expected_nfeats: int,
) -> None:
    """Check each supported feature layout and its linear dependence on ``dm``.

    Linearity requires the first JVP to equal direct feature evaluation on the
    tangent and the second JVP to vanish.
    """
    feature_spec = FeatureSpec(feature_names)
    feature_function = MGGAFeatureFunction(feature_spec)
    ncomp = (expected_deriv + 1) * (expected_deriv + 2) * (expected_deriv + 3) // 6
    ao = torch.arange(1, ncomp * 2 * 3 + 1, dtype=torch.float64).reshape(ncomp, 2, 3)
    if expected_deriv == 0:
        ao = ao[0]
    dm = torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=torch.float64)
    tangent = torch.tensor([[0.2, -0.1], [-0.1, 0.3]], dtype=torch.float64)

    features = feature_function(dm, ao)
    feature_jvp = torch.func.jvp(
        lambda value: feature_function(value, ao),
        (dm,),
        (tangent,),
    )[1]

    def first_jvp(value: torch.Tensor) -> torch.Tensor:
        result = torch.func.jvp(
            lambda inner: feature_function(inner, ao),
            (value,),
            (tangent,),
        )[1]
        assert isinstance(result, torch.Tensor)
        return result

    second_jvp = torch.func.jvp(
        first_jvp,
        (dm,),
        (torch.ones_like(dm),),
    )[1]

    assert feature_function.deriv == expected_deriv
    assert feature_function.nfeats == expected_nfeats
    assert feature_function.feature_spec is feature_spec
    assert features.shape == (expected_nfeats, 3)
    assert set(feature_function.to_dict(features)) == feature_names
    torch.testing.assert_close(feature_jvp, feature_function(tangent, ao))
    torch.testing.assert_close(second_jvp, torch.zeros_like(second_jvp))


@pytest.mark.parametrize(
    "feature_names",
    _MGGA_FEATURE_COMBINATIONS,
)
@pytest.mark.parametrize("spin_channels", [None, 2])
def test_mgga_analytic_vjp_matches_autograd(
    feature_names: tuple[Feature, ...], spin_channels: int | None
) -> None:
    """Match the analytic MGGA VJP to autograd for every feature and spin layout."""
    feature_function = MGGAFeatureFunction(FeatureSpec(feature_names))
    ncomp = (
        (feature_function.deriv + 1)
        * (feature_function.deriv + 2)
        * (feature_function.deriv + 3)
        // 6
    )
    generator = torch.Generator().manual_seed(0)
    ao = torch.randn((ncomp, 3, 5), dtype=torch.float64, generator=generator)
    if feature_function.deriv == 0:
        ao = ao[0]
    dm_shape = (3, 3) if spin_channels is None else (spin_channels, 3, 3)
    dm = torch.randn(dm_shape, dtype=torch.float64, generator=generator)

    vjp_result = torch.func.vjp(lambda value: feature_function(value, ao), dm)
    features = vjp_result[0]
    pullback = vjp_result[1]
    cotangent = torch.randn(features.shape, dtype=features.dtype, generator=generator)

    expected = pullback(cotangent)[0]
    actual = feature_function.vjp(ao, cotangent)

    torch.testing.assert_close(actual, expected)


def test_feature_block_compiled_vjp_matches_eager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Match compiled blockwise feature evaluation and VJP to eager execution."""
    feature_function = MGGAFeatureFunction(FeatureSpec(_MGGA_FEATURES))
    generator = torch.Generator().manual_seed(0)
    block = _AOBlock(
        ao_values=torch.randn((10, 3, 5), dtype=torch.float64, generator=generator),
        active_ao_indices=None,
        grid_slice=slice(1, 6),
    )
    dm = torch.randn((3, 3), dtype=torch.float64, generator=generator)
    cotangent = torch.randn(
        (feature_function.nfeats, 7), dtype=torch.float64, generator=generator
    )
    eager_forward = _evaluate_feature_block(
        feature_function, block, dm, compile_feature_function=False
    )
    eager_vjp = _evaluate_feature_block(
        feature_function,
        block,
        None,
        compile_feature_function=False,
        feature_cotangent=cotangent,
    )

    compile_function = torch.compile
    monkeypatch.setattr(
        torch,
        "compile",
        lambda function: compile_function(function, backend="eager"),
    )

    compiled_forward = _evaluate_feature_block(
        feature_function, block, dm, compile_feature_function=True
    )
    compiled_vjp = _evaluate_feature_block(
        feature_function,
        block,
        None,
        compile_feature_function=True,
        feature_cotangent=cotangent,
    )

    torch.testing.assert_close(compiled_forward, eager_forward)
    torch.testing.assert_close(compiled_vjp, eager_vjp)


@pytest.mark.parametrize("feature_names", [[], [Feature.GRID_WEIGHTS]])
def test_mgga_requires_at_least_one_ao_derived_feature(
    feature_names: list[Feature],
) -> None:
    """Reject MGGA feature functions that contain no AO-derived feature."""
    with pytest.raises(
        ValueError, match="At least one AO-derived feature must be selected"
    ):
        MGGAFeatureFunction(FeatureSpec(feature_names))


def test_patch_ao_screening_restores_previous_decision(carbon: gto.Mole) -> None:
    """Restore the previous screening decision after nested route patches exit."""
    original_decision = xc_integrator_module._should_screen_aos

    with force_ao_screening(False):
        dense_decision = xc_integrator_module._should_screen_aos
        assert not dense_decision(carbon)
        with force_ao_screening(True):
            assert xc_integrator_module._should_screen_aos(carbon)
        assert xc_integrator_module._should_screen_aos is dense_decision

    assert xc_integrator_module._should_screen_aos is original_decision


def test_active_cpu_ao_indices(carbon: gto.Mole) -> None:
    """Expand active CPU shells to AO indices and preserve an empty selection."""
    ao_loc = carbon.ao_loc_nr()
    screen_index = np.zeros((2, carbon.nbas), dtype=np.uint8)
    screen_index[0, 0] = 1
    screen_index[1, -1] = 1

    expected = np.concatenate(
        (
            np.arange(ao_loc[0], ao_loc[1]),
            np.arange(ao_loc[-2], ao_loc[-1]),
        )
    )

    assert np.array_equal(_active_cpu_ao_indices(carbon, screen_index), expected)

    empty = _active_cpu_ao_indices(carbon, np.zeros_like(screen_index))
    assert empty.dtype == np.int64
    assert empty.size == 0


def test_resolve_ao_block_size_modes(carbon: gto.Mole) -> None:
    """Resolve aligned CPU block sizes and reject explicit GPU block sizing."""
    feature_function = MGGAFeatureFunction(FeatureSpec([Feature.DENSITY]))
    backend_block_size = dft.gen_grid.BLKSIZE

    # CPU sizes are aligned locally; GPU sizing is delegated unless explicitly invalid.
    automatic = _resolve_ao_block_size(
        carbon, feature_function, block_size=None, max_memory=0, gpu=False
    )
    explicit = _resolve_ao_block_size(
        carbon,
        feature_function,
        block_size=backend_block_size + 1,
        max_memory=0,
        gpu=False,
    )

    assert automatic == 4 * backend_block_size
    assert explicit == backend_block_size
    assert (
        _resolve_ao_block_size(
            carbon, feature_function, block_size=None, max_memory=0, gpu=True
        )
        is None
    )
    with pytest.raises(ValueError, match="custom block size"):
        _resolve_ao_block_size(
            carbon,
            feature_function,
            block_size=backend_block_size,
            max_memory=0,
            gpu=True,
        )


@pytest.mark.parametrize(("ngrids", "block_size"), [(0, 4), (3, 4), (8, 4), (10, 4)])
def test_decompose_grid_into_spatial_blocks_restores_original_order(
    ngrids: int, block_size: int
) -> None:
    """Produce complete spatial blocks and an inverse restoring original order."""
    coords = np.arange(3 * ngrids, dtype=np.float64).reshape(ngrids, 3)

    forward, inverse = _decompose_grid_into_spatial_blocks(coords, block_size)

    assert np.array_equal(np.sort(forward), np.arange(ngrids))
    assert np.array_equal(coords[forward][inverse], coords)
    assert all(
        len(forward[start : start + block_size]) == block_size
        for start in range(0, ngrids - block_size + 1, block_size)
    )
    assert len(forward) % block_size == ngrids % block_size


def test_decompose_grid_into_spatial_blocks_groups_interleaved_clusters() -> None:
    """Group interleaved spatial clusters into homogeneous evaluator blocks."""
    block_size = 3
    labels = np.tile(np.arange(4), block_size)
    offsets = np.repeat(np.arange(block_size), 4)
    coords = np.column_stack(
        (100.0 * labels + offsets, np.zeros(labels.size), np.zeros(labels.size))
    )

    forward, _ = _decompose_grid_into_spatial_blocks(coords, block_size)

    grouped_labels = labels[forward].reshape(-1, block_size)
    assert np.all(grouped_labels == grouped_labels[:, :1])


def test_decompose_grid_into_spatial_blocks_uses_principal_direction() -> None:
    """Partition an oblique point cloud along its principal spatial direction."""
    longitudinal = np.arange(-3.5, 4.0)
    transverse = 0.45 * (np.square(longitudinal) - np.mean(np.square(longitudinal)))
    coords = np.column_stack(
        (
            longitudinal + transverse,
            longitudinal - transverse,
            np.zeros(longitudinal.size),
        )
    )

    forward, _ = _decompose_grid_into_spatial_blocks(coords, block_size=2)

    assert set(forward[:4]) == set(range(4))
    assert set(forward[4:]) == set(range(4, 8))


def test_decompose_grid_into_spatial_blocks_handles_identical_points() -> None:
    """Keep a deterministic permutation when all grid coordinates are identical."""
    coords = np.ones((10, 3), dtype=np.float64)

    forward, inverse = _decompose_grid_into_spatial_blocks(coords, block_size=4)

    assert np.array_equal(forward, np.arange(coords.shape[0]))
    assert np.array_equal(inverse, forward)


def test_prepare_spatially_sorted_cpu_grids(
    carbon: gto.Mole, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep block-local AO masks aligned with a reversible spatial grid ordering.

    Screened AO evaluation reorders atom-major grid points into spatial blocks before
    PySCF builds its screening mask. Coordinates, weights, and mask must share that
    ordering, while the saved inverse permutation restores model features and leaves
    the caller's original grid unchanged.
    """
    coords = np.arange(18, dtype=np.float64).reshape(6, 3)
    weights = np.arange(6, dtype=np.float64) + 10
    grids = dft.Grids(carbon)
    grids.coords = coords
    grids.weights = weights
    forward = np.array([4, 2, 0, 5, 3, 1], dtype=np.int64)
    inverse = np.argsort(forward)
    non0tab = np.ones((1, carbon.nbas), dtype=np.uint8)
    partition_calls = 0
    decomposition_block_sizes: list[int] = []

    def fake_decompose_grid_into_spatial_blocks(
        coords_arg: np.ndarray, block_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        nonlocal partition_calls
        partition_calls += 1
        decomposition_block_sizes.append(block_size)
        assert coords_arg is grids.coords
        return forward, inverse

    monkeypatch.setattr(
        spatial_grid_layout_module,
        "_decompose_grid_into_spatial_blocks",
        fake_decompose_grid_into_spatial_blocks,
    )

    screen_index_calls = 0
    screened_molecules: list[gto.Mole] = []

    def fake_make_screen_index(
        mol_arg: gto.Mole, sorted_coords: np.ndarray, cutoff: float
    ) -> np.ndarray:
        nonlocal screen_index_calls
        screen_index_calls += 1
        screened_molecules.append(mol_arg)
        assert np.array_equal(sorted_coords, coords[forward])
        assert cutoff == grids.cutoff
        return non0tab

    monkeypatch.setattr(dft.gen_grid, "make_screen_index", fake_make_screen_index)

    device = torch.device("cpu")
    layout = prepare_spatial_grid_layout(carbon, grids, block_size=2, device=device)
    sorted_grids = layout.sorted_grids

    assert sorted_grids is not grids
    assert np.array_equal(grids.coords, coords)
    assert np.array_equal(grids.weights, weights)
    assert np.array_equal(sorted_grids.coords, coords[forward])
    assert np.array_equal(sorted_grids.weights, weights[forward])
    assert sorted_grids.non0tab is non0tab
    torch.testing.assert_close(
        layout.forward_permutation, torch.as_tensor(forward, device=device)
    )
    torch.testing.assert_close(
        layout.inverse_permutation, torch.as_tensor(inverse, device=device)
    )
    assert partition_calls == 1
    assert screen_index_calls == 1
    assert decomposition_block_sizes == [2]
    assert screened_molecules == [carbon]
    assert not hasattr(grids, "_spatial_grid_layout")


def test_grid_reuses_spatial_grid_layout_across_numints(
    carbon: gto.Mole, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cache one spatial layout on each grid independently of the NumInt instance."""
    grids = SkalaGrids(carbon)
    grids.coords = np.arange(18, dtype=np.float64).reshape(6, 3)
    grids.weights = np.arange(6, dtype=np.float64)
    other_grids = SkalaGrids(carbon)
    other_grids.coords = grids.coords.copy()
    other_grids.weights = grids.weights.copy()
    layouts: list[SpatialGridLayout] = []

    def fake_prepare_spatial_grid_layout(
        mol: gto.Mole,
        grids: object,
        block_size: int,
        device: torch.device,
    ) -> SpatialGridLayout:
        layout = SpatialGridLayout(
            block_size=block_size,
            sorted_grids=grids,
            forward_permutation=torch.arange(6, device=device),
            inverse_permutation=torch.arange(6, device=device),
        )
        layouts.append(layout)
        return layout

    monkeypatch.setattr(
        grids_module,
        "prepare_spatial_grid_layout",
        fake_prepare_spatial_grid_layout,
    )
    numint: _NumPyNumInt = SkalaNumInt(QuadraticFunctional())
    other_numint: _NumPyNumInt = SkalaNumInt(QuadraticFunctional())

    layout = numint.integrator._get_spatial_grid_layout(carbon, grids)
    assert other_numint.integrator._get_spatial_grid_layout(carbon, grids) is layout
    assert grids._spatial_grid_layout is layout
    assert len(layouts) == 1

    numint.reset()
    assert numint.integrator._get_spatial_grid_layout(carbon, grids) is layout

    other_layout = numint.integrator._get_spatial_grid_layout(carbon, other_grids)
    assert other_layout is not layout
    assert other_grids._spatial_grid_layout is other_layout
    assert len(layouts) == 2

    assert not grids.coords.flags.writeable
    assert not grids.weights.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        grids.coords[0, 0] = -1
    with pytest.raises(ValueError, match="read-only"):
        grids.weights[0] = -1

    replacement_coords = grids.coords.copy()
    grids.coords = replacement_coords
    assert grids._spatial_grid_layout is None
    assert replacement_coords.flags.writeable


class FakeKS:
    def __init__(self, mol: gto.Mole, grids: object | None = None) -> None:
        self.mol = mol
        self.grids = grids or object()
        self.max_memory = 100

    def make_rdm1(self, mo_coeff: np.ndarray, mo_occ: np.ndarray) -> np.ndarray:
        return np.eye(self.mol.nao_nr())

    def get_j(self, mol: gto.Mole, dm: np.ndarray, hermi: int) -> np.ndarray:
        return np.zeros_like(dm)


def test_call_rejects_second_order_evaluation(carbon: gto.Mole) -> None:
    """Direct callers must use gen_response instead of second_order evaluation."""
    numint: _NumPyNumInt = SkalaNumInt(QuadraticFunctional())

    with pytest.raises(NotImplementedError, match="second-order evaluation"):
        numint(
            carbon,
            dft.Grids(carbon),
            None,
            torch.eye(carbon.nao_nr(), dtype=torch.float64),
            second_order=True,
        )


@pytest.mark.parametrize("expected", [False, True])
@pytest.mark.parametrize("response_safety_fraction", [None, 0.6])
def test_first_and_second_order_use_same_screening_decision(
    carbon: gto.Mole,
    monkeypatch: pytest.MonkeyPatch,
    expected: bool,
    response_safety_fraction: float | None,
) -> None:
    """Route energy and response consistently and propagate response memory policy."""
    routes: list[str] = []
    safety_fractions: list[float] = []

    def fake_generate_features(
        mol: gto.Mole,
        dm: torch.Tensor,
        grids: object,
        features: set[Feature] | None = None,
        **kwargs: object,
    ) -> FeatureMap:
        routes.append("dense")
        density = dm.square().sum().reshape(1).expand(2, 1) / 2
        return {
            Feature.ATOMIC_GRID_SIZES: torch.tensor([1]),
            Feature.DENSITY: density,
            Feature.GRID_WEIGHTS: torch.ones(1, dtype=dm.dtype),
        }

    class FakeSpatialGridLayout:
        block_size = 1
        forward_permutation = torch.tensor([0])
        inverse_permutation = torch.tensor([0])

        def __init__(self, sorted_grids: object) -> None:
            self.sorted_grids = sorted_grids

    class FakeModelFeatureChunks:
        def __init__(self, raw_features: torch.Tensor) -> None:
            self.raw_features = raw_features

        def __iter__(self) -> Iterator[ModelFeatureChunk]:
            raw_features = self.raw_features.detach().requires_grad_()
            yield ModelFeatureChunk(
                grid_indices=torch.tensor([0]),
                raw_features=raw_features,
                model_features={
                    Feature.ATOMIC_GRID_SIZES: torch.tensor([1]),
                    Feature.DENSITY: raw_features.expand(2, 1) / 2,
                    Feature.GRID_WEIGHTS: torch.ones(1, dtype=raw_features.dtype),
                },
            )

    def fake_prepare_spatial_grid_layout(
        mol: gto.Mole,
        grids: object,
        block_size: int,
        device: torch.device,
    ) -> FakeSpatialGridLayout:
        return FakeSpatialGridLayout(grids)

    def fake_evaluate_ao_features_blockwise(
        dm: torch.Tensor,
        *args: object,
    ) -> torch.Tensor:
        routes.append("screened")
        return dm.sum().reshape(1, 1)

    def fake_model_feature_chunker(
        mol: gto.Mole,
        dm: torch.Tensor,
        grids: object,
        atom_major_raw_features: torch.Tensor,
        feature_function: MGGAFeatureFunction,
        deriv_order: int,
        **kwargs: object,
    ) -> FakeModelFeatureChunks:
        safety_fraction = kwargs["safety_fraction"]
        assert isinstance(safety_fraction, float)
        safety_fractions.append(safety_fraction)
        return FakeModelFeatureChunks(atom_major_raw_features)

    monkeypatch.setattr(
        xc_integrator_module, "generate_features", fake_generate_features
    )
    monkeypatch.setattr(
        grids_module,
        "prepare_spatial_grid_layout",
        fake_prepare_spatial_grid_layout,
    )
    monkeypatch.setattr(
        ao_evaluation_module,
        "evaluate_ao_features_blockwise",
        fake_evaluate_ao_features_blockwise,
    )
    monkeypatch.setattr(
        xc_integrator_module,
        "ModelFeatureChunker",
        fake_model_feature_chunker,
    )
    numint: _NumPyNumInt = SkalaNumInt(QuadraticFunctional())
    dm = torch.eye(carbon.nao_nr(), dtype=torch.float64)
    grids = SkalaGrids(carbon)
    grids.coords = np.zeros((1, 3))
    grids.weights = np.ones(1)

    ks = FakeKS(carbon, grids)
    response_kwargs = (
        {}
        if response_safety_fraction is None
        else {"safety_fraction": response_safety_fraction}
    )
    with force_ao_screening(expected):
        numint(carbon, grids, None, dm)
        response = numint.gen_response(
            np.eye(carbon.nao_nr()),
            np.ones(carbon.nao_nr()),
            ks=ks,
            **response_kwargs,
        )
        result = response(np.eye(carbon.nao_nr()))

    assert result.shape == (carbon.nao_nr(), carbon.nao_nr())
    expected_route = "screened" if expected else "dense"
    expected_route_count = 3 if expected else 2
    assert routes == [expected_route] * expected_route_count
    if expected:
        assert safety_fractions == [
            0.8,
            0.8 if response_safety_fraction is None else response_safety_fraction,
        ]
    else:
        assert safety_fractions == []


def test_feature_block_helper_localizes_derivative_vectors() -> None:
    """Apply derivative vectors in the local coordinate space of one AO block.

    A screened block contains only selected AO rows and a slice of the global grid.
    The forward JVP must therefore use the active-AO density submatrix, while the
    adjoint calculation must select only this block's grid cotangent. Comparing both
    operations with direct local formulas catches mixing up AO and grid localization.
    """
    feature_function = MGGAFeatureFunction(FeatureSpec([Feature.DENSITY]))
    block = _AOBlock(
        ao_values=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64),
        active_ao_indices=torch.tensor([0, 2]),
        grid_slice=slice(1, 3),
    )
    tangent_ordered = torch.tensor(
        [[0.5, 1.0, -0.2], [1.0, 0.4, 0.3], [-0.2, 0.3, 0.7]],
        dtype=torch.float64,
    )
    feature_jvp = _evaluate_feature_block(
        feature_function,
        block,
        block.select_active_ao_submatrix(tangent_ordered),
        compile_feature_function=False,
    )
    expected_jvp = feature_function(
        block.select_active_ao_submatrix(tangent_ordered), block.ao_values
    )
    torch.testing.assert_close(feature_jvp, expected_jvp)

    full_grid_cotangent = torch.tensor([[10.0, 0.25, -0.5, 20.0]], dtype=torch.float64)
    feature_vjp = _evaluate_feature_block(
        feature_function,
        block,
        None,
        compile_feature_function=False,
        feature_cotangent=full_grid_cotangent,
    )
    local_cotangent = full_grid_cotangent[0, block.grid_slice]
    expected_vjp = torch.einsum(
        "g,ig,jg->ij", local_cotangent, block.ao_values, block.ao_values
    )
    torch.testing.assert_close(feature_vjp, expected_vjp)


def test_blockwise_ao_feature_transforms_follow_linear_operator(
    carbon: gto.Mole,
) -> None:
    """Check spin-resolved first and second JVPs and the adjoint JVP."""
    grids = _minimal_atom_grid(carbon)
    feature_function = MGGAFeatureFunction(FeatureSpec([Feature.DENSITY]))
    identity = torch.eye(carbon.nao_nr(), dtype=torch.float64)
    dm = torch.stack((identity, 2 * identity))
    tangent = torch.arange(1, dm.numel() + 1, dtype=dm.dtype).reshape(dm.shape)

    def evaluate(value: torch.Tensor) -> torch.Tensor:
        return evaluate_ao_features_blockwise(
            value, carbon, grids, feature_function, None, False
        )

    jvp_result = torch.func.jvp(evaluate, (dm,), (tangent,))
    features = jvp_result[0]
    feature_tangent = jvp_result[1]
    assert features.shape[:1] == dm.shape[:-2]
    torch.testing.assert_close(feature_tangent, evaluate(tangent))

    def first_jvp(value: torch.Tensor) -> torch.Tensor:
        result = torch.func.jvp(evaluate, (value,), (tangent,))[1]
        assert isinstance(result, torch.Tensor)
        return result

    second_jvp = torch.func.jvp(first_jvp, (dm,), (torch.ones_like(dm),))[1]
    torch.testing.assert_close(second_jvp, torch.zeros_like(features))

    feature_cotangent = torch.arange(
        1, features.numel() + 1, dtype=features.dtype
    ).reshape(features.shape)
    cotangent_tangent = torch.flip(feature_cotangent, dims=(-1,))

    def apply_adjoint(value: torch.Tensor) -> torch.Tensor:
        return evaluate_ao_features_blockwise(
            value,
            carbon,
            grids,
            feature_function,
            None,
            False,
            adjoint=True,
        )

    dm_cotangent = apply_adjoint(feature_cotangent)
    assert dm_cotangent.shape == dm.shape
    torch.testing.assert_close(
        torch.sum(features * feature_cotangent),
        torch.sum(dm * dm_cotangent),
    )
    adjoint_tangent = torch.func.jvp(
        apply_adjoint,
        (feature_cotangent,),
        (cotangent_tangent,),
    )[1]
    torch.testing.assert_close(adjoint_tangent, apply_adjoint(cotangent_tangent))


def test_cpu_screening_slices_and_scatters_full_derivatives(
    carbon: gto.Mole, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Slice active CPU AOs per block and scatter VXC and HVP to full matrices."""
    block_size = dft.gen_grid.BLKSIZE
    ngrids = 2 * block_size
    grids = dft.Grids(carbon)
    grids.coords = np.zeros((ngrids, 3))
    grids.weights = np.ones(ngrids)

    ao = np.arange(ngrids * carbon.nao_nr(), dtype=np.float64).reshape(
        ngrids, carbon.nao_nr()
    )
    screen_index = np.zeros((2, carbon.nbas), dtype=np.uint8)
    screen_index[0, 0] = 1
    screen_index[1, -1] = 1
    grids.non0tab = screen_index
    active_ao_indices = _active_cpu_ao_indices(carbon, screen_index)

    class FakeNumInt:
        def block_loop(
            self, *args: object, **kwargs: object
        ) -> Iterator[tuple[np.ndarray, None, np.ndarray, np.ndarray]]:
            assert kwargs["non0tab"] is screen_index
            assert "strict_grid_order" not in kwargs
            for start in range(0, ngrids, block_size):
                grid_slice = slice(start, start + block_size)
                yield (
                    ao[grid_slice],
                    None,
                    grids.weights[grid_slice],
                    grids.coords[grid_slice],
                )

    monkeypatch.setattr(dft.numint, "NumInt", FakeNumInt)

    feature_function = MGGAFeatureFunction(FeatureSpec([Feature.DENSITY]))
    dm = torch.diag(
        torch.arange(1, carbon.nao_nr() + 1, dtype=torch.float64)
    ).requires_grad_()
    features = evaluate_ao_features_blockwise(
        dm, carbon, grids, feature_function, block_size, False
    )

    expected_blocks = []
    for block_index, start in enumerate(range(0, ngrids, block_size)):
        block_active_ao_indices = _active_cpu_ao_indices(
            carbon, screen_index[block_index : block_index + 1]
        )
        grid_slice = slice(start, start + block_size)
        active_ao_values = torch.from_numpy(
            ao[grid_slice][:, block_active_ao_indices]
        ).T
        active_dm_submatrix = dm[
            ...,
            block_active_ao_indices[:, None],
            block_active_ao_indices[None, :],
        ]
        expected_blocks.append(
            torch.sum(
                (active_dm_submatrix @ active_ao_values) * active_ao_values, dim=0
            )
        )
    expected = torch.cat(expected_blocks).unsqueeze(0)
    assert torch.allclose(features, expected)

    energy = features.square().sum()
    (vxc,) = torch.autograd.grad(energy, dm, create_graph=True)
    (hvp,) = torch.autograd.grad(vxc, dm, torch.ones_like(dm))

    inactive_ao_indices = np.setdiff1d(np.arange(carbon.nao_nr()), active_ao_indices)
    assert vxc.shape == dm.shape
    assert hvp.shape == dm.shape
    assert torch.count_nonzero(vxc[inactive_ao_indices]) == 0
    assert torch.count_nonzero(vxc[:, inactive_ao_indices]) == 0
    assert torch.count_nonzero(hvp[inactive_ao_indices]) == 0
    assert torch.count_nonzero(hvp[:, inactive_ao_indices]) == 0


def test_cpu_all_active_block_uses_dense_sentinel(
    carbon: gto.Mole, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Represent an all-active block by ``None`` and a later sparse block by indices."""
    block_size = dft.gen_grid.BLKSIZE
    ngrids = 2 * block_size
    grids = dft.Grids(carbon)
    grids.coords = np.zeros((ngrids, 3))
    grids.weights = np.ones(ngrids)
    screen_index = np.zeros((2, carbon.nbas), dtype=np.uint8)
    screen_index[0] = 1
    screen_index[1, 0] = 1
    grids.non0tab = screen_index
    ao_values = np.ones((ngrids, carbon.nao_nr()))

    class FakeNumInt:
        def block_loop(
            self, *args: object, **kwargs: object
        ) -> Iterator[tuple[np.ndarray, None, np.ndarray, np.ndarray]]:
            for start in range(0, ngrids, block_size):
                grid_slice = slice(start, start + block_size)
                yield (
                    ao_values[grid_slice],
                    None,
                    grids.weights[grid_slice],
                    grids.coords[grid_slice],
                )

    monkeypatch.setattr(dft.numint, "NumInt", FakeNumInt)
    feature_function = MGGAFeatureFunction(FeatureSpec([Feature.DENSITY]))

    blocks = list(_CPUAOBlockLoop(carbon, grids, feature_function, block_size))

    assert len(blocks) == 2
    assert blocks[0].active_ao_indices is None
    assert blocks[0].ao_values.shape == (carbon.nao_nr(), block_size)
    expected_sparse_indices = torch.as_tensor(
        _active_cpu_ao_indices(carbon, screen_index[1:]), dtype=torch.long
    )
    torch.testing.assert_close(blocks[1].active_ao_indices, expected_sparse_indices)
    assert blocks[1].ao_values.shape == (expected_sparse_indices.numel(), block_size)


def test_cpu_no_active_aos_returns_full_zero_derivatives(
    carbon: gto.Mole, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use an empty screen mask and verify full-size zero features, VXC, and HVP."""
    ngrids = dft.gen_grid.BLKSIZE
    grids = dft.Grids(carbon)
    grids.coords = np.zeros((ngrids, 3))
    grids.weights = np.ones(ngrids)
    ao = np.ones((ngrids, carbon.nao_nr()))
    screen_index = np.zeros((1, carbon.nbas), dtype=np.uint8)
    grids.non0tab = screen_index

    class FakeNumInt:
        def block_loop(
            self, *args: object, **kwargs: object
        ) -> Iterator[tuple[np.ndarray, None, np.ndarray, np.ndarray]]:
            assert kwargs["non0tab"] is screen_index
            yield ao, None, grids.weights, grids.coords

    monkeypatch.setattr(dft.numint, "NumInt", FakeNumInt)
    feature_function = MGGAFeatureFunction(FeatureSpec([Feature.DENSITY]))
    dm = torch.eye(carbon.nao_nr(), dtype=torch.float64).requires_grad_()

    features = evaluate_ao_features_blockwise(
        dm, carbon, grids, feature_function, ngrids, False
    )
    (vxc,) = torch.autograd.grad(features.square().sum(), dm, create_graph=True)
    (hvp,) = torch.autograd.grad(vxc, dm, torch.ones_like(dm))

    assert features.shape == (1, ngrids)
    assert vxc.shape == dm.shape
    assert hvp.shape == dm.shape
    assert torch.count_nonzero(features) == 0
    assert torch.count_nonzero(vxc) == 0
    assert torch.count_nonzero(hvp) == 0


def _minimal_atom_grid(mol: gto.Mole) -> SkalaGrids:
    grids = SkalaGrids(mol)
    grids.level = 0
    grids.alignment = 1
    return grids.build(sort_grids=False)


def test_atom_major_features_require_skala_grids(carbon: gto.Mole) -> None:
    """Reject backend grids that do not preserve Skala's atom-major layout."""
    integrator = XCIntegrator(QuadraticFunctional())
    grids = dft.Grids(carbon)
    dm = torch.eye(carbon.nao_nr(), dtype=torch.float64)

    with pytest.raises(TypeError, match=r"requires .*\.SkalaGrids"):
        integrator(carbon, grids, dm)
    with pytest.raises(TypeError, match=r"requires .*\.SkalaGrids"):
        integrator.gen_response(carbon, grids, dm)


def test_skala_grids_invalidate_spatial_layout(carbon: gto.Mole) -> None:
    """Invalidate and rebuild a cached spatial layout after grid state changes."""
    integrator = XCIntegrator(QuadraticFunctional())
    grids = _minimal_atom_grid(carbon)

    layout = integrator._get_spatial_grid_layout(carbon, grids)
    assert integrator._get_spatial_grid_layout(carbon, grids) is layout

    grids.reset()
    assert grids._spatial_grid_layout is None
    grids.level = 0
    grids.alignment = 1
    grids.build(sort_grids=False)
    rebuilt_layout = integrator._get_spatial_grid_layout(carbon, grids)
    assert rebuilt_layout is not layout

    grids.cutoff /= 10
    assert grids._spatial_grid_layout is None


def test_numint_reset_does_not_clear_grid_spatial_layout(carbon: gto.Mole) -> None:
    """Preserve the grid-owned spatial layout when resetting a NumInt instance."""
    numint: _NumPyNumInt = SkalaNumInt(QuadraticFunctional())
    grids = _minimal_atom_grid(carbon)
    spatial_grid_layout = prepare_spatial_grid_layout(
        carbon,
        grids,
        block_size=dft.gen_grid.BLKSIZE,
        device=torch.device("cpu"),
    )
    grids._spatial_grid_layout = spatial_grid_layout

    assert numint.reset() is numint
    assert grids._spatial_grid_layout is spatial_grid_layout


@pytest.mark.parametrize(
    ("atom", "spin", "mean_field_factory", "integration_method"),
    [
        pytest.param(
            "H 0 0 0; H 0 0 0.74",
            0,
            dft.RKS,
            SkalaNumInt.nr_rks,
            id="rks",
        ),
        pytest.param(
            "H 0 0 0",
            1,
            dft.UKS,
            SkalaNumInt.nr_uks,
            id="uks",
        ),
    ],
)
@pytest.mark.parametrize(
    ("result_index", "rtol", "atol"),
    [
        pytest.param(0, 1e-10, 1e-11, id="electron-count"),
        pytest.param(1, 1e-9, 1e-10, id="energy"),
        pytest.param(2, 1e-8, 1e-10, id="potential"),
    ],
)
def test_cpu_rks_uks_dense_screened_equivalence(
    load_functional_cached: Callable[..., ExcFunctionalBase | str],
    atom: str,
    spin: int,
    mean_field_factory: Callable[[gto.Mole], Any],
    integration_method: Callable[..., tuple[float | np.ndarray, float, np.ndarray]],
    result_index: int,
    rtol: float,
    atol: float,
) -> None:
    """Match dense and screened CPU results for restricted and unrestricted spins."""
    mol = gto.M(atom=atom, basis="sto-3g", spin=spin, verbose=0)
    mean_field = mean_field_factory(mol)

    functional = load_functional_cached("skala-1.1")
    assert isinstance(functional, ExcFunctionalBase)
    numint: _NumPyNumInt = SkalaNumInt(functional)
    grids = _minimal_atom_grid(mol)
    dm = mean_field.get_init_guess()

    with force_ao_screening(False):
        dense = integration_method(numint, mol, grids, None, dm)

    with force_ao_screening(True):
        screened = integration_method(numint, mol, grids, None, dm)

    assert np.allclose(
        dense[result_index], screened[result_index], rtol=rtol, atol=atol
    )


def test_cpu_quadratic_dense_screened_equivalence_heteronuclear() -> None:
    """Match dense and screened quadratic integration on a heteronuclear grid."""
    mol = gto.M(atom="H 0 0 0; F 0 0 0.92", basis="sto-3g", spin=0, verbose=0)
    grids = _minimal_atom_grid(mol)
    numint: _NumPyNumInt = SkalaNumInt(QuadraticFunctional())
    dm = dft.RKS(mol).get_init_guess()

    with force_ao_screening(False):
        dense = numint.nr_rks(mol, grids, None, dm)

    with force_ao_screening(True):
        screened = numint.nr_rks(mol, grids, None, dm)

    for dense_value, screened_value in zip(dense, screened, strict=True):
        assert np.allclose(
            cast(float | np.ndarray, dense_value),
            cast(float | np.ndarray, screened_value),
            rtol=1e-10,
            atol=1e-11,
        )


@pytest.mark.parametrize(
    ("features", "energy_feature"),
    [
        pytest.param(
            [Feature.ATOMIC_GRID_SIZES, Feature.GRAD, Feature.GRID_WEIGHTS],
            Feature.GRAD,
            id="density",
        ),
        pytest.param(
            [Feature.ATOMIC_GRID_SIZES, Feature.DENSITY],
            Feature.DENSITY,
            id="grid-weights",
        ),
    ],
)
def test_screened_integration_adds_bookkeeping_features(
    features: list[Feature], energy_feature: Feature
) -> None:
    """Add density and grid weights needed for screened integration bookkeeping."""

    class BookkeepingFunctional(ExcFunctionalBase):
        def __init__(self) -> None:
            super().__init__()
            self.features = features

        def get_exc(self, mol: FeatureMap) -> torch.Tensor:
            return mol[energy_feature].square().sum()

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    grids = _minimal_atom_grid(mol)
    integrator = XCIntegrator(BookkeepingFunctional())
    dm = torch.as_tensor(dft.RKS(mol).get_init_guess())

    with force_ao_screening(False):
        dense = integrator(mol, grids, dm.clone())
    with force_ao_screening(True):
        screened = integrator(mol, grids, dm.clone())

    torch.testing.assert_close(screened.electron_count, dense.electron_count)
    torch.testing.assert_close(screened.energy, dense.energy)
    torch.testing.assert_close(screened.potential, dense.potential)


def test_cpu_response_dense_screened_equivalence() -> None:
    """Match dense and screened CPU Hessian actions for a quadratic functional."""
    mol = gto.M(atom="H 0 0 0; F 0 0 0.92", basis="sto-3g", spin=0, verbose=0)
    grids = _minimal_atom_grid(mol)
    ks = FakeKS(mol, grids)
    numint: _NumPyNumInt = SkalaNumInt(QuadraticFunctional())
    mo_coeff = np.eye(mol.nao_nr())
    mo_occ = np.ones(mol.nao_nr())
    dm1 = np.arange(mol.nao_nr() ** 2, dtype=np.float64).reshape(
        mol.nao_nr(), mol.nao_nr()
    )
    dm1 += dm1.T

    with force_ao_screening(False):
        dense_response = numint.gen_response(mo_coeff, mo_occ, ks=ks)

    with force_ao_screening(True):
        screened_response = numint.gen_response(mo_coeff, mo_occ, ks=ks)

    assert np.allclose(
        dense_response(dm1), screened_response(dm1), rtol=1e-10, atol=1e-11
    )


@pytest.mark.parametrize("screened", [False, True], ids=["dense", "screened"])
def test_response_of_linear_functional_is_zero(screened: bool) -> None:
    """Return a zero Hessian action when the XC potential is constant.

    Energy is linear in the density-derived feature, so its first derivative is
    independent of the density matrix and its directional derivative is zero. The
    dense case exercises the constant-potential guard; the screened case exercises
    the corresponding constant chunk-gradient guard.
    """

    class LinearFunctional(ExcFunctionalBase):
        def __init__(self) -> None:
            super().__init__()
            self.features = [
                Feature.ATOMIC_GRID_SIZES,
                Feature.DENSITY,
                Feature.GRID_WEIGHTS,
            ]

        def get_exc(self, mol: FeatureMap) -> torch.Tensor:
            return (mol[Feature.DENSITY] * mol[Feature.GRID_WEIGHTS]).sum()

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    grids = _minimal_atom_grid(mol)
    integrator = XCIntegrator(LinearFunctional())
    dm = torch.as_tensor(dft.RKS(mol).get_init_guess())

    with force_ao_screening(screened):
        response = integrator.gen_response(mol, grids, dm)
        hessian_action = response(torch.ones_like(dm))

    torch.testing.assert_close(hessian_action, torch.zeros_like(dm))


@pytest.mark.parametrize("func_deriv", [1, 2])
def test_screened_ao_traversals_are_independent_of_model_chunking(
    monkeypatch: pytest.MonkeyPatch,
    func_deriv: int,
) -> None:
    """Keep global screened AO traversal counts independent of model chunk count."""
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    grids = _minimal_atom_grid(mol)
    atom_grid_size = grids.weights.size // mol.natm
    monkeypatch.setattr(
        model_chunking_module,
        "estimate_max_model_atoms_per_chunk",
        lambda *args, **kwargs: {atom_grid_size: 1},
    )

    forward_calls = 0
    backward_calls = 0
    original_apply = ao_evaluation_module._BlockwiseAOFeatureOperator.apply

    def counting_apply(*args: object) -> torch.Tensor:
        nonlocal forward_calls, backward_calls
        if args[-1]:
            backward_calls += 1
        else:
            forward_calls += 1
        result = original_apply(*args)  # type: ignore[no-untyped-call]
        assert isinstance(result, torch.Tensor)
        return result

    monkeypatch.setattr(
        ao_evaluation_module._BlockwiseAOFeatureOperator,
        "apply",
        counting_apply,
    )
    functional = QuadraticFunctional()
    numint: _NumPyNumInt = SkalaNumInt(functional)

    with force_ao_screening(True):
        if func_deriv == 1:
            dm = dft.RKS(mol).get_init_guess()
            numint.nr_rks(mol, grids, None, dm)
            assert forward_calls == 1
        else:
            ks = FakeKS(mol, grids)
            response = numint.gen_response(
                np.eye(mol.nao_nr()), np.ones(mol.nao_nr()), ks=ks
            )
            response(np.eye(mol.nao_nr()))
            assert forward_calls == 2

    assert backward_calls == 1
