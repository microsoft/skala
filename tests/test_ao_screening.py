from collections.abc import Callable, Iterator

import numpy as np
import pytest
import torch
from pyscf import dft, gto
from pyscf.dft import numint as pyscf_numint

from skala.functional.base import ExcFunctionalBase
from skala.pyscf import features as features_module
from skala.pyscf import numint as numint_module
from skala.pyscf.features import (
    ChunkEvalBackward,
    ChunkEvalForward,
    MGGAFeatureFunction,
    _active_cpu_aos,
    _prepare_spatially_sorted_grids,
    _spatial_grid_permutations,
)
from skala.pyscf.numint import SkalaNumInt, _should_screen_aos


@pytest.fixture
def carbon() -> gto.Mole:
    return gto.M(atom="C 0 0 0", basis="sto-3g", spin=2, verbose=0)


@pytest.mark.parametrize(
    ("switch_offset", "expected"), [(1, False), (0, False), (-1, True)]
)
def test_should_screen_aos_at_crossover(
    carbon: gto.Mole,
    monkeypatch: pytest.MonkeyPatch,
    switch_offset: int,
    expected: bool,
) -> None:
    monkeypatch.setattr(
        pyscf_numint,
        "SWITCH_SIZE",
        carbon.nao_nr() + switch_offset,
    )

    assert _should_screen_aos(carbon) is expected


def test_active_cpu_aos(carbon: gto.Mole) -> None:
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

    assert np.array_equal(_active_cpu_aos(carbon, screen_index), expected)

    empty = _active_cpu_aos(carbon, np.zeros_like(screen_index))
    assert empty.dtype == np.int64
    assert empty.size == 0


@pytest.mark.parametrize(("ngrids", "block_size"), [(0, 4), (3, 4), (8, 4), (10, 4)])
def test_spatial_grid_permutations_restore_original_order(
    ngrids: int, block_size: int
) -> None:
    coords = np.arange(3 * ngrids, dtype=np.float64).reshape(ngrids, 3)

    forward, inverse = _spatial_grid_permutations(coords, block_size)

    assert np.array_equal(np.sort(forward), np.arange(ngrids))
    assert np.array_equal(coords[forward][inverse], coords)
    assert all(
        len(forward[start : start + block_size]) == block_size
        for start in range(0, ngrids - block_size + 1, block_size)
    )
    assert len(forward) % block_size == ngrids % block_size


def test_spatial_grid_permutations_group_interleaved_clusters() -> None:
    block_size = 3
    labels = np.tile(np.arange(4), block_size)
    offsets = np.repeat(np.arange(block_size), 4)
    coords = np.column_stack(
        (100.0 * labels + offsets, np.zeros(labels.size), np.zeros(labels.size))
    )

    forward, _ = _spatial_grid_permutations(coords, block_size)

    grouped_labels = labels[forward].reshape(-1, block_size)
    assert np.all(grouped_labels == grouped_labels[:, :1])


def test_prepare_spatially_sorted_cpu_grids(
    carbon: gto.Mole, monkeypatch: pytest.MonkeyPatch
) -> None:
    coords = np.arange(18, dtype=np.float64).reshape(6, 3)
    weights = np.arange(6, dtype=np.float64) + 10
    grids = dft.Grids(carbon)
    grids.coords = coords
    grids.weights = weights
    forward = np.array([4, 2, 0, 5, 3, 1], dtype=np.int64)
    inverse = np.argsort(forward)
    non0tab = np.ones((1, carbon.nbas), dtype=np.uint8)
    partition_calls = 0

    def fake_spatial_grid_permutations(
        coords_arg: np.ndarray, block_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        nonlocal partition_calls
        partition_calls += 1
        assert coords_arg is grids.coords
        assert block_size == 2
        return forward, inverse

    monkeypatch.setattr(
        features_module,
        "_spatial_grid_permutations",
        fake_spatial_grid_permutations,
    )

    screen_index_calls = 0

    def fake_make_screen_index(
        mol_arg: gto.Mole, sorted_coords: np.ndarray, cutoff: float
    ) -> np.ndarray:
        nonlocal screen_index_calls
        screen_index_calls += 1
        assert mol_arg is carbon
        assert np.array_equal(sorted_coords, coords[forward])
        assert cutoff == grids.cutoff
        return non0tab

    monkeypatch.setattr(dft.gen_grid, "make_screen_index", fake_make_screen_index)

    sorted_grids, actual_forward, actual_inverse = _prepare_spatially_sorted_grids(
        carbon, grids, block_size=2, gpu=False
    )

    assert sorted_grids is not grids
    assert np.array_equal(grids.coords, coords)
    assert np.array_equal(grids.weights, weights)
    assert np.array_equal(sorted_grids.coords, coords[forward])
    assert np.array_equal(sorted_grids.weights, weights[forward])
    assert sorted_grids.non0tab is non0tab
    assert actual_forward is forward
    assert actual_inverse is inverse

    cached_grids, cached_forward, cached_inverse = _prepare_spatially_sorted_grids(
        carbon, grids, block_size=2, gpu=False
    )

    assert cached_grids is sorted_grids
    assert cached_forward is forward
    assert cached_inverse is inverse
    assert partition_calls == 1
    assert screen_index_calls == 1

    grids.coords = grids.coords.copy()
    rebuilt_grids, _, _ = _prepare_spatially_sorted_grids(
        carbon, grids, block_size=2, gpu=False
    )

    assert rebuilt_grids is not sorted_grids
    assert partition_calls == 2
    assert screen_index_calls == 2


class QuadraticDensityFunctional(ExcFunctionalBase):
    def __init__(self) -> None:
        super().__init__()
        self.features = ["atomic_grid_sizes", "density", "grid_weights"]

    def get_exc(self, mol: dict[str, torch.Tensor]) -> torch.Tensor:
        return (mol["density"].square() * mol["grid_weights"]).sum()


class FakeKS:
    def __init__(self, mol: gto.Mole, grids: object | None = None) -> None:
        self.mol = mol
        self.grids = grids or object()
        self.max_memory = 100

    def make_rdm1(self, mo_coeff: np.ndarray, mo_occ: np.ndarray) -> np.ndarray:
        return np.eye(self.mol.nao_nr())

    def get_j(self, mol: gto.Mole, dm: np.ndarray, hermi: int) -> np.ndarray:
        return np.zeros_like(dm)


@pytest.mark.parametrize("expected", [False, True])
def test_first_and_second_order_use_same_screening_decision(
    carbon: gto.Mole,
    monkeypatch: pytest.MonkeyPatch,
    expected: bool,
) -> None:
    switch_size = carbon.nao_nr() - 1 if expected else carbon.nao_nr()
    monkeypatch.setattr(pyscf_numint, "SWITCH_SIZE", switch_size)
    routes: list[str] = []

    def fake_generate_features(
        mol: gto.Mole,
        dm: torch.Tensor,
        grids: object,
        features: set[str] | None = None,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        routes.append("dense")
        density = dm.square().sum().reshape(1).expand(2, 1) / 2
        return {
            "atomic_grid_sizes": torch.tensor([1]),
            "density": density,
            "grid_weights": torch.ones(1, dtype=dm.dtype),
        }

    class FakeGlobalScreenedFeatures:
        def __init__(self, dm: torch.Tensor) -> None:
            raw_features = dm.sum().reshape(1, 1)
            self.feature_function = MGGAFeatureFunction(
                with_density=True,
                with_grad=False,
                with_kin=False,
            )
            self.sorted_raw_features = raw_features
            self.atom_major_raw_features = raw_features
            self.forward_permutation = torch.tensor([0])
            self.chunks = [(slice(0, 1), slice(0, 1))]

        def atom_major_jvp(self, dm_tangent: torch.Tensor) -> torch.Tensor:
            return dm_tangent.sum().reshape(1, 1)

        def build_model_chunk(
            self,
            raw_features: torch.Tensor,
            atom_slice: slice,
            grid_slice: slice,
        ) -> dict[str, torch.Tensor]:
            assert atom_slice == slice(0, 1)
            assert grid_slice == slice(0, 1)
            return {
                "atomic_grid_sizes": torch.tensor([1]),
                "density": raw_features.expand(2, 1) / 2,
                "grid_weights": torch.ones(1, dtype=raw_features.dtype),
            }

    def fake_global_screened_features(
        mol: gto.Mole,
        dm: torch.Tensor,
        grids: object,
        features: set[str],
        func_deriv: int,
        **kwargs: object,
    ) -> FakeGlobalScreenedFeatures:
        routes.append("screened")
        return FakeGlobalScreenedFeatures(dm)

    monkeypatch.setattr(numint_module, "generate_features", fake_generate_features)
    monkeypatch.setattr(
        numint_module, "_global_screened_features", fake_global_screened_features
    )
    numint = SkalaNumInt(QuadraticDensityFunctional())
    dm = torch.eye(carbon.nao_nr(), dtype=torch.float64)
    grids = dft.Grids(carbon)
    grids.weights = np.ones(1)

    numint(carbon, grids, None, dm)

    ks = FakeKS(carbon, grids)
    response = numint.gen_response(
        np.eye(carbon.nao_nr()), np.ones(carbon.nao_nr()), ks=ks
    )
    result = response(np.eye(carbon.nao_nr()))

    assert result.shape == (carbon.nao_nr(), carbon.nao_nr())
    expected_route = "screened" if expected else "dense"
    assert routes == [expected_route, expected_route]


def test_cpu_screening_slices_and_scatters_full_derivatives(
    carbon: gto.Mole, monkeypatch: pytest.MonkeyPatch
) -> None:
    ngrids = dft.gen_grid.BLKSIZE
    grids = dft.Grids(carbon)
    grids.coords = np.zeros((ngrids, 3))
    grids.weights = np.ones(ngrids)

    ao = np.arange(ngrids * carbon.nao_nr(), dtype=np.float64).reshape(
        ngrids, carbon.nao_nr()
    )
    screen_index = np.zeros((1, carbon.nbas), dtype=np.uint8)
    screen_index[0, (0, -1)] = 1
    grids.non0tab = screen_index
    active_aos = _active_cpu_aos(carbon, screen_index)

    class FakeNumInt:
        def block_loop(
            self, *args: object, **kwargs: object
        ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
            assert kwargs["non0tab"] is screen_index
            yield ao, screen_index, grids.weights, grids.coords

    monkeypatch.setattr(dft.numint, "NumInt", FakeNumInt)

    feature_function = MGGAFeatureFunction(
        with_density=True, with_grad=False, with_kin=False
    )
    dm = torch.diag(
        torch.arange(1, carbon.nao_nr() + 1, dtype=torch.float64)
    ).requires_grad_()
    features = ChunkEvalForward.apply(  # type: ignore[no-untyped-call]
        dm, carbon, grids, feature_function, ngrids, False, False
    )

    ao_active = torch.from_numpy(ao[:, active_aos]).T
    dm_active = dm[..., active_aos[:, None], active_aos[None, :]]
    expected = torch.sum((dm_active @ ao_active) * ao_active, dim=0).unsqueeze(0)
    assert torch.allclose(features, expected)

    energy = features.square().sum()
    (vxc,) = torch.autograd.grad(energy, dm, create_graph=True)
    (hvp,) = torch.autograd.grad(vxc, dm, torch.ones_like(dm))

    inactive_aos = np.setdiff1d(np.arange(carbon.nao_nr()), active_aos)
    assert vxc.shape == dm.shape
    assert hvp.shape == dm.shape
    assert torch.count_nonzero(vxc[inactive_aos]) == 0
    assert torch.count_nonzero(vxc[:, inactive_aos]) == 0
    assert torch.count_nonzero(hvp[inactive_aos]) == 0
    assert torch.count_nonzero(hvp[:, inactive_aos]) == 0


def test_cpu_no_active_aos_returns_full_zero_derivatives(
    carbon: gto.Mole, monkeypatch: pytest.MonkeyPatch
) -> None:
    ngrids = dft.gen_grid.BLKSIZE
    grids = dft.Grids(carbon)
    grids.coords = np.zeros((ngrids, 3))
    grids.weights = np.ones(ngrids)
    ao = np.ones((ngrids, carbon.nao_nr()))
    screen_index = np.zeros((1, carbon.nbas), dtype=np.uint8)

    class FakeNumInt:
        def block_loop(
            self, *args: object, **kwargs: object
        ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
            yield ao, screen_index, grids.weights, grids.coords

    monkeypatch.setattr(dft.numint, "NumInt", FakeNumInt)
    feature_function = MGGAFeatureFunction(
        with_density=True, with_grad=False, with_kin=False
    )
    dm = torch.eye(carbon.nao_nr(), dtype=torch.float64).requires_grad_()

    features = ChunkEvalForward.apply(  # type: ignore[no-untyped-call]
        dm, carbon, grids, feature_function, ngrids, False, False
    )
    (vxc,) = torch.autograd.grad(features.square().sum(), dm, create_graph=True)
    (hvp,) = torch.autograd.grad(vxc, dm, torch.ones_like(dm))

    assert features.shape == (1, ngrids)
    assert vxc.shape == dm.shape
    assert hvp.shape == dm.shape
    assert torch.count_nonzero(features) == 0
    assert torch.count_nonzero(vxc) == 0
    assert torch.count_nonzero(hvp) == 0


def _minimal_atom_grid(mol: gto.Mole) -> dft.Grids:
    grids = dft.Grids(mol)
    grids.level = 0
    grids.alignment = 1
    return grids.build(sort_grids=False)


@pytest.mark.parametrize("unrestricted", [False, True])
def test_cpu_rks_uks_dense_screened_equivalence(
    monkeypatch: pytest.MonkeyPatch,
    load_functional_cached: Callable[..., ExcFunctionalBase | str],
    unrestricted: bool,
) -> None:
    if unrestricted:
        mol = gto.M(atom="H 0 0 0", basis="sto-3g", spin=1, verbose=0)
        mean_field = dft.UKS(mol)
    else:
        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", spin=0, verbose=0)
        mean_field = dft.RKS(mol)

    functional = load_functional_cached("skala-1.1")
    assert isinstance(functional, ExcFunctionalBase)
    numint = SkalaNumInt(functional)
    grids = _minimal_atom_grid(mol)
    dm = mean_field.get_init_guess()

    monkeypatch.setattr(pyscf_numint, "SWITCH_SIZE", mol.nao_nr())
    dense = (
        numint.nr_uks(mol, grids, None, dm)
        if unrestricted
        else numint.nr_rks(mol, grids, None, dm)
    )

    monkeypatch.setattr(pyscf_numint, "SWITCH_SIZE", mol.nao_nr() - 1)
    screened = (
        numint.nr_uks(mol, grids, None, dm)
        if unrestricted
        else numint.nr_rks(mol, grids, None, dm)
    )

    assert np.allclose(dense[0], screened[0], rtol=1e-10, atol=1e-11)
    assert np.isclose(dense[1], screened[1], rtol=1e-9, atol=1e-10)
    assert np.allclose(dense[2], screened[2], rtol=1e-8, atol=1e-10)


def test_cpu_response_dense_screened_equivalence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", spin=0, verbose=0)
    grids = _minimal_atom_grid(mol)
    ks = FakeKS(mol, grids)
    numint = SkalaNumInt(QuadraticDensityFunctional())
    mo_coeff = np.eye(mol.nao_nr())
    mo_occ = np.ones(mol.nao_nr())
    dm1 = np.arange(mol.nao_nr() ** 2, dtype=np.float64).reshape(
        mol.nao_nr(), mol.nao_nr()
    )
    dm1 += dm1.T

    monkeypatch.setattr(pyscf_numint, "SWITCH_SIZE", mol.nao_nr())
    dense_response = numint.gen_response(mo_coeff, mo_occ, ks=ks)

    monkeypatch.setattr(pyscf_numint, "SWITCH_SIZE", mol.nao_nr() - 1)
    screened_response = numint.gen_response(mo_coeff, mo_occ, ks=ks)

    assert np.allclose(
        dense_response(dm1), screened_response(dm1), rtol=1e-10, atol=1e-11
    )
    assert np.allclose(
        dense_response(dm1), screened_response(dm1), rtol=1e-10, atol=1e-11
    )


@pytest.mark.parametrize("func_deriv", [1, 2])
def test_global_screened_ao_traversals_are_independent_of_model_chunks(
    monkeypatch: pytest.MonkeyPatch,
    func_deriv: int,
) -> None:
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    grids = _minimal_atom_grid(mol)
    atom_grid_size = grids.weights.size // mol.natm
    monkeypatch.setattr(
        features_module,
        "estimate_max_grid_chunk_size",
        lambda *args, **kwargs: atom_grid_size,
    )
    monkeypatch.setattr(pyscf_numint, "SWITCH_SIZE", mol.nao_nr() - 1)

    forward_calls = 0
    backward_calls = 0
    original_forward_apply = ChunkEvalForward.apply
    original_backward_apply = ChunkEvalBackward.apply

    def counting_forward_apply(*args: object) -> torch.Tensor:
        nonlocal forward_calls
        forward_calls += 1
        return original_forward_apply(*args)

    def counting_backward_apply(*args: object) -> torch.Tensor:
        nonlocal backward_calls
        backward_calls += 1
        return original_backward_apply(*args)

    monkeypatch.setattr(ChunkEvalForward, "apply", counting_forward_apply)
    monkeypatch.setattr(ChunkEvalBackward, "apply", counting_backward_apply)
    functional = QuadraticDensityFunctional()
    numint = SkalaNumInt(functional)

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
