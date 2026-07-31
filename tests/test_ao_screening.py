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
    CPU_AO_SCREENING_BLOCK_SIZE,
    ChunkEvalForward,
    MGGAFeatureFunction,
    _active_cpu_aos,
    chunked_features,
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


@pytest.mark.parametrize("screen_aos", [False, True])
def test_chunked_features_routes_screening(
    carbon: gto.Mole,
    monkeypatch: pytest.MonkeyPatch,
    screen_aos: bool,
) -> None:
    ngrids = dft.gen_grid.BLKSIZE
    grids = dft.Grids(carbon)
    grids.coords = np.zeros((ngrids, 3))
    grids.weights = np.ones(ngrids)
    dm = torch.eye(carbon.nao_nr(), dtype=torch.float64)
    calls: list[str] = []

    monkeypatch.setattr(
        features_module,
        "get_grid_features",
        lambda *args, **kwargs: {
            "atomic_grid_sizes": torch.tensor([ngrids]),
            "grid_weights": torch.ones(ngrids, dtype=torch.float64),
        },
    )
    monkeypatch.setattr(
        features_module,
        "estimate_max_grid_chunk_size",
        lambda *args, **kwargs: ngrids,
    )
    monkeypatch.setattr(
        dft.gen_grid,
        "make_screen_index",
        lambda *args, **kwargs: np.ones((1, carbon.nbas), dtype=np.uint8),
    )

    def fake_non_chunk(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append("non_chunk")
        return torch.zeros((1, ngrids), dtype=torch.float64)

    def fake_chunk_eval(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append("ChunkEval")
        return torch.zeros((1, ngrids), dtype=torch.float64)

    monkeypatch.setattr(features_module, "non_chunk", fake_non_chunk)
    monkeypatch.setattr(ChunkEvalForward, "apply", fake_chunk_eval)

    list(
        chunked_features(
            carbon,
            dm,
            grids,
            {"atomic_grid_sizes", "density", "grid_weights"},
            func_deriv=1,
            screen_aos=screen_aos,
        )
    )

    assert calls == ["ChunkEval" if screen_aos else "non_chunk"]

    if screen_aos:
        assert CPU_AO_SCREENING_BLOCK_SIZE == 504


def test_cpu_screening_spatially_groups_each_atom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    atom_grid_size = dft.gen_grid.BLKSIZE
    ngrids = 2 * atom_grid_size
    coords = np.zeros((ngrids, 3))
    coords[:, 0] = np.arange(ngrids)
    weights = np.arange(ngrids, dtype=np.float64) + 100
    atomic_grid_weights = torch.arange(ngrids, dtype=torch.float64) + 200
    grids = dft.Grids(mol)
    grids.coords = coords
    grids.weights = weights
    grouped_slices: list[np.ndarray] = []

    monkeypatch.setattr(
        features_module,
        "get_grid_features",
        lambda *args, **kwargs: {
            "atomic_grid_sizes": torch.tensor([atom_grid_size, atom_grid_size]),
            "grid_coords": torch.from_numpy(coords.copy()),
            "grid_weights": torch.from_numpy(weights.copy()),
            "atomic_grid_weights": atomic_grid_weights,
        },
    )
    monkeypatch.setattr(
        features_module,
        "estimate_max_grid_chunk_size",
        lambda *args, **kwargs: ngrids,
    )

    def fake_group_grids(mol_arg: gto.Mole, atom_coords: np.ndarray) -> np.ndarray:
        assert mol_arg is mol
        grouped_slices.append(atom_coords.copy())
        return np.arange(atom_grid_size - 1, -1, -1)

    monkeypatch.setattr(dft.gen_grid, "arg_group_grids", fake_group_grids)

    sort_indices = np.concatenate(
        (
            np.arange(atom_grid_size - 1, -1, -1),
            np.arange(ngrids - 1, atom_grid_size - 1, -1),
        )
    )

    def fake_make_screen_index(
        mol_arg: gto.Mole, sorted_coords: np.ndarray, cutoff: float
    ) -> np.ndarray:
        assert mol_arg is mol
        assert np.array_equal(sorted_coords, coords[sort_indices])
        return np.ones((2, mol.nbas), dtype=np.uint8)

    monkeypatch.setattr(dft.gen_grid, "make_screen_index", fake_make_screen_index)

    def fake_chunk_eval(
        dm: torch.Tensor,
        mol_arg: gto.Mole,
        sorted_grids: dft.Grids,
        feature_function: MGGAFeatureFunction,
        block_size: int,
        compile_feature_function: bool,
        gpu: bool,
    ) -> torch.Tensor:
        assert mol_arg is mol
        assert feature_function.with_density
        assert block_size == CPU_AO_SCREENING_BLOCK_SIZE
        assert not compile_feature_function
        assert not gpu
        assert np.array_equal(sorted_grids.coords, coords[sort_indices])
        assert np.array_equal(sorted_grids.weights, weights[sort_indices])
        return torch.from_numpy(sorted_grids.coords[:, 0]).to(dm).unsqueeze(0)

    monkeypatch.setattr(ChunkEvalForward, "apply", fake_chunk_eval)

    (feature_chunk,) = list(
        chunked_features(
            mol,
            torch.eye(mol.nao_nr(), dtype=torch.float64),
            grids,
            {
                "atomic_grid_sizes",
                "atomic_grid_weights",
                "density",
                "grid_coords",
                "grid_weights",
            },
            func_deriv=1,
            screen_aos=True,
        )
    )

    assert len(grouped_slices) == 2
    assert np.array_equal(grouped_slices[0], coords[:atom_grid_size])
    assert np.array_equal(grouped_slices[1], coords[atom_grid_size:])
    assert torch.equal(
        feature_chunk["atomic_grid_sizes"],
        torch.tensor([atom_grid_size, atom_grid_size]),
    )
    assert torch.equal(
        feature_chunk["grid_coords"], torch.from_numpy(coords[sort_indices])
    )
    assert torch.equal(
        feature_chunk["grid_weights"], torch.from_numpy(weights[sort_indices])
    )
    assert torch.equal(
        feature_chunk["atomic_grid_weights"], atomic_grid_weights[sort_indices]
    )
    expected_density = torch.from_numpy(coords[sort_indices, 0]) / 2
    assert torch.equal(feature_chunk["density"][0], expected_density)
    assert torch.equal(feature_chunk["density"][1], expected_density)


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
    decisions: list[bool] = []

    def fake_chunked_features(
        mol: gto.Mole,
        dm: torch.Tensor,
        grids: object,
        features: set[str],
        func_deriv: int,
        *,
        screen_aos: bool,
        **kwargs: object,
    ) -> Iterator[dict[str, torch.Tensor]]:
        decisions.append(screen_aos)
        density = dm.square().sum().reshape(1).expand(2, 1) / 2
        yield {
            "atomic_grid_sizes": torch.tensor([1]),
            "density": density,
            "grid_weights": torch.ones(1, dtype=dm.dtype),
        }

    monkeypatch.setattr(numint_module, "chunked_features", fake_chunked_features)
    numint = SkalaNumInt(QuadraticDensityFunctional())
    dm = torch.eye(carbon.nao_nr(), dtype=torch.float64)

    numint(carbon, object(), None, dm)

    ks = FakeKS(carbon)
    response = numint.gen_response(
        np.eye(carbon.nao_nr()), np.ones(carbon.nao_nr()), ks=ks
    )
    result = response(np.eye(carbon.nao_nr()))

    assert result.shape == (carbon.nao_nr(), carbon.nao_nr())
    assert decisions == [expected, expected]


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
