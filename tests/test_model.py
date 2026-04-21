# SPDX-License-Identifier: MIT

"""Snapshot regression tests for SkalaFunctional components.

Each test captures deterministic numerical output using torch.manual_seed(42)
on CPU. Any change that alters model output will be caught.

RNG state is forked via an autouse fixture so that seeding inside tests
does not mutate the global RNG visible to other tests in the process.
"""

import math
from collections.abc import Iterator

import pytest
import torch

from skala.functional import ExcFunctionalBase, load_functional
from skala.functional.model import (
    ANGSTROM_TO_BOHR,
    ExpRadialScaleModel,
    NonLocalModel,
    O3Linear,
    SemiLocalFeatures,
    SkalaFunctional,
    TensorProduct,
    _prepare_features_raw,
)
from skala.functional.utils.irreps import Irreps


@pytest.fixture(autouse=True)
def _isolated_rng() -> Iterator[None]:
    """Fork the PyTorch RNG so manual_seed calls inside tests don't leak."""
    with torch.random.fork_rng():
        yield


def exp_radial_func(dist: torch.Tensor, num_basis: int, dim: int = 3) -> torch.Tensor:
    """Legacy standalone version, kept here for snapshot regression testing."""
    min_std = 0.32 * ANGSTROM_TO_BOHR / 2
    max_std = 2.32 * ANGSTROM_TO_BOHR / 2
    s = torch.linspace(min_std, max_std, num_basis, device=dist.device)
    temps = 2 * s**2
    x2 = dist[..., None] ** 2
    return (
        torch.exp(-x2 / temps) * 2 / dim * x2 / temps / (math.pi * temps) ** (0.5 * dim)
    )


def make_mol(
    num_atoms: int,
    grid_per_atom: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    total_grid = num_atoms * grid_per_atom
    return {
        "density": torch.randn(2, total_grid, dtype=dtype, device=device),
        "grad": torch.randn(2, 3, total_grid, dtype=dtype, device=device),
        "kin": torch.randn(2, total_grid, dtype=dtype, device=device),
        "grid_coords": torch.randn(total_grid, 3, dtype=dtype, device=device),
        "grid_weights": torch.randn(total_grid, dtype=dtype, device=device).abs(),
        "atomic_grid_weights": torch.randn(
            total_grid, dtype=dtype, device=device
        ).abs(),
        "atomic_grid_sizes": torch.tensor(
            [grid_per_atom] * num_atoms, dtype=torch.int64, device=device
        ),
        "coarse_0_atomic_coords": torch.randn(num_atoms, 3, dtype=dtype, device=device),
        "atomic_grid_size_bound_shape": torch.zeros(
            grid_per_atom, 0, dtype=torch.int64, device=device
        ),
    }


def make_mol_variable_grid(
    atomic_grid_sizes: list[int],
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    """Create a mol dict with variable grid sizes per atom."""
    sizes = torch.tensor(atomic_grid_sizes, dtype=torch.int64, device=device)
    num_atoms = len(atomic_grid_sizes)
    total_grid = sum(atomic_grid_sizes)
    size_bound = max(atomic_grid_sizes)
    return {
        "density": torch.randn(2, total_grid, dtype=dtype, device=device),
        "grad": torch.randn(2, 3, total_grid, dtype=dtype, device=device),
        "kin": torch.randn(2, total_grid, dtype=dtype, device=device),
        "grid_coords": torch.randn(total_grid, 3, dtype=dtype, device=device),
        "grid_weights": torch.randn(total_grid, dtype=dtype, device=device).abs(),
        "atomic_grid_weights": torch.randn(
            total_grid, dtype=dtype, device=device
        ).abs(),
        "atomic_grid_sizes": sizes,
        "coarse_0_atomic_coords": torch.randn(num_atoms, 3, dtype=dtype, device=device),
        "atomic_grid_size_bound_shape": torch.zeros(
            size_bound, 0, dtype=torch.int64, device=device
        ),
    }


def small_model() -> SkalaFunctional:
    return SkalaFunctional(
        num_mid_layers=1,
        num_non_local_layers=1,
        non_local_hidden_nf=3,
        correlation=1,
    )


def test_prepare_features_raw_snapshot() -> None:
    torch.manual_seed(42)
    model = small_model()
    torch.manual_seed(42)
    mol = make_mol(4, 10)
    packed = model.pack_features(mol)
    out = _prepare_features_raw(packed)

    assert out.shape == (10, 4, 7)
    torch.testing.assert_close(
        out.sum(),
        torch.tensor(1.104726756002241e01, dtype=torch.float64),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        out.abs().sum(),
        torch.tensor(2.873065774435380e02, dtype=torch.float64),
        rtol=1e-5,
        atol=1e-5,
    )
    expected_first = torch.tensor(
        [
            -1.2054195578684468,
            -1.2484940336629657,
            0.9278196025135572,
            1.1062339879348806,
            -0.00458365814011141,
            -4.091211863274872,
            1.9987709853481832,
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(out[0, 0, :], expected_first, rtol=1e-5, atol=1e-5)


def test_prepare_features_snapshot() -> None:
    torch.manual_seed(42)
    model = small_model()
    torch.manual_seed(42)
    mol = make_mol(4, 10)
    packed = model.pack_features(mol)
    semi_local = SemiLocalFeatures()
    features_ab, features_ba = semi_local(packed)

    assert features_ab.shape == (10, 4, 7)
    assert features_ba.shape == (10, 4, 7)
    torch.testing.assert_close(
        features_ab.sum(),
        torch.tensor(1.104726756002241e01, dtype=torch.float64),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        features_ba.sum(),
        torch.tensor(1.104726756002241e01, dtype=torch.float64),
        rtol=1e-5,
        atol=1e-5,
    )
    # features_ba is the column-swapped version of features_ab
    expected_ba = torch.stack(
        [features_ab[..., i] for i in [1, 0, 3, 2, 5, 4, 6]], dim=-1
    )
    torch.testing.assert_close(features_ba, expected_ba, rtol=0, atol=0)


def test_pack_features_snapshot() -> None:
    torch.manual_seed(42)
    model = small_model()
    torch.manual_seed(42)
    mol = make_mol(4, 10)
    packed = model.pack_features(mol)

    assert packed["density"].shape == (2, 10, 4)
    assert packed["kin"].shape == (2, 10, 4)
    assert packed["grad"].shape == (2, 3, 10, 4)
    assert packed["grid_coords"].shape == (10, 4, 3)
    assert packed["atomic_grid_weights"].shape == (10, 4)
    assert packed["coarse_0_atomic_coords"].shape == (4, 3)

    torch.testing.assert_close(
        packed["density"].sum(),
        torch.tensor(1.020635438470402e01, dtype=torch.float64),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        packed["atomic_grid_weights"].sum(),
        torch.tensor(4.032819661608873e01, dtype=torch.float64),
        rtol=1e-5,
        atol=1e-5,
    )


def test_exp_radial_func_snapshot() -> None:
    torch.manual_seed(42)
    dist = torch.randn(5, 3, dtype=torch.float64).abs()
    out = exp_radial_func(dist, num_basis=16)

    assert out.shape == (5, 3, 16)
    torch.testing.assert_close(
        out.sum(),
        torch.tensor(6.552108459223353e00, dtype=torch.float64),
        rtol=1e-5,
        atol=1e-5,
    )
    # All values should be non-negative (Gaussian-based radial function)
    assert (out >= 0).all()

    # ExpRadialScaleModel module must produce identical output
    radial_basis = ExpRadialScaleModel(embedding_size=16).double()
    out_module = radial_basis(dist.unsqueeze(-1) ** 2)
    torch.testing.assert_close(out_module, out, rtol=1e-6, atol=1e-7)


def test_tensor_product_snapshot() -> None:
    torch.manual_seed(42)
    irreps_in1 = Irreps("3x0e")
    irreps_in2 = Irreps("1x0e+1x1e")
    irreps_out = Irreps("3x0e+3x1e")
    tp = TensorProduct(irreps_in1, irreps_in2, irreps_out)

    torch.manual_seed(42)
    x1 = torch.randn(5, 4, irreps_in1.dim, dtype=torch.float32)
    x2 = torch.randn(5, 4, irreps_in2.dim, dtype=torch.float32)
    out = tp(x1, x2)

    assert out.shape == (5, 4, 12)
    torch.testing.assert_close(
        out.sum(),
        torch.tensor(5.987227916717529e00, dtype=torch.float32),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        out.abs().sum(),
        torch.tensor(1.092445983886719e02, dtype=torch.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_o3_linear_snapshot() -> None:
    torch.manual_seed(42)
    irreps_in = Irreps("3x0e+3x1e")
    irreps_out = Irreps("3x0e+3x1e")
    linear = O3Linear(irreps_in, irreps_out)

    torch.manual_seed(42)
    x = torch.randn(5, irreps_in.dim, dtype=torch.float32)
    out = linear(x)

    assert out.shape == (5, 12)
    torch.testing.assert_close(
        out.sum(),
        torch.tensor(1.546258926391602e01, dtype=torch.float32),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        out.abs().sum(),
        torch.tensor(5.258611297607422e01, dtype=torch.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_nonlocal_model_snapshot() -> None:
    torch.manual_seed(42)
    sph_irreps = Irreps.spherical_harmonics(1, p=1)
    nlm = NonLocalModel(
        input_nf=256,
        hidden_nf=3,
        lmax=1,
        edge_irreps=sph_irreps,
        coarse_linear_type="decomp-identity",
        correlation=1,
    ).float()

    torch.manual_seed(42)
    num_fine, num_coarse = 10, 4
    h = torch.randn(num_fine, num_coarse, 256, dtype=torch.float32)
    distance_ft = torch.randn(num_fine, num_coarse, 3, dtype=torch.float32).abs()
    direction_ft = torch.randn(num_fine, num_coarse, 4, dtype=torch.float32)
    grid_weights = torch.randn(num_fine, num_coarse, dtype=torch.float32).abs()
    exp_m1_rho = torch.randn(num_fine, num_coarse, 1, dtype=torch.float32).abs()
    out = nlm(h, distance_ft, direction_ft, grid_weights, exp_m1_rho)

    assert out.shape == (10, 4, 256)
    torch.testing.assert_close(
        out.sum(),
        torch.tensor(7.943189086914062e02, dtype=torch.float32),
        rtol=1e-4,
        atol=1e-1,
    )
    torch.testing.assert_close(
        out.abs().sum(),
        torch.tensor(2.372190673828125e03, dtype=torch.float32),
        rtol=1e-4,
        atol=1e-1,
    )


def test_get_exc_density_snapshot_4atoms() -> None:
    torch.manual_seed(42)
    model = small_model()
    torch.manual_seed(42)
    mol = make_mol(4, 10)
    out = model.get_exc_density(mol)

    assert out.shape == (40,)
    torch.testing.assert_close(
        out.sum(),
        torch.tensor(-3.425105949459996e01, dtype=torch.float64),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        out.abs().sum(),
        torch.tensor(3.425105949459996e01, dtype=torch.float64),
        rtol=1e-5,
        atol=1e-5,
    )


def test_get_exc_density_snapshot_17atoms() -> None:
    torch.manual_seed(42)
    model = small_model()
    torch.manual_seed(42)
    mol = make_mol(17, 10)
    out = model.get_exc_density(mol)

    assert out.shape == (170,)
    torch.testing.assert_close(
        out.sum(),
        torch.tensor(-1.321246666755034e02, dtype=torch.float64),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        out.abs().sum(),
        torch.tensor(1.321246666755034e02, dtype=torch.float64),
        rtol=1e-5,
        atol=1e-5,
    )


def test_get_exc_snapshot() -> None:
    torch.manual_seed(42)
    model = small_model()
    torch.manual_seed(42)
    mol = make_mol(4, 10)
    out = model.get_exc(mol)

    torch.testing.assert_close(
        out,
        torch.tensor(-2.821835255217404e01, dtype=torch.float64),
        rtol=1e-5,
        atol=1e-5,
    )


def test_get_exc_density_variable_grid_sizes() -> None:
    """Test that get_exc_density returns the correct unpadded shape and values with variable grid sizes."""
    torch.manual_seed(42)
    model = small_model()
    torch.manual_seed(42)
    sizes = [5, 10, 8, 3]
    mol = make_mol_variable_grid(sizes)
    out = model.get_exc_density(mol)

    assert out.shape == (sum(sizes),), (
        f"Expected shape ({sum(sizes)},) but got {out.shape}. "
        "get_exc_density should return (num_grid_points,), not padded shape."
    )
    torch.testing.assert_close(
        out.sum(),
        torch.tensor(-1.642121553088128e01, dtype=torch.float64),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        out.abs().sum(),
        torch.tensor(1.642121553088128e01, dtype=torch.float64),
        rtol=1e-5,
        atol=1e-5,
    )


def test_get_exc_variable_grid_sizes() -> None:
    """Test that get_exc still works correctly with variable grid sizes."""
    torch.manual_seed(42)
    model = small_model()
    torch.manual_seed(42)
    mol = make_mol_variable_grid([5, 10, 8, 3])
    out = model.get_exc(mol)

    torch.testing.assert_close(
        out,
        torch.tensor(-1.121880984584683e01, dtype=torch.float64),
        rtol=1e-5,
        atol=1e-5,
    )


def test_traced_functional_and_loaded_functional_are_equal() -> None:
    # This test ensures that the traced functional and the loaded functional
    # give the same output for the same input.

    traced_model = load_functional("skala-1.1")
    assert isinstance(traced_model, ExcFunctionalBase)

    clean_state_dict = {
        k.replace("_traced_model.", ""): v for k, v in traced_model.state_dict().items()
    }

    model = SkalaFunctional(lmax=3, num_non_local_layers=3, num_mid_layers=4)
    model.load_state_dict(clean_state_dict, strict=True)

    # Create a dummy input using the same helper as the other tests
    torch.manual_seed(42)
    features_dict = make_mol(num_atoms=1, grid_per_atom=10)

    original_output = model.get_exc(features_dict)
    traced_model_output = traced_model.get_exc(features_dict)

    # Compare outputs
    assert torch.allclose(original_output, traced_model_output, atol=1e-5)
