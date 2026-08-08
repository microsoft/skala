import pytest
import torch

from skala.pyscf.memory_estimators import (
    estimate_global_raw_feature_buffer_memory,
    estimate_global_screened_buffer_memory,
    estimate_max_model_atoms_per_chunk,
    estimate_model_memory_per_grid_point,
)


@pytest.mark.parametrize(
    ("dm_shape", "func_deriv", "buffer_count"),
    [((10, 10), 1, 4), ((2, 10, 10), 1, 4), ((10, 10), 2, 5)],
)
def test_global_raw_feature_buffer_memory(
    dm_shape: tuple[int, ...], func_deriv: int, buffer_count: int
) -> None:
    dm = torch.zeros(dm_shape, dtype=torch.float64)
    nfeatures = 5
    ngrids = 123
    batch_size = dm.numel() // (dm.shape[-2] * dm.shape[-1])

    actual = estimate_global_raw_feature_buffer_memory(
        dm, nfeatures, ngrids, func_deriv
    )

    assert actual == buffer_count * batch_size * nfeatures * ngrids * 8


def test_global_raw_feature_buffer_memory_rejects_unsupported_order() -> None:
    with pytest.raises(ValueError, match="func_deriv 1 or 2"):
        estimate_global_raw_feature_buffer_memory(
            torch.eye(2, dtype=torch.float64), 1, 1, func_deriv=0
        )


def test_global_screened_buffer_memory_uses_atomic_grid_sizes() -> None:
    dm = torch.eye(10, dtype=torch.float64)
    atomic_grid_sizes = torch.tensor([10, 10, 20])

    actual = estimate_global_screened_buffer_memory(
        dm, nfeatures=5, atomic_grid_sizes=atomic_grid_sizes, func_deriv=1
    )

    raw_feature_bytes = 4 * 5 * 40 * 8
    dense_buffer_bytes = int(37.0 * 10**2)
    assert actual == raw_feature_bytes + dense_buffer_bytes


def test_model_atom_limits_are_estimated_per_atomic_grid_size() -> None:
    dm = torch.eye(10, dtype=torch.float64)
    atomic_grid_sizes = torch.tensor([10, 10, 20])

    actual = estimate_max_model_atoms_per_chunk(
        dm,
        atomic_grid_sizes=atomic_grid_sizes,
        nfeatures=5,
        max_memory_in_mb=10,
        safety_fraction=1.0,
        func_deriv=1,
    )

    available_memory = 10_000_000 - estimate_global_screened_buffer_memory(
        dm, 5, atomic_grid_sizes, 1
    )
    bytes_per_point = estimate_model_memory_per_grid_point(1)
    assert actual == {
        10: available_memory // (10 * bytes_per_point),
        20: available_memory // (20 * bytes_per_point),
    }


@pytest.mark.parametrize(
    ("func_deriv", "elements_per_point"),
    [(0, 5830), (1, 6680), (2, 24230)],
)
def test_model_memory_per_grid_point_depends_on_functional_derivative(
    func_deriv: int, elements_per_point: int
) -> None:
    assert estimate_model_memory_per_grid_point(func_deriv) == 8 * elements_per_point


@pytest.mark.parametrize("safety_fraction", [-0.1, 0.0, 1.1])
def test_model_grid_point_limit_rejects_invalid_safety_fraction(
    safety_fraction: float,
) -> None:
    with pytest.raises(
        ValueError, match="safety_fraction must be greater than 0 and at most 1"
    ):
        estimate_max_model_atoms_per_chunk(
            torch.eye(2, dtype=torch.float64),
            atomic_grid_sizes=torch.tensor([10]),
            nfeatures=5,
            max_memory_in_mb=100,
            safety_fraction=safety_fraction,
        )
