import pytest
import torch

from skala.pyscf.memory_estimators import (
    estimate_global_raw_feature_buffer_memory,
    estimate_max_model_grid_points,
    linear_peak_memory_model,
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


def test_reserved_memory_reduces_grid_chunk_size() -> None:
    dm = torch.eye(10, dtype=torch.float64)
    bytes_per_point, _ = linear_peak_memory_model(nao=10, deriv=1, func_deriv=1)
    base_chunk_size = estimate_max_model_grid_points(
        dm,
        deriv=1,
        max_memory_in_mb=100,
        safety_fraction=1.0,
        func_deriv=1,
    )
    reserved_points = 123
    reserved_chunk_size = estimate_max_model_grid_points(
        dm,
        deriv=1,
        max_memory_in_mb=100,
        safety_fraction=1.0,
        func_deriv=1,
        reserved_memory_in_bytes=int(bytes_per_point * reserved_points),
    )

    assert base_chunk_size - reserved_chunk_size == reserved_points


@pytest.mark.parametrize("safety_fraction", [-0.1, 0.0, 1.1])
def test_model_grid_point_limit_rejects_invalid_safety_fraction(
    safety_fraction: float,
) -> None:
    with pytest.raises(
        ValueError, match="safety_fraction must be greater than 0 and at most 1"
    ):
        estimate_max_model_grid_points(
            torch.eye(2, dtype=torch.float64),
            deriv=1,
            max_memory_in_mb=100,
            safety_fraction=safety_fraction,
        )
