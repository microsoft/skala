# SPDX-License-Identifier: MIT

import torch

from skala.functional.base import spin_symmetrized_enhancement_factor


def test_spin_symmetrized_enhancement_factor() -> None:
    n = 16
    dim_ab = 6
    dim_agnostic = 2

    def _test_func(features: torch.Tensor, dummy_arg: int) -> torch.Tensor:
        return (
            2 * features[:, :dim_ab].sum(dim=-1)
            + features[:, dim_ab:-dim_agnostic].sum(dim=-1)
        ) - features[:, -dim_agnostic:].sum(dim=-1)

    tensor_a = torch.randn(n, dim_ab)
    tensor_b = torch.randn(n, dim_ab)
    spin_agnostic_tensor = torch.randn(n, dim_agnostic)

    additional_args = {"dummy_arg": 1}

    result_ab = spin_symmetrized_enhancement_factor(
        tensor_a, tensor_b, spin_agnostic_tensor, _test_func, **additional_args
    )
    result_ba = spin_symmetrized_enhancement_factor(
        tensor_b, tensor_a, spin_agnostic_tensor, _test_func, **additional_args
    )
    assert torch.allclose(result_ab, result_ba)
