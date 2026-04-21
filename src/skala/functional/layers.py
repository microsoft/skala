# SPDX-License-Identifier: MIT

"""
Neural network layers for Skala functionals.

This module provides specialized PyTorch layers used in the construction
of neural exchange-correlation functionals, including squashing functions,
skip connections, and scaled activations.
"""

from typing import Any

import torch
from torch import nn


class Squasher(nn.Module):
    """
    Elementwise squashing function log(|x| + eta).

    This layer applies a logarithmic squashing to prevent extreme values
    and improve numerical stability in neural functionals.
    """

    eta: float
    """Small constant added before taking logarithm for numerical stability."""

    def __init__(self, eta: float):
        super().__init__()
        self.eta = eta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply squashing function log(|x| + eta)."""
        return (x.abs() + self.eta).log()


class LinearSkip(nn.Linear):
    """
    Linear layer with skip connection, used to initialize close to identity.

    This layer computes: output = input + W @ input + b
    where W is initialized to small values around zero.
    """

    def __init__(self, in_features: int, out_features: int, **kwargs: Any):
        """
        Initialize linear skip layer.

        Parameters
        ----------
        in_features : int
            Number of input features.
        out_features : int
            Number of output features. Must equal in_features.
        **kwargs
            Additional arguments passed to nn.Linear.
        """
        super().__init__(in_features=in_features, out_features=out_features, **kwargs)
        assert in_features == out_features, (
            f"Expecting args in_features == out_features, got {in_features} != {out_features}."
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights to be close to the identity transformation."""
        nn.init.trunc_normal_(
            self.weight.data, mean=0.0, std=0.0625, a=-0.125, b=0.125
        )  # std value is copied from loaded graph of checkpointed DM21 model

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply linear transformation with skip connection."""
        return input + nn.functional.linear(input, self.weight, self.bias)


class ScaledSigmoid(nn.Sigmoid):
    """
    Sigmoid activation function with learnable scaling.

    Computes: scale * sigmoid(x / scale)
    """

    def __init__(self, scale: float = 1.0):
        """
        Initialize scaled sigmoid.

        Parameters
        ----------
        scale : float, optional
            Scaling factor for the sigmoid. Default: 1.0.
        """
        super().__init__()
        self.scale = scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply scaled sigmoid activation."""
        return self.scale * super().forward(input / self.scale)
