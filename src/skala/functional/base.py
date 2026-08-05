# SPDX-License-Identifier: MIT

"""
Base classes for exchange-correlation functionals.

This module defines the abstract base classes and utility functions
for implementing exchange-correlation functionals in Skala.
"""

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

VxcType = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class ExcFunctionalBase(nn.Module):
    """
    Abstract base class for exchange-correlation functionals.

    This class defines the interface that all exchange-correlation functionals
    must implement. Functionals can compute exchange-correlation energy and
    energy density from molecular features.
    """

    features: list[str]
    """List of features that this functional requires."""

    def get_d3_settings(self) -> str | None:
        """
        Returns the D3 settings that this functional expects.
        If the functional does not use D3, it returns None.
        """
        return None

    def get_exc_density(self, mol: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Returns the exchange-correlation density for the given molecule.
        It should return a tensor of shape (G,) where G is the number of grid points
        that can be integrated by taking a dot product with the grid weights to get
        EXC.
        """
        raise NotImplementedError(
            "get_exc_density not implemented for this functional."
        )

    def get_exc(self, mol: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute the exchange-correlation energy.

        Parameters
        ----------
        mol : dict[str, torch.Tensor]
            Dictionary containing molecular features including density,
            gradients, kinetic energy, grid coordinates, and grid weights.

        Returns
        -------
        torch.Tensor
            The total exchange-correlation energy.
        """
        exc_density = self.get_exc_density(mol).double()
        grid_weights = mol["grid_weights"].double()

        return (exc_density * grid_weights).sum()


def spin_symmetrized_enhancement_factor(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    spin_agnostic_tensor: torch.Tensor,
    enhancement_func: Callable[..., torch.Tensor],
    **kwargs: Any,
) -> torch.Tensor:
    """
    Apply spin symmetrization to an enhancement factor function.

    Runs a model twice with spin features swapped and averages the output
    to ensure spin symmetry.

    Parameters
    ----------
    tensor_a : torch.Tensor
        Features for spin-up electrons.
    tensor_b : torch.Tensor
        Features for spin-down electrons.
    spin_agnostic_tensor : torch.Tensor
        Features that are the same for both spins.
    enhancement_func : Callable
        Function to compute enhancement factor.
    **kwargs
        Additional arguments for enhancement_func.

    Returns
    -------
    torch.Tensor
        Spin-symmetrized enhancement factor.
    """
    x_ab = torch.cat((tensor_a, tensor_b, spin_agnostic_tensor), dim=-1)
    x_ba = torch.cat((tensor_b, tensor_a, spin_agnostic_tensor), dim=-1)
    enhancement_factor = (
        enhancement_func(x_ab, **kwargs) + enhancement_func(x_ba, **kwargs)
    ) / 2
    return enhancement_factor


# The spin-agnostic version of the lda exchange is
#     E_x(rho) = \int d^3r e_x(rho(r))
#     e_x(rho) = - (3/4) * (3/pi)**(1/3) * rho ** (4/3)
# The spin-polarized version of the lda exchange is
#     E_x(rho_up, rho_down) = 0.5 * \int d^3r e_x(2 rho_up(r)) + e_x(2 rho_down(r))
#                           = 0.5 * 2 ** (4/3) * \int d^3r (e_x(rho_up(r)) + e_x(rho_down(r)))
#                           = [ - 2 ** (1/3) * (3/4) * (3/pi)**(1/3) ]
#                              * \int d^3 r rho_up**(4/3) + rho_down**(4/3)
# The prefactor in the squared bracket here is -0.9305257363491001

LDA_PREFACTOR = -0.9305257363491001


def enhancement_density_inner_product(
    enhancement_factor: torch.Tensor,
    density: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the enhancement density as inner product with LDA reference.

    Parameters
    ----------
    enhancement_factor : torch.Tensor
        Enhancement factor with shape (n, 1).
    density : torch.Tensor
        Electron density with shape (2, n) for 2 spins and n grid points.

    Returns
    -------
    torch.Tensor
        Enhanced exchange-correlation density.

    Notes
    -----
    This function computes:
    enhancement_factor * LDA_exchange_density
    where LDA_exchange_density uses the prefactor -0.9305257363491001.
    """
    lda = LDA_PREFACTOR * torch.pow(torch.clip(density.double(), 0), 4 / 3).sum(
        dim=0
    ).view(-1, 1)

    return (enhancement_factor.to(lda.dtype) * lda).squeeze(1)
