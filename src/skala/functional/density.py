# SPDX-License-Identifier: MIT

"""
Density-related utility functions for exchange-correlation functionals.

This module provides functions for manipulating and computing derived
quantities from electron density and its derivatives, including spin
polarization, reduced gradients, and kinetic energy densities.
"""

import math
from collections.abc import Callable

import torch
from torch import Tensor

from skala.features import Feature, FeatureMap

EPS = 1e-10
IMMUTABLES: frozenset[Feature] = frozenset([Feature.GRID_COORDS, Feature.GRID_WEIGHTS])


def _map(mol_features: FeatureMap, f: Callable[[Tensor], Tensor]) -> FeatureMap:
    """
    Apply a function to mutable molecular features.

    Parameters
    ----------
    mol_features : dict[str, Tensor]
        Dictionary of molecular features.
    f : Callable[[Tensor], Tensor]
        Function to apply to each feature.

    Returns
    -------
    dict[str, Tensor]
        Features with function applied to mutable entries.
    """
    return {
        key: value if key in IMMUTABLES else f(value)
        for key, value in mol_features.items()
    }


def separate(
    mol_features: FeatureMap,
) -> tuple[FeatureMap, FeatureMap]:
    """
    Separate molecular features into spin-up and spin-down components.

    Creates two copies of molecular features: one with spin-down features
    set to zero, and another with spin-up features set to zero.

    Parameters
    ----------
    mol_features : dict[str, Tensor]
        Dictionary containing molecular features with spin components.

    Returns
    -------
    tuple[dict[str, Tensor], dict[str, Tensor]]
        (spin_up_features, spin_down_features)
    """
    mol_a = {}
    mol_b = {}
    for key, value in mol_features.items():
        if key in IMMUTABLES:
            mol_a[key] = value
            mol_b[key] = value
        else:
            mol_a[key] = torch.stack([value[0], torch.zeros_like(value[0])])
            mol_b[key] = torch.stack([torch.zeros_like(value[1]), value[1]])

    return mol_a, mol_b


def scale_by(mol_features: FeatureMap, factor: float) -> FeatureMap:
    """
    Scale molecular features by a constant factor.

    Parameters
    ----------
    mol_features : dict[str, Tensor]
        Dictionary of molecular features.
    factor : float
        Scaling factor to apply.

    Returns
    -------
    dict[str, Tensor]
        Scaled molecular features.
    """
    return _map(mol_features, lambda x: factor * x)


def zeta(rho: Tensor) -> Tensor:
    """
    Compute the spin polarization parameter.

    Parameters
    ----------
    rho : Tensor
        Electron density with shape (2, ...) for spin-up and spin-down.

    Returns
    -------
    Tensor
        Spin polarization ζ = (ρ_up - ρ_down) / (ρ_up + ρ_down).
    """
    rho = rho.abs()
    return (rho[0] - rho[1]) / (rho.sum(0) + EPS)


def grad_zeta(rho: Tensor, grad: Tensor) -> Tensor:
    """
    Compute the gradient of the spin polarization parameter.

    Parameters
    ----------
    rho : Tensor
        Electron density with shape (2, ...).
    grad : Tensor
        Density gradient with shape (2, 3, ...).

    Returns
    -------
    Tensor
        Gradient of spin polarization ∇ζ.
    """
    rho_total = rho.sum(dim=0)
    grad_total = grad.sum(dim=0)
    return (grad[0] - grad[1]) / torch.clamp(rho_total, EPS) - grad_total * (
        rho[0] - rho[1]
    ) / torch.clamp(rho_total**2, EPS)


def kF(rho: Tensor) -> Tensor:
    """
    Compute the Fermi wave vector.

    Parameters
    ----------
    rho : Tensor
        Electron density.

    Returns
    -------
    Tensor
        Fermi wave vector k_F = (3π²ρ)^(1/3).
    """
    return (3 * math.pi**2 * torch.clamp(rho, EPS)) ** (1 / 3)


def reduced_gradient(rho: Tensor, grad: Tensor) -> Tensor:
    """
    Compute the reduced density gradient.

    Parameters
    ----------
    rho : Tensor
        Electron density.
    grad : Tensor
        Density gradient.

    Returns
    -------
    Tensor
        Reduced gradient |∇ρ|/(2k_F ρ).
    """
    return grad_norm(grad) / torch.clamp(2 * kF(rho) * rho, EPS)


def grad_norm(grad: Tensor) -> Tensor:
    """
    Compute the norm of the density gradient.

    Parameters
    ----------
    grad : Tensor
        Density gradient with shape [..., 3, ...].

    Returns
    -------
    Tensor
        Gradient norm |∇ρ|.
    """
    # expecting [(2,)3,G)]
    return grad.norm(dim=-2)


def z(rho: Tensor, grad: Tensor, kin: Tensor) -> Tensor:
    """
    Compute the z parameter for meta-GGA functionals.

    Parameters
    ----------
    rho : Tensor
        Electron density.
    grad : Tensor
        Density gradient.
    kin : Tensor
        Kinetic energy density.

    Returns
    -------
    Tensor
        z parameter: |∇ρ|²/(8ρτ).
    """
    return (
        1 / 8 * torch.clamp(grad_norm(grad), EPS) ** 2 / torch.clamp(rho, EPS)
    ) / torch.clamp(kin, EPS)
