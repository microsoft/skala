# SPDX-License-Identifier: MIT

"""Backend-independent PyTorch operations for PySCF nuclear gradients."""

from collections.abc import Iterable
from types import EllipsisType
from typing import TypeAlias

import torch
from torch import Tensor

from pyscf import gto
from skala.features import Feature, FeatureMap
from skala.functional.base import ExcFunctionalBase
from skala.pyscf.backend import Grid
from skala.pyscf.evaluation import FeatureSpec
from skala.pyscf.feature_math import PackedAO
from skala.pyscf.model_chunking import (
    evaluate_chunked_feature_gradients,
    evaluate_model_features,
)

SUPPORTED_NUCLEAR_GRADIENT_FEATURES = frozenset(
    {
        Feature.DENSITY,
        Feature.GRAD,
        Feature.KIN,
        Feature.GRID_COORDS,
        Feature.GRID_WEIGHTS,
        Feature.ATOMIC_GRID_WEIGHTS,
        Feature.COARSE_0_ATOMIC_COORDS,
    }
)


_FeatureSlice: TypeAlias = tuple[slice] | tuple[EllipsisType, slice]


def grid_derivative_block(
    derivatives: FeatureMap, grid_start: int, grid_end: int
) -> FeatureMap:
    """Select a slice of the feature derivatives from `grid_start` to `grid_end`.

    Density-like features store the grid-point dimension last, whereas grid
    coordinates and weights store it first. Non-grid features, such as atomic
    coordinates, are intentionally omitted.

    Args:
        derivatives: XC energy derivatives keyed by molecular feature.
        grid_start: Start of this block in the full molecular feature tensors.
        grid_end: End of this block in the full molecular feature tensors.

    Returns:
        Grid-resolved derivatives restricted to ``[grid_start:grid_end]``.
    """
    grid_slice = slice(grid_start, grid_end)
    feature_slices: dict[Feature, _FeatureSlice] = {
        Feature.GRID_COORDS: (grid_slice,),
        Feature.GRID_WEIGHTS: (grid_slice,),
        Feature.DENSITY: (..., grid_slice),
        Feature.GRAD: (..., grid_slice),
        Feature.KIN: (..., grid_slice),
    }
    return {
        feature: derivative[feature_slice]
        for feature, derivative in derivatives.items()
        if (feature_slice := feature_slices.get(feature)) is not None
    }


def evaluate_nuclear_feature_derivatives(
    functional: ExcFunctionalBase,
    mol: gto.Mole,
    grid: Grid,
    rdm1: Tensor,
    features: set[Feature] | None = None,
    *,
    max_memory_in_mb: int,
) -> tuple[int, FeatureMap]:
    """Evaluate model derivatives needed for a nuclear gradient.

    Args:
        functional: XC functional whose energy is differentiated.
        mol: Molecule associated with the density matrix and grid.
        grid: Atom-major integration grid used for feature evaluation.
        rdm1: One-particle density matrix.
        features: Features to differentiate, defaulting to the functional inputs.
        max_memory_in_mb: Memory limit for AO evaluation and CPU model-gradient
            chunking. CUDA model-gradient chunking probes available device memory.

    Returns:
        Required AO derivative order and XC derivatives by feature.

    Raises:
        NotImplementedError: If a requested feature has no nuclear-gradient rule.
    """
    differentiable_features = set(functional.features if features is None else features)
    differentiable_features -= {
        Feature.ATOMIC_GRID_SIZES,
        Feature.ATOMIC_GRID_SIZE_BOUND_SHAPE,
    }
    unsupported = differentiable_features - SUPPORTED_NUCLEAR_GRADIENT_FEATURES
    if unsupported:
        raise NotImplementedError(
            f"Not supported features for nuclear gradient: {unsupported}"
        )

    ao_feature_spec = FeatureSpec(differentiable_features).ao_features
    # Nuclear gradients need one order beyond feature evaluation; non-AO paths need none.
    ao_deriv = 0 if ao_feature_spec is None else ao_feature_spec.nderiv + 1
    differentiable_features.discard(Feature.ATOMIC_GRID_WEIGHTS)
    model_features = evaluate_model_features(
        mol,
        rdm1,
        grid,
        FeatureSpec((*functional.features, Feature.ATOMIC_GRID_SIZES)),
        max_memory_in_mb=max_memory_in_mb,
    )
    derivatives = evaluate_chunked_feature_gradients(
        functional,
        rdm1,
        model_features,
        differentiable_features,
        max_memory_in_mb=max_memory_in_mb if rdm1.device.type == "cpu" else None,
    )
    return ao_deriv, derivatives


def assemble_nuclear_gradient(
    derivatives: FeatureMap,
    rdm1: Tensor,
    natm: int,
    atom_grid_blocks: Iterable[tuple[Tensor, int, Tensor]],
) -> tuple[Tensor, Tensor]:
    """Assemble effective-potential and explicit nuclear-gradient contributions.

    Args:
        derivatives: XC energy derivatives keyed by molecular feature.
        rdm1: One-particle density matrix.
        natm: Number of atoms in the molecule.
        atom_grid_blocks: Tuples containing AO values, grid-point count, and
            grid-weight derivatives for each atom.

    Returns:
        Negative effective potential and explicit nuclear gradient.
    """
    nao = rdm1.shape[-1]
    veff = rdm1.new_zeros((2, 3, nao, nao))
    nuclear_gradient = rdm1.new_zeros((natm, 3))

    grid_start = 0
    for atom, (ao, grid_size, weight_derivatives) in enumerate(atom_grid_blocks):
        grid_end = grid_start + grid_size
        atom_derivatives = grid_derivative_block(derivatives, grid_start, grid_end)
        atom_veff = contract_ao_derivative_block(ao, atom_derivatives)

        if Feature.GRID_COORDS in atom_derivatives:
            nuclear_gradient[atom] += atom_derivatives[Feature.GRID_COORDS].sum(dim=0)

        if Feature.GRID_WEIGHTS in atom_derivatives:
            grid_weight_derivative = atom_derivatives[Feature.GRID_WEIGHTS]
            nuclear_gradient += weight_derivatives @ grid_weight_derivative
            if rdm1.ndim == 2:
                nuclear_gradient[atom] += torch.einsum("sxpq,qp->x", atom_veff, rdm1)
            else:
                nuclear_gradient[atom] += (
                    torch.einsum("sxpq,sqp->x", atom_veff, rdm1) * 2
                )

        veff += atom_veff
        grid_start = grid_end

    if Feature.COARSE_0_ATOMIC_COORDS in derivatives:
        nuclear_gradient += derivatives[Feature.COARSE_0_ATOMIC_COORDS]

    if rdm1.ndim == 2:
        veff = veff.sum(0) / 2

    return -veff, nuclear_gradient


def contract_ao_derivative_block(
    ao: torch.Tensor, feature_derivatives: FeatureMap
) -> torch.Tensor:
    """Contract XC feature derivatives with one-sided spatial AO derivatives.

    For each Cartesian direction, form one grid block's contribution to the
    spatial derivative of the AO-basis XC potential. The AO associated with
    the first matrix index is differentiated, while the AO associated with the
    second index is held fixed. Consequently, the returned matrices are not
    generally symmetric in their AO indices.

    Density, density-gradient, and kinetic-energy-density contributions are
    included when their corresponding feature derivatives are present.

    Args:
        ao: Packed component-major AO values with shape
            ``(ncomponents, npoints, nao)``. Components follow PySCF's ordering:
            ``value, x, y, z, xx, xy, xz, yy, yz, zz``.
        feature_derivatives: XC energy derivatives with respect to features on
            this grid block. Density and kinetic derivatives have shape
            ``(2, npoints)``; gradient derivatives have shape
            ``(2, 3, npoints)``.

    Returns:
        One-sided AO-potential derivatives with shape ``(2, 3, nao, nao)``,
        indexed by spin, Cartesian direction, and the two AO indices. The result
        has the same device and dtype as ``ao``.
    """
    packed_ao = PackedAO(ao)
    nao = ao.shape[-1]
    potential_derivatives = ao.new_zeros((2, 3, nao, nao))

    for feature, feature_derivative in feature_derivatives.items():
        if feature == Feature.DENSITY:
            potential_derivatives += torch.einsum(
                "si, xip, iq -> sxpq",
                feature_derivative,
                packed_ao.gradient,
                packed_ao.value,
            )
        elif feature == Feature.GRAD:
            potential_derivatives += torch.einsum(
                "syi, xip, yiq -> sxpq",
                feature_derivative,
                packed_ao.gradient,
                packed_ao.gradient,
            )
            for ao_direction, feature_direction, ao_hessian in packed_ao.hessian():
                potential_derivatives[:, ao_direction] += torch.einsum(
                    "si, ip, iq -> spq",
                    feature_derivative[:, feature_direction],
                    ao_hessian,
                    packed_ao.value,
                )
        elif feature == Feature.KIN:
            for ao_direction, feature_direction, ao_hessian in packed_ao.hessian():
                potential_derivatives[:, ao_direction] += (
                    torch.einsum(
                        "si, ip, iq -> spq",
                        feature_derivative,
                        ao_hessian,
                        packed_ao.gradient[feature_direction],
                    )
                    / 2
                )

    return potential_derivatives
