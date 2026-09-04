# SPDX-License-Identifier: MIT

"""Backend-independent PyTorch operations for PySCF nuclear gradients."""

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from enum import IntEnum
from types import EllipsisType
from typing import ClassVar, TypeAlias

import torch

from skala.features import Feature, FeatureMap


class _Direction(IntEnum):
    """Cartesian direction indices used by feature and potential tensors."""

    X = 0
    Y = 1
    Z = 2


@dataclass(frozen=True)
class _PackedAO:
    """Views into PySCF's packed AO derivative dimension."""

    tensor: torch.Tensor

    # PySCF's ``eval_ao(..., deriv=2)`` order is
    # value, x, y, z, xx, xy, xz, yy, yz, zz. Mixed derivatives are reused
    # across the symmetric off-diagonal entries of the Cartesian Hessian.
    _HESSIAN_COMPONENTS: ClassVar[tuple[tuple[int, int, int], ...]] = (
        (4, 5, 6),
        (5, 7, 8),
        (6, 8, 9),
    )

    @property
    def value(self) -> torch.Tensor:
        """AO values with shape ``(npoints, nao)``."""
        return self.tensor[0]

    @property
    def gradient(self) -> torch.Tensor:
        """AO gradients with shape ``(3, npoints, nao)``."""
        return self.tensor[1:4]

    def hessian(self) -> Iterator[tuple[_Direction, _Direction, torch.Tensor]]:
        """Yield both Cartesian directions and each AO Hessian component."""
        for ao_direction in _Direction:
            for feature_direction in _Direction:
                component = self._HESSIAN_COMPONENTS[ao_direction][feature_direction]
                yield ao_direction, feature_direction, self.tensor[component]


_FeatureSlice: TypeAlias = tuple[slice] | tuple[EllipsisType, slice]


def _disconnected_features_error(features: Iterable[Feature]) -> RuntimeError:
    feature_names = ", ".join(sorted(feature.value for feature in features))
    return RuntimeError(
        f"XC energy is disconnected from requested features: {feature_names}"
    )


def feature_derivatives(
    exc_func: Callable[[FeatureMap], torch.Tensor], features: FeatureMap
) -> FeatureMap:
    """Differentiate a scalar XC energy with respect to molecular features.

    Ordinary autograd tensors are used instead of ``torch.func.vjp`` functional
    tensors because traced TorchScript models may require accessible backing
    storage.

    Args:
        exc_func: Callable accepting the differentiable features and returning
            scalar XC energy.
        features: Molecular features to differentiate, keyed by feature name.

    Returns:
        XC energy derivatives keyed by feature name.

    Raises:
        RuntimeError: If the XC energy is disconnected from a requested feature.

    """
    if not features:
        return {}

    differentiable_features = {
        feature: tensor.detach().requires_grad_(True)
        for feature, tensor in features.items()
    }
    exc = exc_func(differentiable_features)
    if not exc.requires_grad:
        raise _disconnected_features_error(differentiable_features)

    gradients = torch.autograd.grad(
        exc,
        tuple(differentiable_features.values()),
        create_graph=False,
        retain_graph=False,
        allow_unused=True,
    )
    derivatives: FeatureMap = {
        feature: gradient.detach()
        for feature, gradient in zip(differentiable_features, gradients, strict=True)
        if gradient is not None
    }
    if len(derivatives) != len(differentiable_features):
        disconnected_features = differentiable_features.keys() - derivatives.keys()
        raise _disconnected_features_error(disconnected_features)
    return derivatives


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
    packed_ao = _PackedAO(ao)
    nao = ao.shape[-1]
    potential_derivatives = ao.new_zeros((2, 3, nao, nao))

    if Feature.DENSITY in feature_derivatives:
        potential_derivatives += torch.einsum(
            "si, xip, iq -> sxpq",
            feature_derivatives[Feature.DENSITY],
            packed_ao.gradient,
            packed_ao.value,
        )

    if Feature.GRAD in feature_derivatives:
        exc_dgrad = feature_derivatives[Feature.GRAD]
        potential_derivatives += torch.einsum(
            "syi, xip, yiq -> sxpq",
            exc_dgrad,
            packed_ao.gradient,
            packed_ao.gradient,
        )
        for ao_direction, feature_direction, ao_hessian in packed_ao.hessian():
            potential_derivatives[:, ao_direction] += torch.einsum(
                "si, ip, iq -> spq",
                exc_dgrad[:, feature_direction],
                ao_hessian,
                packed_ao.value,
            )

    if Feature.KIN in feature_derivatives:
        exc_dkin = feature_derivatives[Feature.KIN]
        for ao_direction, feature_direction, ao_hessian in packed_ao.hessian():
            potential_derivatives[:, ao_direction] += (
                torch.einsum(
                    "si, ip, iq -> spq",
                    exc_dkin,
                    ao_hessian,
                    packed_ao.gradient[feature_direction],
                )
                / 2
            )

    return potential_derivatives
