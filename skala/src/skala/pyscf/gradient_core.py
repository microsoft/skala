# SPDX-License-Identifier: MIT

"""Backend-independent PyTorch operations for PySCF nuclear gradients."""

from collections.abc import Callable, Iterator, Sequence
from enum import IntEnum

import torch

from skala.features import Feature, FeatureMap


class _AOComponent(IntEnum):
    """Indices in PySCF's packed AO derivative dimension."""

    VALUE = 0
    X = 1
    Y = 2
    Z = 3
    XX = 4
    XY = 5
    XZ = 6
    YY = 7
    YZ = 8
    ZZ = 9


class _Direction(IntEnum):
    """Cartesian direction indices used by feature and potential tensors."""

    X = 0
    Y = 1
    Z = 2


_AO_GRADIENT = slice(_AOComponent.X, _AOComponent.XX)
_GRADIENT_COMPONENTS = (_AOComponent.X, _AOComponent.Y, _AOComponent.Z)
_HESSIAN_COMPONENTS = (
    (_AOComponent.XX, _AOComponent.XY, _AOComponent.XZ),
    (_AOComponent.XY, _AOComponent.YY, _AOComponent.YZ),
    (_AOComponent.XZ, _AOComponent.YZ, _AOComponent.ZZ),
)


def _hessian_components() -> Iterator[tuple[_Direction, _Direction, _AOComponent]]:
    """Yield force direction, response direction, and packed AO component."""
    for force_direction in _Direction:
        for response_direction in _Direction:
            yield (
                force_direction,
                response_direction,
                _HESSIAN_COMPONENTS[force_direction][response_direction],
            )


# Grid metadata is laid out with one row per point: coordinates have shape
# (npoints, 3), while weights have shape (npoints,).
_POINT_FIRST_FEATURES = (Feature.GRID_COORDS, Feature.GRID_WEIGHTS)

# Electronic features keep spin and, for GRAD, Cartesian components before the
# grid dimension: DENSITY and KIN have shape (2, npoints), and GRAD has shape
# (2, 3, npoints).
_POINT_LAST_FEATURES = (Feature.DENSITY, Feature.GRAD, Feature.KIN)


def feature_derivatives(
    exc_func: Callable[..., torch.Tensor], feature_tensors: Sequence[torch.Tensor]
) -> tuple[torch.Tensor, ...]:
    """Differentiate a scalar XC energy with respect to molecular features.

    Ordinary autograd tensors are used instead of ``torch.func.vjp`` functional
    tensors because traced TorchScript models may require accessible backing
    storage.

    Args:
        exc_func: Callable accepting the feature tensors and returning scalar XC energy.
        feature_tensors: Feature tensors in the order expected by ``exc_func``.

    Returns:
        Feature derivatives in the same order as ``feature_tensors``.

    """
    if not feature_tensors:
        return ()

    differentiable_features = tuple(
        tensor.detach().requires_grad_(True) for tensor in feature_tensors
    )
    exc = exc_func(*differentiable_features)
    if not exc.requires_grad:
        return tuple(torch.zeros_like(feature) for feature in differentiable_features)

    gradients = torch.autograd.grad(
        exc,
        differentiable_features,
        create_graph=False,
        retain_graph=False,
        allow_unused=True,
    )
    return tuple(
        torch.zeros_like(feature) if gradient is None else gradient.detach()
        for feature, gradient in zip(differentiable_features, gradients, strict=True)
    )


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
    block = {
        feature: derivative[grid_start:grid_end]
        for feature, derivative in derivatives.items()
        if feature in _POINT_FIRST_FEATURES
    }
    block.update(
        {
            feature: derivative[..., grid_start:grid_end]
            for feature, derivative in derivatives.items()
            if feature in _POINT_LAST_FEATURES
        }
    )
    return block


def contract_veff_block(ao: torch.Tensor, derivatives: FeatureMap) -> torch.Tensor:
    """Contract one atom's AO values with XC feature derivatives.

    Args:
        ao: Component-major AO values for this atom's grid block, shaped
            ``(ncomponents, npoints, nao)``. Here ``npoints`` must equal
            ``grid_end - grid_start`` and ``nao`` is the number of molecular
            AOs. PySCF supplies 1, 4, or 10 components for derivative orders
            0, 1, or 2, respectively, packed as ``value, x, y, z, xx, xy,
            xz, yy, yz, zz``. The requested features determine how many of
            these leading components the contraction reads.
        derivatives: XC energy derivatives already sliced to this grid block.

    Returns:
        Spin- and Cartesian-resolved effective-potential contribution with shape
        ``(2, 3, nao, nao)`` on the same device and with the same dtype as ``ao``.
    """
    nao = ao.shape[-1]
    veff = ao.new_zeros((2, 3, nao, nao))

    if Feature.DENSITY in derivatives:
        veff += torch.einsum(
            "si, xip, iq -> sxpq",
            derivatives[Feature.DENSITY],
            ao[_AO_GRADIENT],
            ao[_AOComponent.VALUE],
        )

    if Feature.GRAD in derivatives:
        exc_dgrad = derivatives[Feature.GRAD]
        ao_gradient = ao[_AO_GRADIENT]
        veff += torch.einsum(
            "syi, xip, yiq -> sxpq", exc_dgrad, ao_gradient, ao_gradient
        )
        for force_direction, response_direction, component in _hessian_components():
            veff[:, force_direction] += torch.einsum(
                "si, ip, iq -> spq",
                exc_dgrad[:, response_direction],
                ao[component],
                ao[_AOComponent.VALUE],
            )

    if Feature.KIN in derivatives:
        exc_dkin = derivatives[Feature.KIN]
        for force_direction, response_direction, component in _hessian_components():
            veff[:, force_direction] += (
                torch.einsum(
                    "si, ip, iq -> spq",
                    exc_dkin,
                    ao[component],
                    ao[_GRADIENT_COMPONENTS[response_direction]],
                )
                / 2
            )

    return veff
