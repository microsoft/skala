# SPDX-License-Identifier: MIT

"""Backend-independent PyTorch operations for PySCF nuclear gradients."""

from collections.abc import Callable

import torch

from skala.features import Feature, FeatureMap

# PySCF packs AO derivatives as value, x, y, z, xx, xy, xz, yy, yz, zz.
# Each row below selects the Hessian components for one nuclear displacement
# direction, with columns ordered by the x, y, and z density responses.
_HESSIAN_COMPONENTS = ((4, 5, 6), (5, 7, 8), (6, 8, 9))

# Grid metadata is laid out with one row per point: coordinates have shape
# (npoints, 3), while weights have shape (npoints,).
_POINT_FIRST_FEATURES = {Feature.GRID_COORDS, Feature.GRID_WEIGHTS}

# Electronic features keep spin and, for GRAD, Cartesian components before the
# grid dimension: DENSITY and KIN have shape (2, npoints), and GRAD has shape
# (2, 3, npoints).
_POINT_LAST_FEATURES = {Feature.DENSITY, Feature.GRAD, Feature.KIN}


def feature_derivatives(
    exc_func: Callable[..., torch.Tensor],
    feature_tensors: list[torch.Tensor],
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
        tuple(differentiable_features),
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
    """Slice grid-resolved feature derivatives to one atom's grid block.

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


def contract_veff_block(
    ao: torch.Tensor,
    derivatives: FeatureMap,
) -> torch.Tensor:
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
            ao[1:4],
            ao[0],
        )

    if Feature.GRAD in derivatives:
        exc_dgrad = derivatives[Feature.GRAD]
        veff += torch.einsum("syi, xip, yiq -> sxpq", exc_dgrad, ao[1:4], ao[1:4])
        for force_direction, components in enumerate(_HESSIAN_COMPONENTS):
            for response_direction, component in enumerate(components):
                veff[:, force_direction] += torch.einsum(
                    "si, ip, iq -> spq",
                    exc_dgrad[:, response_direction],
                    ao[component],
                    ao[0],
                )

    if Feature.KIN in derivatives:
        exc_dkin = derivatives[Feature.KIN]
        for force_direction, components in enumerate(_HESSIAN_COMPONENTS):
            for response_direction, component in enumerate(components):
                veff[:, force_direction] += (
                    torch.einsum(
                        "si, ip, iq -> spq",
                        exc_dkin,
                        ao[component],
                        # AO components 1, 2, and 3 are the x, y, and z derivatives.
                        ao[response_direction + 1],
                    )
                    / 2
                )

    return veff
