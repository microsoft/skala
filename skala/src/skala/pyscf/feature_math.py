# SPDX-License-Identifier: MIT

"""Raw density-feature mathematics and model formatting."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from enum import IntEnum
from typing import ClassVar

import torch
from torch import nn

from skala.features import AOFeatureSpec, Feature, FeatureMap


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


class AODirection(IntEnum):
    """Cartesian direction indices used by feature and potential tensors."""

    X = 0
    Y = 1
    Z = 2


class PackedAO:
    """Component-axis views over packed PySCF AO derivative values.

    PySCF stores AO values and Cartesian derivatives along the leading tensor
    dimension in the order ``value, x, y, z, xx, xy, xz, yy, yz, zz``. This
    wrapper normalizes rank-two value-only input by inserting a leading
    component dimension. For packed input, it interprets only the leading
    dimension and leaves all trailing dimensions unchanged. It therefore
    supports both the ``(component, nao, ngrid)`` layout used by density-feature
    evaluation and the ``(component, ngrid, nao)`` layout used by
    nuclear-gradient contractions.

    Valid packed component counts are one for AO values, four through first
    derivatives, and ten through second derivatives. Accessors require the
    corresponding components to be available: :attr:`gradient` requires first
    derivatives, while :attr:`diagonal_hessian` and :meth:`hessian` require
    second derivatives.

    Args:
        tensor: AO values, either unpacked with rank two or with PySCF derivative
            components on the leading axis.

    Raises:
        ValueError: If a packed tensor has an unsupported component count.
    """

    # PySCF's ``eval_ao(..., deriv=2)`` order is
    # value, x, y, z, xx, xy, xz, yy, yz, zz. Mixed derivatives are reused
    # across the symmetric off-diagonal entries of the Cartesian Hessian.
    _HESSIAN_COMPONENTS: ClassVar[tuple[tuple[int, int, int], ...]] = (
        (4, 5, 6),
        (5, 7, 8),
        (6, 8, 9),
    )

    def __init__(self, tensor: torch.Tensor) -> None:
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)

        ncomponents = tensor.shape[0]
        if ncomponents not in (1, 4, 10):
            raise ValueError(
                "Packed AO values must contain 1, 4, or 10 derivative components; "
                f"got {ncomponents}."
            )
        self._tensor = tensor

    @property
    def value(self) -> torch.Tensor:
        """Return the AO values."""
        return self._tensor[0]

    @property
    def gradient(self) -> torch.Tensor:
        """Return the three Cartesian AO gradients."""
        return self._tensor[1:4]

    @property
    def diagonal_hessian(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the three diagonal Cartesian AO Hessian components."""
        return self._tensor[4], self._tensor[7], self._tensor[9]

    def hessian(self) -> Iterator[tuple[AODirection, AODirection, torch.Tensor]]:
        """Yield both Cartesian directions and each AO Hessian component."""
        for ao_direction in AODirection:
            for feature_direction in AODirection:
                component = self._HESSIAN_COMPONENTS[ao_direction][feature_direction]
                yield ao_direction, feature_direction, self._tensor[component]


def maybe_expand_and_divide(
    feature: torch.Tensor, expand: bool, divisor: float
) -> torch.Tensor:
    """Expand a feature across spin channels and divide it when requested."""
    if expand:
        return torch.stack([feature / divisor, feature / divisor], dim=0)
    return feature


class LinearFeature(nn.Module, ABC):
    """Linear raw-feature map from a density matrix and fixed AO values."""

    deriv: int
    nfeats: int

    @abstractmethod
    def forward(self, dm: torch.Tensor, ao: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def vjp(self, ao: torch.Tensor, cotangent: torch.Tensor) -> torch.Tensor:
        """Apply the adjoint feature map to a feature-space cotangent."""

    @abstractmethod
    def to_dict(self, features: torch.Tensor) -> FeatureMap: ...


class MGGAFeatureFunction(LinearFeature):
    """Evaluate the requested linear meta-GGA density features."""

    def __init__(self, feature_spec: AOFeatureSpec) -> None:
        super().__init__()

        self._feature_spec = feature_spec
        self.deriv = feature_spec.nderiv
        self.nfeats = feature_spec.nfeats

    def to_dict(self, features: torch.Tensor) -> FeatureMap:
        """Convert a packed feature tensor to its named feature tensors."""
        return {
            feature: features[..., feature_slice, :].squeeze(-2)
            for feature, feature_slice in self._feature_spec
        }

    def forward(self, dm: torch.Tensor, ao: torch.Tensor) -> torch.Tensor:
        dm_view = dm.view(-1, dm.shape[-2], dm.shape[-1])
        dm_view = 0.5 * (dm_view + dm_view.transpose(-1, -2))

        features = torch.zeros(
            (dm_view.shape[0], self.nfeats, ao.shape[-1]),
            device=dm.device,
            dtype=dm.dtype,
        )

        packed_ao = PackedAO(ao)
        phi = packed_ao.value
        c0 = dm_view @ phi
        gradient_contraction = None
        if Feature.KIN in self._feature_spec or Feature.LAPL in self._feature_spec:
            gradient_contraction = features.new_zeros((dm_view.shape[0], ao.shape[-1]))
            for ao_gradient in packed_ao.gradient:
                ci = dm_view @ ao_gradient
                gradient_contraction += torch.sum(ci * ao_gradient[None], dim=-2)

        for feature, feature_slice in self._feature_spec:
            if feature == Feature.DENSITY:
                feature_values = torch.sum(c0 * phi[None], dim=-2).unsqueeze(-2)
            elif feature == Feature.GRAD:
                feature_values = 2 * torch.sum(
                    c0[:, None] * packed_ao.gradient[None], dim=-2
                )
            elif feature == Feature.KIN:
                assert gradient_contraction is not None
                feature_values = (0.5 * gradient_contraction).unsqueeze(-2)
            else:
                assert feature == Feature.LAPL
                assert gradient_contraction is not None
                laplacian = 2 * gradient_contraction
                for ao_hessian in packed_ao.diagonal_hessian:
                    laplacian += 2 * torch.sum(c0 * ao_hessian[None], dim=-2)
                feature_values = laplacian.unsqueeze(-2)
            features[..., feature_slice, :] = feature_values

        if dm.dim() == 2:
            return features.reshape((self.nfeats, -1))
        return features.reshape((*dm.shape[:-2], self.nfeats, -1))

    def vjp(self, ao: torch.Tensor, cotangent: torch.Tensor) -> torch.Tensor:
        """Apply the analytic adjoint of the linear MGGA feature map."""
        batch_shape = cotangent.shape[:-2]
        ngrids = cotangent.shape[-1]
        weights = cotangent.reshape(-1, self.nfeats, ngrids)
        packed_ao = PackedAO(ao)
        phi = packed_ao.value
        nao = phi.shape[-2]

        left = weights.new_zeros((weights.shape[0], nao, ngrids))
        derivative_weight = weights.new_zeros((weights.shape[0], 1, ngrids))
        for feature, feature_slice in self._feature_spec:
            feature_weight = weights[:, feature_slice, :]
            if feature == Feature.DENSITY:
                left += feature_weight * phi
            elif feature == Feature.GRAD:
                left += 2 * torch.sum(
                    feature_weight[:, :, None] * packed_ao.gradient[None], dim=1
                )
            elif feature == Feature.KIN:
                derivative_weight += 0.5 * feature_weight
            else:
                assert feature == Feature.LAPL
                derivative_weight += 2 * feature_weight
                for ao_hessian in packed_ao.diagonal_hessian:
                    left.addcmul_(feature_weight, ao_hessian, value=2)

        result = left @ phi.transpose(-1, -2)
        if Feature.KIN in self._feature_spec or Feature.LAPL in self._feature_spec:
            for ao_gradient in packed_ao.gradient:
                weighted_derivative = derivative_weight * ao_gradient
                result += weighted_derivative @ ao_gradient.transpose(-1, -2)

        result = 0.5 * (result + result.transpose(-1, -2))
        return result.reshape(*batch_shape, nao, nao)
