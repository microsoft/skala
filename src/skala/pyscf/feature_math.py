# SPDX-License-Identifier: MIT

"""Raw density-feature mathematics and model formatting."""

from abc import ABC, abstractmethod

import torch
from torch import nn

from skala.features import Feature, FeatureMap
from skala.pyscf.evaluation import FeatureSpec


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

    def __init__(self, feature_spec: FeatureSpec) -> None:
        super().__init__()

        if not feature_spec.requires_ao_evaluation:
            raise ValueError("At least one AO-derived feature must be selected.")
        self.feature_spec = feature_spec
        self.deriv = feature_spec.ao_derivative_order
        self.nfeats = feature_spec.mgga_feature_count

    def to_dict(self, features: torch.Tensor) -> FeatureMap:
        """Convert a packed feature tensor to its named feature tensors."""
        feature_index = 0
        feature_dict: FeatureMap = {}
        if self.feature_spec.with_density:
            feature_dict[Feature.DENSITY] = features[..., feature_index, :]
            feature_index += 1
        if self.feature_spec.with_grad:
            feature_dict[Feature.GRAD] = features[
                ..., feature_index : feature_index + 3, :
            ]
            feature_index += 3
        if self.feature_spec.with_kin:
            feature_dict[Feature.KIN] = features[..., feature_index, :]
            feature_index += 1
        if self.feature_spec.with_lapl:
            feature_dict[Feature.LAPL] = features[..., feature_index, :]
        return feature_dict

    def forward(self, dm: torch.Tensor, ao: torch.Tensor) -> torch.Tensor:
        dm_view = dm.view(-1, dm.shape[-2], dm.shape[-1])
        dm_view = 0.5 * (dm_view + dm_view.transpose(-1, -2))

        features = torch.zeros(
            (dm_view.shape[0], self.nfeats, ao.shape[-1]),
            device=dm.device,
            dtype=dm.dtype,
        )

        if self.deriv == 0:
            c0 = dm_view @ ao
            features[..., 0, :] = torch.sum(c0 * ao[None, :, :], dim=-2)
            if len(dm.shape) == 2:
                return features.reshape((self.nfeats, -1))
            return features.reshape((*dm.shape[:-2], self.nfeats, -1))

        c0 = dm_view @ ao[0]

        feature_index = 0
        if self.feature_spec.with_density:
            features[..., feature_index, :] = torch.sum(c0 * ao[0, None, :, :], dim=-2)
            feature_index += 1

        if self.feature_spec.with_grad:
            for component in range(3):
                features[..., feature_index, :] = 2 * torch.sum(
                    c0 * ao[component + 1, None, :, :], dim=-2
                )
                feature_index += 1

        if self.feature_spec.with_kin or self.feature_spec.with_lapl:
            for component in range(3):
                ci = dm_view @ ao[component + 1]
                features[..., feature_index, :] += 0.5 * torch.sum(
                    ci * ao[component + 1, None, :, :], dim=-2
                )

            if self.feature_spec.with_kin:
                feature_index += 1
                if self.feature_spec.with_lapl:
                    features[..., feature_index, :] = (
                        4 * features[..., feature_index - 1, :]
                    )
            else:
                features[..., feature_index, :] *= 4.0

            if self.feature_spec.with_lapl:
                for component in (4, 7, 9):
                    features[..., feature_index, :] += 2 * torch.sum(
                        c0 * ao[component, None, :, :], dim=-2
                    )

        if len(dm.shape) == 2:
            return features.reshape((self.nfeats, -1))
        return features.reshape((*dm.shape[:-2], self.nfeats, -1))

    def vjp(self, ao: torch.Tensor, cotangent: torch.Tensor) -> torch.Tensor:
        """Apply the analytic adjoint of the linear MGGA feature map."""
        batch_shape = cotangent.shape[:-2]
        ngrids = cotangent.shape[-1]
        weights = cotangent.reshape(-1, self.nfeats, ngrids)
        phi = ao if self.deriv == 0 else ao[0]
        nao = phi.shape[-2]

        if self.deriv == 0:
            result = (weights[:, 0, None, :] * phi) @ phi.transpose(-1, -2)
            return result.reshape(*batch_shape, nao, nao)

        left = weights.new_zeros((weights.shape[0], nao, ngrids))
        feature_index = 0
        if self.feature_spec.with_density:
            left += weights[:, feature_index, None, :] * phi
            feature_index += 1

        if self.feature_spec.with_grad:
            for component in range(3):
                left.addcmul_(
                    weights[:, feature_index + component, None, :],
                    ao[component + 1],
                    value=2,
                )
            feature_index += 3

        derivative_weight = weights.new_zeros((weights.shape[0], ngrids))
        if self.feature_spec.with_kin:
            derivative_weight += 0.5 * weights[:, feature_index]
            feature_index += 1

        if self.feature_spec.with_lapl:
            laplacian_weight = weights[:, feature_index]
            derivative_weight += 2 * laplacian_weight
            for component in (4, 7, 9):
                left.addcmul_(laplacian_weight[:, None, :], ao[component], value=2)

        result = left @ phi.transpose(-1, -2)
        if self.feature_spec.with_kin or self.feature_spec.with_lapl:
            for component in range(1, 4):
                weighted_derivative = derivative_weight[:, None, :] * ao[component]
                result += weighted_derivative @ ao[component].transpose(-1, -2)

        result = 0.5 * (result + result.transpose(-1, -2))
        return result.reshape(*batch_shape, nao, nao)
