# SPDX-License-Identifier: MIT

"""Names of built-in molecular features."""

from collections.abc import Iterable, Iterator
from enum import StrEnum
from typing import TypeAlias

from torch import Tensor


class Feature(StrEnum):
    """features understood by Skala."""

    DENSITY = "density"
    GRAD = "grad"
    KIN = "kin"
    LAPL = "lapl"
    GRID_COORDS = "grid_coords"
    GRID_WEIGHTS = "grid_weights"
    ATOMIC_GRID_WEIGHTS = "atomic_grid_weights"
    ATOMIC_GRID_SIZES = "atomic_grid_sizes"
    ATOMIC_GRID_SIZE_BOUND_SHAPE = "atomic_grid_size_bound_shape"
    COARSE_0_ATOMIC_COORDS = "coarse_0_atomic_coords"


AO_FEATURES = frozenset(
    {
        Feature.DENSITY,
        Feature.GRAD,
        Feature.KIN,
        Feature.LAPL,
    }
)


def ao_derivative_order(features: Iterable[Feature]) -> int:
    """Return the highest AO derivative order required by the features.

    Args:
        features: Molecular features to inspect.

    Returns:
        Required AO derivative order, or zero when no AO feature is present.
    """
    ao_features = AO_FEATURES & set(features)
    if Feature.LAPL in ao_features:
        return 2
    if ao_features & {Feature.GRAD, Feature.KIN}:
        return 1
    return 0


class AOFeatureSpec:
    """Normalized non-empty set of AO-derived features."""

    def __init__(self, features: Iterable[Feature]) -> None:
        self._features = frozenset(features)
        unsupported = self._features - AO_FEATURES
        if unsupported:
            unsupported_names = ", ".join(
                sorted(str(feature) for feature in unsupported)
            )
            raise ValueError(f"Unsupported AO features: {unsupported_names}")
        if not self._features:
            raise ValueError("At least one AO-derived feature must be selected.")

        self._feature_slices: dict[Feature, slice] = {}
        feature_index = 0
        for feature, width in (
            (Feature.DENSITY, 1),
            (Feature.GRAD, 3),
            (Feature.KIN, 1),
            (Feature.LAPL, 1),
        ):
            if feature in self._features:
                self._feature_slices[feature] = slice(
                    feature_index, feature_index + width
                )
                feature_index += width
        self._nfeats = feature_index

    def __contains__(self, feature: object) -> bool:
        return feature in self._features

    def __iter__(self) -> Iterator[tuple[Feature, slice]]:
        return iter(self._feature_slices.items())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AOFeatureSpec):
            return NotImplemented
        return self._features == other._features

    def __hash__(self) -> int:
        return hash(self._features)

    @property
    def nderiv(self) -> int:
        """Return the required AO derivative order."""
        return ao_derivative_order(self._features)

    @property
    def nfeats(self) -> int:
        """Return the number of packed scalar feature channels."""
        return self._nfeats


FeatureMap: TypeAlias = dict[Feature, Tensor]
