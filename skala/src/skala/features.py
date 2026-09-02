# SPDX-License-Identifier: MIT

"""Names of built-in molecular features."""

from collections.abc import Iterable
from enum import Enum
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from torch import Tensor


class Feature(str, Enum):  # noqa: UP042 - Python 3.10-compatible StrEnum
    """String-compatible names of features understood by Skala."""

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

    __str__ = str.__str__


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


FeatureMap: TypeAlias = dict[Feature, "Tensor"]
