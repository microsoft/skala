# SPDX-License-Identifier: MIT

"""Names of built-in molecular features."""

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


FeatureMap: TypeAlias = dict[Feature, "Tensor"]
