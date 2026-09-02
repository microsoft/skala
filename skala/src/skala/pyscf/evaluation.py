# SPDX-License-Identifier: MIT

"""Feature requirements and numerical-evaluation policy."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from skala.features import AO_FEATURES, Feature

_ATOMIC_LAYOUT_FEATURES = frozenset(
    {
        Feature.ATOMIC_GRID_WEIGHTS,
        Feature.ATOMIC_GRID_SIZES,
        Feature.ATOMIC_GRID_SIZE_BOUND_SHAPE,
    }
)


class FeatureSpec:
    """Normalized set of named molecular features."""

    def __init__(self, features: Iterable[Feature]) -> None:
        self._features = frozenset(features)

    def __iter__(self) -> Iterator[Feature]:
        return iter(self._features)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FeatureSpec):
            return NotImplemented
        return self._features == other._features

    def __hash__(self) -> int:
        return hash(self._features)

    def requests(self, feature: Feature) -> bool:
        """Return whether a feature is requested."""
        return feature in self._features

    def __or__(self, other: FeatureSpec | Iterable[Feature]) -> FeatureSpec:
        """Return a feature specification containing both operands."""
        return FeatureSpec(self._features | frozenset(other))

    @property
    def ao_features(self) -> frozenset[Feature]:
        """Return the requested AO-derived features."""
        return self._features & AO_FEATURES

    @property
    def requires_ao_evaluation(self) -> bool:
        """Return whether AO-derived features are requested."""
        return bool(self.ao_features)

    @property
    def requires_atomic_layout(self) -> bool:
        """Return whether grid points must retain per-atom ordering."""
        return bool(self._features & _ATOMIC_LAYOUT_FEATURES)

    @property
    def supports_spatial_decomposition(self) -> bool:
        """Return whether spatial decomposition is supported."""
        return Feature.ATOMIC_GRID_SIZES in self._features


@dataclass(frozen=True)
class EvaluationPolicy:
    """Settings shared by dense and screened AO feature evaluation."""

    ao_block_size: int | None = None
    safety_fraction: float = 0.8
