# SPDX-License-Identifier: MIT

"""Feature requirements and numerical-evaluation policy."""

from collections.abc import Iterable
from dataclasses import dataclass

from skala.features import Feature

_AO_FEATURES = frozenset(
    {
        Feature.DENSITY,
        Feature.GRAD,
        Feature.KIN,
        Feature.LAPL,
    }
)
_ATOMIC_LAYOUT_FEATURES = frozenset(
    {
        Feature.ATOMIC_GRID_WEIGHTS,
        Feature.ATOMIC_GRID_SIZES,
        Feature.ATOMIC_GRID_SIZE_BOUND_SHAPE,
    }
)


class FeatureSpec:
    """Normalized feature names and their evaluation requirements."""

    def __init__(self, names: Iterable[Feature]) -> None:
        self._names = frozenset(names)

    @property
    def names(self) -> frozenset[Feature]:
        """Return the normalized feature names."""
        return self._names

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FeatureSpec):
            return NotImplemented
        return self.names == other.names

    def __hash__(self) -> int:
        return hash(self.names)

    def requests(self, feature: Feature) -> bool:
        """Return whether a feature is requested."""
        return feature in self.names

    @property
    def with_density(self) -> bool:
        """Return whether density is requested."""
        return self.requests(Feature.DENSITY)

    @property
    def with_grad(self) -> bool:
        """Return whether the density gradient is requested."""
        return self.requests(Feature.GRAD)

    @property
    def with_kin(self) -> bool:
        """Return whether kinetic-energy density is requested."""
        return self.requests(Feature.KIN)

    @property
    def with_lapl(self) -> bool:
        """Return whether the density Laplacian is requested."""
        return self.requests(Feature.LAPL)

    @property
    def requires_ao_evaluation(self) -> bool:
        """Return whether AO-derived features are requested."""
        return bool(self.names & _AO_FEATURES)

    @property
    def mgga_feature_count(self) -> int:
        """Return the scalar width of the requested meta-GGA features."""
        return self.with_density + 3 * self.with_grad + self.with_kin + self.with_lapl

    @property
    def ao_derivative_order(self) -> int:
        """Return the highest AO derivative order needed by the features."""
        if Feature.LAPL in self.names:
            return 2
        if self.names & {Feature.GRAD, Feature.KIN}:
            return 1
        return 0

    @property
    def requires_atomic_layout(self) -> bool:
        """Return whether grid points must retain per-atom ordering."""
        return bool(self.names & _ATOMIC_LAYOUT_FEATURES)

    @property
    def supports_spatial_decomposition(self) -> bool:
        """Return whether spatial decomposition is supported."""
        return Feature.ATOMIC_GRID_SIZES in self.names


@dataclass(frozen=True)
class EvaluationPolicy:
    """Settings shared by dense and screened AO feature evaluation."""

    ao_block_size: int | None = None
    safety_fraction: float = 0.8
