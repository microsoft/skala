# SPDX-License-Identifier: MIT

"""Feature requirements and numerical-evaluation policy."""

from collections.abc import Iterable
from dataclasses import dataclass

_MGGA_FEATURES = frozenset({"density", "grad", "kin", "lapl"})
_ATOMIC_LAYOUT_FEATURES = frozenset(
    {
        "atomic_grid_weights",
        "atomic_grid_sizes",
        "atomic_grid_size_bound_shape",
    }
)


@dataclass(frozen=True, init=False)
class FeatureSpec:
    """Normalized feature names and their evaluation requirements."""

    names: frozenset[str]

    def __init__(self, names: Iterable[str]) -> None:
        object.__setattr__(self, "names", frozenset(names))

    def requests(self, feature: str) -> bool:
        """Return whether a feature is requested."""
        return feature in self.names

    @property
    def with_density(self) -> bool:
        """Return whether density is requested."""
        return self.requests("density")

    @property
    def with_grad(self) -> bool:
        """Return whether the density gradient is requested."""
        return self.requests("grad")

    @property
    def with_kin(self) -> bool:
        """Return whether kinetic-energy density is requested."""
        return self.requests("kin")

    @property
    def with_lapl(self) -> bool:
        """Return whether the density Laplacian is requested."""
        return self.requests("lapl")

    @property
    def requires_mgga(self) -> bool:
        """Return whether AO-based meta-GGA features are requested."""
        return bool(self.names & _MGGA_FEATURES)

    @property
    def mgga_feature_count(self) -> int:
        """Return the scalar width of the requested meta-GGA features."""
        return self.with_density + 3 * self.with_grad + self.with_kin + self.with_lapl

    @property
    def ao_derivative_order(self) -> int:
        """Return the highest AO derivative order needed by the features."""
        if "lapl" in self.names:
            return 2
        if self.names & {"grad", "kin"}:
            return 1
        return 0

    @property
    def requires_atomic_layout(self) -> bool:
        """Return whether grid points must retain per-atom ordering."""
        return bool(self.names & _ATOMIC_LAYOUT_FEATURES)

    @property
    def supports_screened_evaluation(self) -> bool:
        """Return whether atom-aligned screened evaluation is supported."""
        return "atomic_grid_sizes" in self.names


@dataclass(frozen=True)
class EvaluationPolicy:
    """Settings shared by dense and screened AO feature evaluation."""

    ao_block_size: int | None = None
    safety_fraction: float = 0.8
