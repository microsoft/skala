# SPDX-License-Identifier: MIT

from logging import getLogger
from typing import TYPE_CHECKING, Any

from gpu4pyscf.dft import gen_grid
from pyscf import gto

if TYPE_CHECKING:
    from skala.pyscf.screening import SpatialGridLayout

LOG = getLogger(__name__)


class SkalaGrids(gen_grid.Grids):  # type: ignore
    """GPU4PySCF grids with atom-major ordering and Skala layout caching."""

    _spatial_grid_layout: "SpatialGridLayout | None"
    _initializing: bool

    def __init__(self, mol: gto.Mole | None = None) -> None:
        super().__setattr__("_initializing", True)
        super().__init__(mol)
        super().__setattr__("alignment", 1)
        super().__setattr__("_initializing", False)

    def __setattr__(self, key: str, value: Any) -> None:
        if (
            key == "alignment"
            and value != 1
            and not getattr(self, "_initializing", False)
        ):
            raise ValueError(f"SkalaGrids alignment must be 1, got {value}")
        if key in {"coords", "weights", "cutoff"}:
            super().__setattr__("_spatial_grid_layout", None)
        super().__setattr__(key, value)

    def build(
        self,
        mol: gto.Mole | None = None,
        with_non0tab: bool = False,
        sort_grids: bool = True,
        sort_grids_of_each_atom: bool = False,
        **kwargs: Any,
    ) -> "SkalaGrids":
        if sort_grids or sort_grids_of_each_atom:
            LOG.debug("sorted grids not supported, forcing unsorted grids")
        return super().build(
            mol,
            with_non0tab,
            sort_grids=False,
            sort_grids_of_each_atom=False,
            **kwargs,
        )

    def get_cached_spatial_grid_layout(self) -> "SpatialGridLayout | None":
        """Return the spatial layout cached for the current grid state."""
        return getattr(self, "_spatial_grid_layout", None)

    def cache_spatial_grid_layout(self, layout: "SpatialGridLayout") -> None:
        """Cache a spatial layout until layout-defining grid state changes."""
        self._spatial_grid_layout = layout
