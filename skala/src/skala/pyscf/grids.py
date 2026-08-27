# SPDX-License-Identifier: MIT

from logging import getLogger
from typing import Any

import torch
from pyscf.dft import gen_grid

from pyscf import gto
from skala.pyscf.spatial_grid_layout import (
    CPU_AO_SCREENING_BLOCK_SIZE,
    SpatialGridLayout,
    prepare_spatial_grid_layout,
)

LOG = getLogger(__name__)


class SkalaGrids(gen_grid.Grids):  # type: ignore[misc]
    """PySCF grids with atom-major ordering and Skala layout caching.

    Configure and build the grid before preparing its spatial layout. Preparing
    the layout marks ``coords`` and ``weights`` read-only; assigning replacements
    or changing ``cutoff`` invalidates the cache. Restoring array writeability or
    changing other grid internals between layout requests can produce incoherent
    source-grid, permutation, and screening state.
    """

    _spatial_grid_layout: "SpatialGridLayout | None"
    _initializing: bool

    def __init__(self, mol: gto.Mole | None = None) -> None:
        # PySCF installs a non-unit alignment during its constructor. Permit that
        # transient value, then establish the Skala cache and alignment invariants.
        super().__setattr__("_initializing", True)
        super().__init__(mol)
        super().__setattr__("_spatial_grid_layout", None)
        super().__setattr__("alignment", 1)
        super().__setattr__("_initializing", False)

    def __setattr__(self, key: str, value: Any) -> None:
        # Alignment padding would break the exact atom-major grid layout expected
        # by Skala, so only the base-class constructor may set a non-unit value.
        if (
            key == "alignment"
            and value != 1
            and not getattr(self, "_initializing", False)
        ):
            raise ValueError(f"SkalaGrids alignment must be 1, got {value}")
        # The spatial permutations and screening data are derived from these
        # attributes and must be rebuilt after any assignment.
        if key in {"coords", "weights", "cutoff"}:
            super().__setattr__("_spatial_grid_layout", None)
        super().__setattr__(key, value)

    def build(
        self,
        mol: gto.Mole | None = None,
        with_non0tab: bool = False,
        sort_grids: bool = True,
        **kwargs: Any,
    ) -> "SkalaGrids":
        if sort_grids:
            LOG.debug("sorted grids not supported, forcing unsorted grids")
        result = super().build(mol, with_non0tab, sort_grids=False, **kwargs)
        assert isinstance(result, SkalaGrids)
        return result

    def prepare_spatial_grid_layout(
        self,
        mol: gto.Mole,
        device: torch.device,
    ) -> SpatialGridLayout:
        """Return the cached spatial layout, creating it when needed."""
        if self._spatial_grid_layout is None:
            spatial_grid_layout = prepare_spatial_grid_layout(
                mol,
                self,
                CPU_AO_SCREENING_BLOCK_SIZE,
                device,
            )
            # Freeze the source arrays before publishing their derived layout.
            # Reassignment remains supported and invalidates the cache above.
            self.coords.setflags(write=False)
            self.weights.setflags(write=False)
            self._spatial_grid_layout = spatial_grid_layout
        return self._spatial_grid_layout
