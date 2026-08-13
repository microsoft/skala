# SPDX-License-Identifier: MIT

from logging import getLogger
from typing import Any

import torch
from pyscf import gto
from pyscf.dft import gen_grid

from skala.pyscf.spatial_grid_layout import (
    CPU_AO_SCREENING_BLOCK_SIZE,
    SpatialGridLayout,
    prepare_spatial_grid_layout,
)

LOG = getLogger(__name__)


class SkalaGrids(gen_grid.Grids):  # type: ignore
    """PySCF grids with atom-major ordering and Skala layout caching."""

    _spatial_grid_layout: "SpatialGridLayout | None"
    _initializing: bool

    def __init__(self, mol: gto.Mole | None = None) -> None:
        super().__setattr__("_initializing", True)
        super().__init__(mol)
        super().__setattr__("_spatial_grid_layout", None)
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
        **kwargs: Any,
    ) -> "SkalaGrids":
        if sort_grids:
            LOG.debug("sorted grids not supported, forcing unsorted grids")
        return super().build(mol, with_non0tab, sort_grids=False, **kwargs)

    def prepare_spatial_grid_layout(
        self,
        mol: gto.Mole,
        device: torch.device,
    ) -> SpatialGridLayout:
        """Return the cached spatial layout, creating it when needed."""
        if self._spatial_grid_layout is None:
            self._spatial_grid_layout = prepare_spatial_grid_layout(
                mol,
                self,
                CPU_AO_SCREENING_BLOCK_SIZE,
                device,
            )
        return self._spatial_grid_layout
