# SPDX-License-Identifier: MIT

from logging import getLogger
from typing import Any

import torch
from gpu4pyscf.dft import gen_grid, numint
from typing_extensions import override

from pyscf import gto
from skala.pyscf.spatial_grid_layout import (
    SpatialGridLayout,
    prepare_spatial_grid_layout,
)

LOG = getLogger(__name__)


class SkalaGrids(gen_grid.Grids):
    """GPU4PySCF grids with atom-major ordering and Skala layout caching.

    Configure and build the grid before preparing its spatial layout. Assigning
    ``coords``, ``weights``, or ``cutoff`` invalidates the cached layout, but
    CuPy provides no equivalent to NumPy's writeability flag. Mutating arrays in
    place or changing other grid internals between layout requests therefore
    bypasses invalidation and can produce incoherent source-grid, permutation,
    and screening state.
    """

    # Initialized by the borrowed CPU constructor, not visible on the GPU base class.
    atom_grid: dict[str, tuple[int, int] | list[int]] | tuple[int, int] | list[int]

    _spatial_grid_layout: "SpatialGridLayout | None"
    _initializing: bool

    def __init__(self, mol: gto.Mole | None = None) -> None:
        # GPU4PySCF installs a non-unit alignment during its constructor. Permit
        # that transient value, then establish the Skala cache and alignment invariants.
        super().__setattr__("_initializing", True)
        # GPU4PySCF assigns the CPU Grids constructor to its unrelated GPU class.
        # pyrefly: ignore[bad-argument-type]
        super().__init__(mol)
        super().__setattr__("_spatial_grid_layout", None)
        super().__setattr__("alignment", 1)
        super().__setattr__("_initializing", False)

    @override
    def __setattr__(self, key: str, val: Any) -> None:
        # Alignment padding would break the exact atom-major grid layout expected
        # by Skala, so only the base-class constructor may set a non-unit value.
        if (
            key == "alignment"
            and val != 1
            and not getattr(self, "_initializing", False)
        ):
            raise ValueError(f"SkalaGrids alignment must be 1, got {val}")
        # The spatial permutations and screening data are derived from these
        # attributes and must be rebuilt after any assignment.
        if key in {"coords", "weights", "cutoff"}:
            super().__setattr__("_spatial_grid_layout", None)
        super().__setattr__(key, val)

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
        result = super().build(
            mol,
            with_non0tab,
            sort_grids=False,
            sort_grids_of_each_atom=False,
            **kwargs,
        )
        assert isinstance(result, SkalaGrids)
        return result

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
                int(numint.MIN_BLK_SIZE),
                device,
            )
        return self._spatial_grid_layout
