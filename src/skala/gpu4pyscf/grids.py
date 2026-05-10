# SPDX-License-Identifier: MIT

from logging import getLogger
from typing import Any

from gpu4pyscf.dft import gen_grid
from pyscf import gto

LOG = getLogger(__name__)


class UnsortableGrids(gen_grid.Grids):  # type: ignore
    def build(
        self, mol: gto.Mole | None = None, with_non0tab: bool = False, **kwargs: Any
    ) -> "UnsortableGrids":
        sort_grids = kwargs.pop("sort_grids", None)
        if sort_grids:
            LOG.debug("sorted grids not supported, forcing unsorted grids")
        return super().build(mol, with_non0tab, sort_grids=False, **kwargs)
