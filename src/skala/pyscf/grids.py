# SPDX-License-Identifier: MIT

from logging import getLogger

from pyscf.dft import gen_grid

LOG = getLogger(__name__)


class Grids(gen_grid.Grids):
    def build(self, mol=None, with_non0tab=False, **kwargs) -> "Grids":
        sort_grids = kwargs.pop("sort_grids", None)
        if sort_grids is None:
            LOG.debug("sorted grids not supported, forcing unsorted grids")
        super().build(mol, with_non0tab, sort_grids=False, **kwargs)
        return self
