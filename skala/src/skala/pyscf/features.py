# SPDX-License-Identifier: MIT

"""
Methods for generating and manipulating density features.
"""

import numpy as np
import torch
from torch import Tensor

from pyscf import gto
from skala.features import Feature, FeatureMap
from skala.pyscf.backend import Grid, from_numpy_or_cupy
from skala.pyscf.evaluation import FeatureSpec


def get_grid_features(
    mol: gto.Mole,
    dm: Tensor,
    grids: Grid,
    feature_spec: FeatureSpec,
) -> FeatureMap:
    grid_features: FeatureMap = {}

    if feature_spec.requests(Feature.GRID_COORDS):
        grid_features[Feature.GRID_COORDS] = from_numpy_or_cupy(
            grids.coords, device=dm.device, dtype=dm.dtype
        )

    if feature_spec.requests(Feature.GRID_WEIGHTS):
        grid_features[Feature.GRID_WEIGHTS] = from_numpy_or_cupy(
            grids.weights, device=dm.device, dtype=dm.dtype
        )

    if feature_spec.requests(Feature.COARSE_0_ATOMIC_COORDS):
        grid_features[Feature.COARSE_0_ATOMIC_COORDS] = from_numpy_or_cupy(
            mol.atom_coords(), device=dm.device, dtype=dm.dtype
        )

    if feature_spec.requires_atomic_layout:
        atom_grids_tab = grids.gen_atomic_grids(
            mol, grids.atom_grid, grids.radi_method, grids.level, grids.prune
        )
        sizes = [len(atom_grids_tab[mol.atom_symbol(ia)][1]) for ia in range(mol.natm)]

        n_atomic = sum(sizes)
        n_grid = grids.weights.shape[0]
        if n_atomic != n_grid:
            raise ValueError(
                f"Grid size mismatch: sum of atomic grid sizes ({n_atomic}) does not match "
                f"total grid points ({n_grid}). This is likely caused by grid alignment padding "
                f"(grids.alignment={getattr(grids, 'alignment', '?')}). "
                f"Set grids.alignment = 1 before building grids to disable padding."
            )

        if feature_spec.requests(Feature.ATOMIC_GRID_SIZES):
            grid_features[Feature.ATOMIC_GRID_SIZES] = torch.tensor(
                sizes, dtype=torch.long, device=dm.device
            )

        if feature_spec.requests(Feature.ATOMIC_GRID_SIZE_BOUND_SHAPE):
            max_size = max(sizes)
            grid_features[Feature.ATOMIC_GRID_SIZE_BOUND_SHAPE] = torch.zeros(
                max_size, 0, dtype=torch.long, device=dm.device
            )

        if feature_spec.requests(Feature.ATOMIC_GRID_WEIGHTS):
            raw_weights = np.concatenate(
                [atom_grids_tab[mol.atom_symbol(ia)][1] for ia in range(mol.natm)]
            )
            grid_features[Feature.ATOMIC_GRID_WEIGHTS] = from_numpy_or_cupy(
                raw_weights, device=dm.device, dtype=dm.dtype
            )

    return grid_features
