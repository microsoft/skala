# SPDX-License-Identifier: MIT

"""
Methods for generating and manipulating density features.
"""

import numpy as np
import torch
from torch import Tensor

from pyscf import gto
from skala.features import Feature, FeatureMap
from skala.pyscf import ao_evaluation, feature_math
from skala.pyscf.backend import Grid, from_numpy_or_cupy
from skala.pyscf.evaluation import EvaluationPolicy, FeatureSpec

DEFAULT_FEATURES = [
    Feature.DENSITY,
    Feature.KIN,
    Feature.GRAD,
    Feature.GRID_COORDS,
    Feature.GRID_WEIGHTS,
]
DEFAULT_FEATURES_SET = set(DEFAULT_FEATURES)


def generate_features(
    mol: gto.Mole,
    dm: Tensor,
    grids: Grid,
    features: set[Feature] | None = None,
    chunk_size: int | None = None,
    max_memory: int = 2000,
    gpu: bool = False,
) -> FeatureMap:
    """Generate density features for a given molecule. The density features are stored in a dictionary
    with the keys matching the requested features.

    Parameters
    ----------
    mol: gto.Mole
      the molecule
    dm: Tensor
      the density matrix
    grids: Grid
      the grid
    features: set[str] | None
      the requested features
    chunk_size: int | None
        a manually specified chunk size for processing the grids, if None the chunk size is determined automatically
    max_memory: int
      the maximum memory to use for calculating the features
    gpu: bool
        whether to use the GPU(4pyscf) for calculations

    Returns
    -------
    dict[str, Tensor]
        A dictionary containing the requested features. The keys are the feature names,
        and the values are the corresponding tensors.
    """
    feature_spec = FeatureSpec(DEFAULT_FEATURES_SET if features is None else features)
    evaluation_policy = EvaluationPolicy(ao_block_size=chunk_size)

    # if dm is a 3D tensor, then we have a spin-polarized system
    is_spin_polarized = len(dm.shape) == 3

    if gpu and dm.device.type != "cuda":
        raise ValueError("Density matrix must be on the GPU when gpu=True.")

    mol_features = get_grid_features(mol, dm, grids, feature_spec)

    if feature_spec.requires_ao_evaluation:
        mgga_features = ao_evaluation.auto_chunk(
            dm,
            mol,
            grids,
            feature_math.MGGAFeatureFunction(feature_spec),
            block_size=evaluation_policy.ao_block_size,
            max_memory=max_memory,
            gpu=gpu,
        )

        for feature in mgga_features:
            mol_features[feature] = feature_math.maybe_expand_and_divide(
                mgga_features[feature], not is_spin_polarized, 2
            )

    return mol_features


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
