#!/usr/bin/env python3

"""
This script downloads the Skala model, as well as a reference LDA functional from HuggingFace.

The LDA functional computes -3 / 4 * (3 / math.pi) ** (1 / 3) * density.abs() ** (4 / 3),
and can be used to verify that the C++ integration is working correctly.
"""

import shutil

from huggingface_hub import hf_hub_download

from skala.functional._hashes import KNOWN_HASHES
from skala.functional.load import TracedFunctional

GRID_SIZE = "grid_size"
NUM_ATOMS = "num_atoms"
MAX_GRID_PER_ATOM = "max_grid_per_atom"

feature_shapes = {
    "density": [2, GRID_SIZE],
    "kin": [2, GRID_SIZE],
    "grad": [2, 3, GRID_SIZE],
    "grid_coords": [GRID_SIZE, 3],
    "grid_weights": [GRID_SIZE],
    "coarse_0_atomic_coords": [NUM_ATOMS, 3],
    "atomic_grid_weights": [GRID_SIZE],
    "atomic_grid_sizes": [NUM_ATOMS],
    "atomic_grid_size_bound_shape": [MAX_GRID_PER_ATOM, 0],
}

# The integer dtype features below are bookkeeping for the per-atom grid layout
# required by Skala's non-local model; they are not floating-point physical data.
feature_dtypes = {
    "density": "float64",
    "kin": "float64",
    "grad": "float64",
    "grid_coords": "float64",
    "grid_weights": "float64",
    "coarse_0_atomic_coords": "float64",
    "atomic_grid_weights": "float64",
    "atomic_grid_sizes": "int64",
    "atomic_grid_size_bound_shape": "int64",
}

feature_labels = {
    "density": "Electron density on grid, two spin channels",
    "kin": "Kinetic energy density on grid, two spin channels",
    "grad": "Gradient of electron density on grid, two spin channels",
    "grid_coords": "Coordinates of grid points",
    "grid_weights": "Weights of grid points",
    "coarse_0_atomic_coords": "Atomic coordinates",
    # The grid is the concatenation of per-atom atomic grids, in atom-major order.
    # atomic_grid_weights are the raw quadrature weights for integrating over each
    # individual atomic grid, i.e. they are NOT multiplied by the Becke partition
    # function (unlike grid_weights, which integrate the molecular energy). They are
    # used by the non-local model as the integration measure over each atom's grid.
    "atomic_grid_weights": "Per-atom-grid quadrature weights (not partition-weighted)",
    "atomic_grid_sizes": "Number of grid points belonging to each atom",
    # atomic_grid_size_bound_shape carries no data: it is an empty tensor whose first
    # dimension equals max(atomic_grid_sizes). It exists only to pass that static
    # padding bound as a tensor shape rather than a tensor value, which is an
    # implementation detail required by the LibTorch/TorchScript export of the model
    # (data-dependent output shapes are not allowed).
    "atomic_grid_size_bound_shape": (
        "Empty tensor whose first dim is max(atomic_grid_sizes); "
        "TorchScript/LibTorch implementation detail"
    ),
}


def main() -> None:
    for huggingface_repo_id, filename in (
        ("microsoft/skala-1.1", "skala-1.1-rev1.fun"),
        ("microsoft/skala-baselines", "ldax.fun"),
    ):
        output_path = filename.split("/")[-1]
        download_model(huggingface_repo_id, filename, output_path)


def download_model(huggingface_repo_id: str, filename: str, output_path: str) -> None:
    path = hf_hub_download(repo_id=huggingface_repo_id, filename=filename)
    shutil.copyfile(path, output_path)

    print(f"Downloaded the {filename} functional to {output_path}")

    expected_hash = KNOWN_HASHES.get((huggingface_repo_id, filename))
    fun = TracedFunctional.load(output_path, expected_hash=expected_hash)

    print("\nExpected inputs:")
    for feature in fun.features:
        print(
            f"- {feature} {feature_shapes[feature]} in {feature_dtypes[feature]} ({feature_labels[feature]})"
        )

    print(f"\nExpected D3 dispersion settings: {fun.expected_d3_settings}\n")


if __name__ == "__main__":
    main()
