#!/usr/bin/env python3

"""
This script downloads the Skala model, as well as a reference LDA functional from HuggingFace.

The LDA functional computes -3 / 4 * (3 / math.pi) ** (1 / 3) * density.abs() ** (4 / 3),
and can be used to verify that the C++ integration is working correctly.
"""

import shutil

from huggingface_hub import hf_hub_download

from skala.functional.load import TracedFunctional

GRID_SIZE = "grid_size"
NUM_ATOMS = "num_atoms"

feature_shapes = {
    "density": [2, GRID_SIZE],
    "kin": [2, GRID_SIZE],
    "grad": [2, 3, GRID_SIZE],
    "grid_coords": [GRID_SIZE, 3],
    "grid_weights": [GRID_SIZE],
    "coarse_0_atomic_coords": [NUM_ATOMS, 3],
}

feature_labels = {
    "density": "Electron density on grid, two spin channels",
    "kin": "Kinetic energy density on grid, two spin channels",
    "grad": "Gradient of electron density on grid, two spin channels",
    "grid_coords": "Coordinates of grid points",
    "grid_weights": "Weights of grid points",
    "coarse_0_atomic_coords": "Atomic coordinates",
}


def main() -> None:
    huggingface_repo_id = "microsoft/skala"
    for filename in ("skala-1.0.fun", "baselines/ldax.fun"):
        output_path = filename.split("/")[-1]
        download_model(huggingface_repo_id, filename, output_path)


def download_model(huggingface_repo_id: str, filename: str, output_path: str) -> None:
    path = hf_hub_download(repo_id=huggingface_repo_id, filename=filename)
    shutil.copyfile(path, output_path)

    print(f"Downloaded the {filename} functional to {output_path}")

    fun = TracedFunctional.load(output_path)

    print("\nExpected inputs:")
    for feature in fun.features:
        print(
            f"- {feature} {feature_shapes[feature]} in float64 ({feature_labels[feature]})"
        )

    print(f"\nExpected D3 dispersion settings: {fun.expected_d3_settings}\n")


if __name__ == "__main__":
    main()
