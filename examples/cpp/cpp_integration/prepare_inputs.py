#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch
from pyscf import dft, gto
from pyscf.dft import gen_grid

from skala.functional.traditional import LDA
from skala.pyscf.features import generate_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=".",
        type=Path,
        help="Output directory for generated feature files.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    molecule = gto.Mole(
        atom="H 0 0 0; H 0 0 1",
        basis="def2-qzvp",
        verbose=0,
    ).build()

    # Create a set of meta-GGA features for this molecule.
    dm = get_density_matrix(molecule)
    grid = gen_grid.Grids(molecule)
    grid.level = 3
    grid.build()
    features = generate_features(molecule, dm, grid)

    # Add a feature called `coarse_0_atomic_coords` containing the atomic coordinates.
    features["coarse_0_atomic_coords"] = torch.from_numpy(molecule.atom_coords())

    # Save all features as individual .pt files.
    for key, value in features.items():
        torch.save(value, str(args.output_dir / f"{key}.pt"))

    print(f"Saved features to {args.output_dir}")

    lda_exc = LDA().get_exc(features)
    print(f"For reference, LDAx Exc = {lda_exc.item()}")


def get_density_matrix(mol: gto.Mole) -> torch.Tensor:
    """Computes an example density matrix for a given molecule using PySCF."""
    ks = dft.RKS(mol, xc="b3lyp5")
    ks = ks.density_fit()
    ks.kernel()
    dm = torch.from_numpy(ks.make_rdm1())
    return dm


if __name__ == "__main__":
    main()
