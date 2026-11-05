# SPDX-License-Identifier: MIT

"""Test-only conversion from PySCF objects to SkalaXC Python objects."""

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import skalaxc

from pyscf import gto

SKALA_1_1_REV1_SHA256 = (
    "7f3e8622e1eb520ccd88a55464c3e359ac4d7e5ccbd1fb77a26afa1e1c20a5cd"
)
SKALA_1_1_REV1_CUDA_SHA256 = (
    "f848eae769dca91741a518ae7275d10caac398ab21db649f91bc1f136872f223"
)


@dataclass(frozen=True)
class ParityTolerances:
    """Absolute tolerances for fixed-density cross-backend observables."""

    energy: float
    scalar_potential: float
    spin_potential: float
    gradient: float


def pyscf_to_skalaxc(
    pyscf_molecule: gto.Mole,
) -> tuple[skalaxc.Molecule, skalaxc.BasisSet]:
    """Construct SkalaXC molecule and basis objects from a PySCF molecule."""
    molecule = skalaxc.Molecule()
    coordinates = pyscf_molecule.atom_coords(unit="Bohr")
    for atomic_number, center in zip(
        pyscf_molecule.atom_charges(), coordinates, strict=True
    ):
        molecule.append(
            skalaxc.Atom(
                int(atomic_number),
                float(center[0]),
                float(center[1]),
                float(center[2]),
            )
        )

    basis = skalaxc.BasisSet()
    for atom_index, (atom_label, _) in enumerate(pyscf_molecule._atom):
        center = coordinates[atom_index].tolist()
        for pyscf_shell in pyscf_molecule._basis[atom_label]:
            angular_momentum = int(pyscf_shell[0])
            primitives = pyscf_shell[1:]
            exponents = [float(primitive[0]) for primitive in primitives]
            for contraction_index in range(1, len(primitives[0])):
                coefficients = [
                    float(primitive[contraction_index]) for primitive in primitives
                ]
                basis.append(
                    skalaxc.Shell(
                        angular_momentum,
                        not pyscf_molecule.cart and angular_momentum != 1,
                        exponents,
                        coefficients,
                        center,
                        normalize=True,
                    )
                )

    return molecule, basis


def uks_density_channels(
    density: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Convert alpha/beta density matrices to scalar/spin-z matrices."""
    density_array = np.asarray(density, dtype=np.float64)
    if density_array.ndim != 3 or density_array.shape[0] != 2:
        raise ValueError(
            f"Expected UKS density shape (2, nbf, nbf), got {density_array.shape}"
        )
    if density_array.shape[1] != density_array.shape[2]:
        raise ValueError(
            f"Expected square UKS density matrices, got {density_array.shape}"
        )

    scalar_density = np.asfortranarray(density_array[0] + density_array[1])
    spin_density = np.asfortranarray(density_array[0] - density_array[1])
    return scalar_density, spin_density


def make_skalaxc_integrator(
    pyscf_molecule: gto.Mole,
    model: str,
    execution_space: skalaxc.ExecutionSpace = skalaxc.ExecutionSpace.HOST,
    grid_size: skalaxc.AtomicGridSize = skalaxc.AtomicGridSize.ULTRA_FINE,
    device_memory_fraction: float = 0.5,
    device_memory_cap_bytes: int = 1024**3,
) -> skalaxc.XCIntegrator:
    """Construct a weighted SkalaXC integrator directly from a PySCF molecule."""
    runtime_environment: Any = skalaxc.RuntimeEnvironment
    molecule, basis = pyscf_to_skalaxc(pyscf_molecule)
    grid = skalaxc.MolGridFactory.create_default(
        molecule,
        pruning_scheme=skalaxc.PruningScheme.UNPRUNED,
        radial_quad=skalaxc.RadialQuad.MURA_KNOWLES,
        grid_size=grid_size,
    )
    if skalaxc.CUDA_ENABLED and execution_space != skalaxc.ExecutionSpace.HOST:
        device_settings = skalaxc.DeviceRuntimeSettings()
        device_settings.memory_fraction = device_memory_fraction
        memory_cap_variable = "GAUXC_DEVICE_MEMORY_CAP"
        previous_memory_cap = os.environ.get(memory_cap_variable)
        if previous_memory_cap is None:
            os.environ[memory_cap_variable] = str(device_memory_cap_bytes)
        try:
            if skalaxc.MPI_ENABLED:
                from mpi4py import MPI

                runtime = runtime_environment(MPI.COMM_SELF, device_settings)
            else:
                runtime = runtime_environment(device_settings)
        finally:
            if previous_memory_cap is None:
                del os.environ[memory_cap_variable]
    elif skalaxc.MPI_ENABLED:
        from mpi4py import MPI

        runtime = runtime_environment(MPI.COMM_SELF)
    else:
        runtime = runtime_environment()

    load_balancer = skalaxc.LoadBalancerFactory(execution_space).get_instance(
        runtime, molecule, grid, basis
    )
    weights = skalaxc.MolecularWeightsFactory(execution_space).get_instance()
    weights.modify_weights(load_balancer)
    return skalaxc.XCIntegratorFactory(execution_space).get_instance(
        skalaxc.Functional(model), load_balancer
    )
