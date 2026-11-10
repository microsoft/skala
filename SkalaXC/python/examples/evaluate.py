"""Evaluate a trusted bundled UKS density fixture with the installed binding."""

import argparse

import h5py
import numpy as np

import skalaxc


def main() -> None:
    """Run the host evaluation pipeline and report its outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture")
    parser.add_argument("model", nargs="?", default="PBE")
    args = parser.parse_args()

    if skalaxc.MPI_ENABLED:
        from mpi4py import MPI

        runtime = skalaxc.RuntimeEnvironment(MPI.COMM_WORLD)
    else:
        runtime = skalaxc.RuntimeEnvironment()
    molecule = skalaxc.Molecule.from_hdf5(args.fixture)
    basis = skalaxc.BasisSet.from_hdf5(args.fixture)
    grid = skalaxc.MolGridFactory.create_default(molecule)
    load_balancer = skalaxc.LoadBalancerFactory(
        skalaxc.ExecutionSpace.HOST
    ).get_instance(runtime, molecule, grid, basis)
    weights = skalaxc.MolecularWeightsFactory(
        skalaxc.ExecutionSpace.HOST
    ).get_instance()
    weights.modify_weights(load_balancer)
    integrator = skalaxc.XCIntegratorFactory(skalaxc.ExecutionSpace.HOST).get_instance(
        skalaxc.Functional(args.model), load_balancer
    )
    with h5py.File(args.fixture) as handle:
        scalar = np.asfortranarray(handle["/DENSITY_SCALAR"], dtype=np.float64)
        spin = np.asfortranarray(handle["/DENSITY_Z"], dtype=np.float64)
    energy, scalar_potential, _ = integrator.eval_exc_vxc(scalar, spin)
    gradient = integrator.eval_exc_grad(scalar, spin)
    print(f"EXC={energy:.10f} |VXC_scalar|_F={np.linalg.norm(scalar_potential):.6e}")
    print(f"gradient shape={gradient.shape}")


if __name__ == "__main__":
    main()
