from __future__ import annotations

import gc
import sys

import h5py
import numpy as np
import pytest
import skalaxc
from mpi4py import MPI
from test_integration import REFERENCE_DATA, build_integrator

pytestmark = pytest.mark.skipif(
    not skalaxc.MPI_ENABLED, reason="SkalaXC was built without MPI"
)


def test_runtime_requires_valid_explicit_intracommunicator() -> None:
    runtime = skalaxc.RuntimeEnvironment(MPI.COMM_SELF)
    assert runtime.rank == 0
    assert runtime.size == 1

    with pytest.raises(TypeError):
        skalaxc.RuntimeEnvironment()
    with pytest.raises(TypeError, match="explicit mpi4py"):
        skalaxc.RuntimeEnvironment(None)
    with pytest.raises(ValueError, match="COMM_NULL"):
        skalaxc.RuntimeEnvironment(MPI.COMM_NULL)

    freed = MPI.COMM_SELF.Dup()
    freed.Free()
    with pytest.raises(ValueError, match="freed"):
        skalaxc.RuntimeEnvironment(freed)


def test_integrator_retains_communicator_owner() -> None:
    communicator = MPI.COMM_SELF.Dup()
    references_before = sys.getrefcount(communicator)
    runtime = skalaxc.RuntimeEnvironment(communicator)
    assert sys.getrefcount(communicator) == references_before + 1

    integrator = build_integrator(
        REFERENCE_DATA / "skala_he_def2qzvp_lda_uks.hdf5",
        runtime=runtime,
    )
    del runtime
    gc.collect()
    assert sys.getrefcount(communicator) >= references_before + 1

    del integrator
    gc.collect()
    assert sys.getrefcount(communicator) == references_before
    communicator.Free()


def test_explicit_subcommunicator_evaluation() -> None:
    world = MPI.COMM_WORLD
    communicator = world.Split(color=0, key=world.size - world.rank)
    try:
        runtime = skalaxc.RuntimeEnvironment(communicator)
        assert runtime.rank == communicator.rank
        assert runtime.size == communicator.size
        fixture = REFERENCE_DATA / "skala_he_def2qzvp_lda_uks.hdf5"
        integrator = build_integrator(fixture, runtime=runtime)
        with h5py.File(fixture) as reference:
            scalar_density = np.asfortranarray(reference["/DENSITY_SCALAR"])
            spin_density = np.asfortranarray(reference["/DENSITY_Z"])
        energy, _, _ = integrator.eval_exc_vxc(scalar_density, spin_density)
        energies = communicator.allgather(energy)
        assert energies == pytest.approx([energies[0]] * communicator.size)
    finally:
        communicator.Free()


def test_intercommunicator_is_rejected() -> None:
    world = MPI.COMM_WORLD
    if world.size != 2:
        pytest.skip("intercommunicator construction requires exactly two ranks")
    local = world.Split(color=world.rank, key=0)
    intercommunicator = local.Create_intercomm(
        0, world, remote_leader=1 - world.rank, tag=717
    )
    try:
        with pytest.raises(ValueError, match="intercommunicators"):
            skalaxc.RuntimeEnvironment(intercommunicator)
    finally:
        intercommunicator.Free()
        local.Free()
