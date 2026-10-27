from __future__ import annotations

import numpy as np
import pytest
import skalaxc

pytestmark = pytest.mark.skipif(
    not skalaxc.CUDA_ENABLED, reason="SkalaXC was built without CUDA"
)


def make_h2_sto3g(
    x_displacement: float = 0.0,
) -> tuple[skalaxc.Molecule, skalaxc.BasisSet]:
    molecule = skalaxc.Molecule()
    molecule.append(skalaxc.Atom(1, -0.252 - x_displacement, 0.336, -0.56))
    molecule.append(skalaxc.Atom(1, 0.252 + x_displacement, -0.336, 0.56))

    exponents = [3.42525091, 0.62391373, 0.16885540]
    coefficients = [0.15432897, 0.53532814, 0.44463454]
    basis = skalaxc.BasisSet()
    for atom in (molecule[0], molecule[1]):
        basis.append(
            skalaxc.Shell(
                0,
                False,
                exponents,
                coefficients,
                [atom.x, atom.y, atom.z],
            )
        )
    return molecule, basis


def make_runtime(execution_space: skalaxc.ExecutionSpace) -> skalaxc.RuntimeEnvironment:
    if skalaxc.MPI_ENABLED:
        from mpi4py import MPI

        if execution_space == skalaxc.ExecutionSpace.DEVICE:
            settings = skalaxc.DeviceRuntimeSettings()
            settings.memory_fraction = 0.8
            return skalaxc.RuntimeEnvironment(MPI.COMM_SELF, settings)
        return skalaxc.RuntimeEnvironment(MPI.COMM_SELF)

    if execution_space == skalaxc.ExecutionSpace.DEVICE:
        settings = skalaxc.DeviceRuntimeSettings()
        settings.memory_fraction = 0.8
        return skalaxc.RuntimeEnvironment(settings)
    return skalaxc.RuntimeEnvironment()


def make_integrator(
    execution_space: skalaxc.ExecutionSpace,
    batch_mode: skalaxc.DomainBatchMode,
    runtime: skalaxc.RuntimeEnvironment | None = None,
    x_displacement: float = 0.0,
) -> skalaxc.XCIntegrator:
    molecule, basis = make_h2_sto3g(x_displacement)
    grid = skalaxc.MolGridFactory.create_default(
        molecule, grid_size=skalaxc.AtomicGridSize.FINE
    )
    if runtime is None:
        runtime = make_runtime(execution_space)
    load_balancer = skalaxc.LoadBalancerFactory(execution_space).get_instance(
        runtime, molecule, grid, basis
    )
    weights = skalaxc.MolecularWeightsFactory(execution_space).get_instance()
    weights.modify_weights(load_balancer)
    return skalaxc.XCIntegratorFactory(
        execution_space, domain_batch_mode=batch_mode
    ).get_instance(skalaxc.Functional("TPSS"), load_balancer)


@pytest.mark.parametrize(
    "batch_mode",
    [
        skalaxc.DomainBatchMode.CONSERVATIVE,
        skalaxc.DomainBatchMode.AGGRESSIVE,
    ],
)
def test_cuda_matches_host_through_python_binding(
    batch_mode: skalaxc.DomainBatchMode,
) -> None:
    scalar_density = np.asfortranarray([[0.5, 0.5], [0.5, 0.5]])
    spin_density = np.zeros((2, 2), dtype=np.float64, order="F")

    host = make_integrator(skalaxc.ExecutionSpace.HOST, batch_mode)
    device = make_integrator(skalaxc.ExecutionSpace.DEVICE, batch_mode)
    host_energy, host_scalar, host_spin = host.eval_exc_vxc(
        scalar_density, spin_density
    )
    device_energy, device_scalar, device_spin = device.eval_exc_vxc(
        scalar_density, spin_density
    )
    host_gradient = host.eval_exc_grad(scalar_density, spin_density)
    device_gradient = device.eval_exc_grad(scalar_density, spin_density)

    assert device_energy == pytest.approx(host_energy, abs=1e-10)
    assert np.linalg.norm(device_scalar - host_scalar) / device.nbf <= 1e-7
    assert np.linalg.norm(device_spin - host_spin) / device.nbf <= 1e-10
    assert np.max(np.abs(device_gradient - host_gradient)) <= 1e-6

    diagnostics = device.diagnostics()
    assert diagnostics.backend == skalaxc.ExecutionSpace.DEVICE
    assert diagnostics.device_id == 0
    assert diagnostics.device_memory_fraction == pytest.approx(0.8)
    assert diagnostics.domain_batch_mode == batch_mode
    assert diagnostics.exc_vxc_calls == 1
    assert diagnostics.exc_gradient_calls == 1
    assert (
        diagnostics.timing(skalaxc.TimingMetric.TOTAL_EXC_GRADIENT).status
        == skalaxc.TimingStatus.UNAVAILABLE
    )


@pytest.mark.skipif(not skalaxc.MPI_ENABLED, reason="SkalaXC was built without MPI")
def test_cuda_uses_mpi_world_with_an_idle_domain_rank() -> None:
    from mpi4py import MPI

    if MPI.COMM_WORLD.size != 3:
        pytest.skip("CUDA idle-domain coverage requires exactly three MPI ranks")

    settings = skalaxc.DeviceRuntimeSettings()
    settings.memory_fraction = 0.8 / MPI.COMM_WORLD.size
    host_runtime = skalaxc.RuntimeEnvironment(MPI.COMM_WORLD)
    device_runtime = skalaxc.RuntimeEnvironment(MPI.COMM_WORLD, settings)
    host = make_integrator(
        skalaxc.ExecutionSpace.HOST,
        skalaxc.DomainBatchMode.CONSERVATIVE,
        host_runtime,
    )
    device = make_integrator(
        skalaxc.ExecutionSpace.DEVICE,
        skalaxc.DomainBatchMode.CONSERVATIVE,
        device_runtime,
    )
    scalar_density = np.asfortranarray([[0.5, 0.5], [0.5, 0.5]])
    spin_density = np.zeros((2, 2), dtype=np.float64, order="F")

    host_energy, host_scalar, host_spin = host.eval_exc_vxc(
        scalar_density, spin_density
    )
    device_energy, device_scalar, device_spin = device.eval_exc_vxc(
        scalar_density, spin_density
    )
    host_gradient = host.eval_exc_grad(scalar_density, spin_density)
    device_gradient = device.eval_exc_grad(scalar_density, spin_density)

    assert device_energy == pytest.approx(host_energy, abs=1e-10)
    assert np.linalg.norm(device_scalar - host_scalar) / device.nbf <= 1e-7
    assert np.linalg.norm(device_spin - host_spin) / device.nbf <= 1e-10
    assert np.max(np.abs(device_gradient - host_gradient)) <= 1e-6
    assert MPI.COMM_WORLD.allgather(device_energy) == pytest.approx(
        [device_energy] * MPI.COMM_WORLD.size
    )

    diagnostics = device.diagnostics()
    local_atoms = MPI.COMM_WORLD.allgather(diagnostics.local_atoms)
    assert diagnostics.communicator_size == MPI.COMM_WORLD.size
    assert sum(local_atoms) == 2
    assert 0 in local_atoms


@pytest.mark.skipif(not skalaxc.MPI_ENABLED, reason="SkalaXC was built without MPI")
def test_cuda_uses_runtime_mpi_subcommunicator() -> None:
    from mpi4py import MPI

    if MPI.COMM_WORLD.size != 4:
        pytest.skip("CUDA subcommunicator coverage requires exactly four MPI ranks")

    color = MPI.COMM_WORLD.rank % 2
    communicator = MPI.COMM_WORLD.Split(color=color, key=MPI.COMM_WORLD.rank)
    host: skalaxc.XCIntegrator | None = None
    device: skalaxc.XCIntegrator | None = None
    host_runtime: skalaxc.RuntimeEnvironment | None = None
    device_runtime: skalaxc.RuntimeEnvironment | None = None
    try:
        settings = skalaxc.DeviceRuntimeSettings()
        settings.memory_fraction = 0.8 / MPI.COMM_WORLD.size
        host_runtime = skalaxc.RuntimeEnvironment(communicator)
        device_runtime = skalaxc.RuntimeEnvironment(communicator, settings)
        displacement = 0.04 * color
        host = make_integrator(
            skalaxc.ExecutionSpace.HOST,
            skalaxc.DomainBatchMode.CONSERVATIVE,
            host_runtime,
            displacement,
        )
        device = make_integrator(
            skalaxc.ExecutionSpace.DEVICE,
            skalaxc.DomainBatchMode.CONSERVATIVE,
            device_runtime,
            displacement,
        )
        assert host is not None
        assert device is not None
        scalar_density = np.asfortranarray(
            [[0.5 + 0.02 * color, 0.5], [0.5, 0.5 - 0.02 * color]]
        )
        spin_density = np.zeros((2, 2), dtype=np.float64, order="F")

        host_energy, host_scalar, host_spin = host.eval_exc_vxc(
            scalar_density, spin_density
        )
        device_energy, device_scalar, device_spin = device.eval_exc_vxc(
            scalar_density, spin_density
        )
        host_gradient = host.eval_exc_grad(scalar_density, spin_density)
        device_gradient = device.eval_exc_grad(scalar_density, spin_density)

        assert device_energy == pytest.approx(host_energy, abs=1e-10)
        assert np.linalg.norm(device_scalar - host_scalar) / device.nbf <= 1e-7
        assert np.linalg.norm(device_spin - host_spin) / device.nbf <= 1e-10
        assert np.max(np.abs(device_gradient - host_gradient)) <= 1e-6
        assert communicator.allgather(device_energy) == pytest.approx(
            [device_energy] * communicator.size
        )

        world_energies = MPI.COMM_WORLD.allgather(device_energy)
        assert world_energies[0] == pytest.approx(world_energies[2])
        assert world_energies[1] == pytest.approx(world_energies[3])
        assert world_energies[0] != pytest.approx(world_energies[1], abs=1e-8)
        assert device.diagnostics().communicator_size == communicator.size
    finally:
        del host, device, host_runtime, device_runtime
        communicator.Free()
