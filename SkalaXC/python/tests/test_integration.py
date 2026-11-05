from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import skalaxc

REFERENCE_DATA = Path(__file__).parents[2] / "tests" / "ref_data"
SKALA_1_1_REV1_SHA256 = (
    "7f3e8622e1eb520ccd88a55464c3e359ac4d7e5ccbd1fb77a26afa1e1c20a5cd"
)
SKALA_1_1_REV1_CUDA_SHA256 = (
    "f848eae769dca91741a518ae7275d10caac398ab21db649f91bc1f136872f223"
)


def test_cuda_version_compatibility_uses_major_family() -> None:
    assert skalaxc._cuda_versions_compatible("13.3.73", "13.0")
    assert skalaxc._cuda_versions_compatible("12.9", "12.1")
    assert not skalaxc._cuda_versions_compatible("13.0", "12.9")


def test_model_directory_matches_package_layout() -> None:
    if skalaxc.PYTHON_LAYOUT == "WHEEL":
        expected = Path(skalaxc.__file__).parent / "models"
    else:
        assert skalaxc.PYTHON_LAYOUT == "CONDA"
        expected = Path(sys.prefix) / "share" / "skalaxc" / "skala_models"

    assert skalaxc.MODEL_DIR == expected
    for model in ("ldax.fun", "pbe.fun", "tpss.fun", "skala-1.1.fun"):
        assert (skalaxc.MODEL_DIR / model).is_file()

    skala_model = skalaxc.MODEL_DIR / "skala-1.1.fun"
    assert hashlib.sha256(skala_model.read_bytes()).hexdigest() == SKALA_1_1_REV1_SHA256
    if skalaxc.CUDA_ENABLED:
        cuda_model = skalaxc.MODEL_DIR / "skala-1.1-cuda.fun"
        assert cuda_model.is_file()
        assert (
            hashlib.sha256(cuda_model.read_bytes()).hexdigest()
            == SKALA_1_1_REV1_CUDA_SHA256
        )


def build_integrator(
    fixture: Path,
    model: str = "LDA",
    runtime: skalaxc.RuntimeEnvironment | None = None,
) -> skalaxc.XCIntegrator:
    molecule = skalaxc.Molecule.from_hdf5(str(fixture))
    basis = skalaxc.BasisSet.from_hdf5(str(fixture))
    grid = skalaxc.MolGridFactory.create_default(molecule)
    if runtime is None:
        if skalaxc.MPI_ENABLED:
            from mpi4py import MPI

            runtime = skalaxc.RuntimeEnvironment(MPI.COMM_SELF)
        else:
            runtime = skalaxc.RuntimeEnvironment()
    load_balancer = skalaxc.LoadBalancerFactory(
        skalaxc.ExecutionSpace.HOST
    ).get_instance(runtime, molecule, grid, basis)
    weights = skalaxc.MolecularWeightsFactory(
        skalaxc.ExecutionSpace.HOST
    ).get_instance()
    weights.modify_weights(load_balancer)
    return skalaxc.XCIntegratorFactory(skalaxc.ExecutionSpace.HOST).get_instance(
        skalaxc.Functional(model), load_balancer
    )


@pytest.mark.parametrize(
    ("filename", "model"),
    [
        ("skala_he_def2qzvp_lda_uks.hdf5", "LDA"),
        ("skala_he_def2qzvp_pbe_uks.hdf5", "PBE"),
        ("skala_he_def2qzvp_tpss_uks.hdf5", "TPSS"),
    ],
)
def test_exc_vxc_matches_native_reference(filename: str, model: str) -> None:
    fixture = REFERENCE_DATA / filename
    integrator = build_integrator(fixture, model)
    with h5py.File(fixture) as reference:
        scalar_density = np.asfortranarray(reference["/DENSITY_SCALAR"])
        spin_density = np.asfortranarray(reference["/DENSITY_Z"])
        scalar_reference = np.asarray(reference["/VXC_SCALAR"])
        spin_reference = np.asarray(reference["/VXC_Z"])
        energy_reference = float(np.asarray(reference["/EXC"]).reshape(-1)[0])

    energy, scalar_potential, spin_potential = integrator.eval_exc_vxc(
        scalar_density, spin_density
    )

    assert energy == pytest.approx(energy_reference, rel=1e-5)
    assert np.linalg.norm(scalar_potential - scalar_reference) / integrator.nbf < 1e-7
    assert np.linalg.norm(spin_potential - spin_reference) / integrator.nbf < 1e-10
    assert scalar_potential.shape == (integrator.nbf, integrator.nbf)
    assert spin_potential.shape == (integrator.nbf, integrator.nbf)
    assert scalar_potential.dtype == np.float64
    assert spin_potential.dtype == np.float64
    assert scalar_potential.flags.f_contiguous
    assert spin_potential.flags.f_contiguous
    assert integrator.diagnostics().exc_vxc_calls == 1


def test_packaged_skala_1_1_model_evaluates() -> None:
    fixture = REFERENCE_DATA / "skala_he_def2qzvp_tpss_uks.hdf5"
    model = skalaxc.MODEL_DIR / "skala-1.1.fun"
    integrator = build_integrator(fixture, str(model))
    with h5py.File(fixture) as reference:
        scalar_density = np.asfortranarray(reference["/DENSITY_SCALAR"])
        spin_density = np.asfortranarray(reference["/DENSITY_Z"])

    energy, scalar_potential, spin_potential = integrator.eval_exc_vxc(
        scalar_density, spin_density
    )

    assert np.isfinite(energy)
    assert np.isfinite(scalar_potential).all()
    assert np.isfinite(spin_potential).all()
    assert scalar_potential.shape == (integrator.nbf, integrator.nbf)
    assert spin_potential.shape == (integrator.nbf, integrator.nbf)


def test_density_validation_rejects_wrong_shape() -> None:
    fixture = REFERENCE_DATA / "skala_he_def2qzvp_lda_uks.hdf5"
    integrator = build_integrator(fixture)
    wrong_shape = np.zeros((integrator.nbf, integrator.nbf - 1), order="F")
    density = np.zeros((integrator.nbf, integrator.nbf), order="F")

    with pytest.raises(ValueError, match="shape"):
        integrator.eval_exc_vxc(wrong_shape, density)


def test_density_inputs_are_coerced_to_float64_fortran_order() -> None:
    fixture = REFERENCE_DATA / "skala_he_def2qzvp_lda_uks.hdf5"
    integrator = build_integrator(fixture)
    with h5py.File(fixture) as reference:
        scalar_values = np.array(
            reference["/DENSITY_SCALAR"], dtype=np.float32, order="C"
        )
        spin_values = np.array(reference["/DENSITY_Z"], dtype=np.float32, order="C")

    scalar_strided_storage = np.empty(
        (2 * integrator.nbf, 2 * integrator.nbf), dtype=np.float32
    )
    spin_strided_storage = np.empty_like(scalar_strided_storage)
    scalar_strided = scalar_strided_storage[::2, ::2]
    spin_strided = spin_strided_storage[::2, ::2]
    scalar_strided[...] = scalar_values
    spin_strided[...] = spin_values

    scalar_read_only = scalar_values.copy()
    spin_read_only = spin_values.copy()
    scalar_read_only.flags.writeable = False
    spin_read_only.flags.writeable = False

    assert scalar_values.flags.c_contiguous
    assert not scalar_values.flags.f_contiguous
    assert not scalar_strided.flags.c_contiguous
    assert not scalar_strided.flags.f_contiguous
    assert not scalar_read_only.flags.writeable

    inputs = [
        (scalar_values, spin_values),
        (scalar_strided, spin_strided),
        (scalar_read_only, spin_read_only),
    ]
    for scalar_density, spin_density in inputs:
        expected = integrator.eval_exc_vxc(
            np.asfortranarray(scalar_density, dtype=np.float64),
            np.asfortranarray(spin_density, dtype=np.float64),
        )
        actual = integrator.eval_exc_vxc(scalar_density, spin_density)

        assert actual[0] == expected[0]
        np.testing.assert_allclose(actual[1], expected[1], rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(actual[2], expected[2], rtol=0.0, atol=1e-14)
        assert actual[1].dtype == np.float64
        assert actual[2].dtype == np.float64
        assert actual[1].flags.f_contiguous
        assert actual[2].flags.f_contiguous


def test_gradient_is_atom_major_and_tracks_diagnostics() -> None:
    fixture = (
        Path(__file__).parents[2]
        / "external"
        / "GauXC"
        / "tests"
        / "ref_data"
        / "h2o2_def2-tzvp.hdf5"
    )
    integrator = build_integrator(fixture, "TPSS")
    with h5py.File(fixture) as reference:
        scalar_density = np.asfortranarray(reference["/DENSITY"])
    spin_density = np.zeros_like(scalar_density, order="F")

    gradient = integrator.eval_exc_grad(scalar_density, spin_density)

    assert gradient.shape == (integrator.natoms, 3)
    assert gradient.dtype == np.float64
    assert gradient.flags.c_contiguous
    assert np.isfinite(gradient).all()
    assert np.linalg.norm(gradient) > 1e-3
    assert np.abs(gradient.sum(axis=0)).max() < 1e-10
    diagnostics = integrator.diagnostics()
    assert diagnostics.exc_gradient_calls == 1
    assert (
        diagnostics.timing(skalaxc.TimingMetric.TOTAL_EXC_GRADIENT).status
        == skalaxc.TimingStatus.COMPLETE
    )
