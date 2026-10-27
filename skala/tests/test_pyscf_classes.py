from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
from skala.functional.base import ExcFunctionalBase
from skala.pyscf import SkalaKS
from skala.pyscf.dft import SkalaRKS, SkalaUKS
from skala.pyscf.gradients import SkalaRKSGradient, SkalaUKSGradient
from skala.pyscf.grids import SkalaGrids

from pyscf import dft, gto


@pytest.fixture(params=["skala-1.0", "skala-1.1"])
def skala_xc(
    request: pytest.FixtureRequest,
    load_functional_cached: Callable[..., ExcFunctionalBase | str],
) -> ExcFunctionalBase:
    """Load the Skala functional under test."""
    func = load_functional_cached(cast(str, request.param))
    assert isinstance(func, ExcFunctionalBase)
    return func


@pytest.fixture(params=["H", "H2"])
def mol(request: pytest.FixtureRequest) -> gto.Mole:
    molecule = cast(str, request.param)
    if molecule == "H":
        return gto.M(atom="H", basis="sto-3g", spin=1)
    if molecule == "H2":
        return gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    raise ValueError(f"Unknown molecule: {molecule}")


@pytest.fixture(params=["dfj", "no df"])
def with_density_fit(request: pytest.FixtureRequest) -> bool:
    return cast(str, request.param) == "dfj"


@pytest.fixture(params=["soscf", "scf"])
def with_newton(request: pytest.FixtureRequest) -> bool:
    return cast(str, request.param) == "soscf"


@pytest.fixture(params=["d3", "no d3"])
def with_dftd3(request: pytest.FixtureRequest) -> bool:
    return cast(str, request.param) == "d3"


def test_skala_class(
    mol: gto.Mole,
    skala_xc: ExcFunctionalBase,
    with_density_fit: bool,
    with_newton: bool,
    with_dftd3: bool,
) -> None:
    """Test whether classes get correctly preserved."""
    ks = SkalaKS(
        mol,
        xc=skala_xc,
        with_density_fit=with_density_fit,
        auxbasis="def2-universal-jkfit" if with_density_fit else None,
        with_newton=with_newton,
        with_dftd3=with_dftd3,
    )
    assert ks.xc == "custom"
    assert isinstance(ks, SkalaRKS if mol.spin == 0 else SkalaUKS)
    assert ks.with_dftd3 is not None if with_dftd3 else ks.with_dftd3 is None
    assert isinstance(ks.grids, SkalaGrids)

    ks_scanner = ks.as_scanner()
    assert isinstance(ks_scanner, SkalaRKS if mol.spin == 0 else SkalaUKS)
    assert (
        ks_scanner.with_dftd3 is not None
        if with_dftd3
        else ks_scanner.with_dftd3 is None
    )

    grad = ks.nuc_grad_method()
    assert isinstance(grad, SkalaRKSGradient if mol.spin == 0 else SkalaUKSGradient)
    assert grad.with_dftd3 is not None if with_dftd3 else grad.with_dftd3 is None
    assert isinstance(grad.grids, SkalaGrids)

    grad = ks.Gradients()
    assert isinstance(grad, SkalaRKSGradient if mol.spin == 0 else SkalaUKSGradient)
    assert grad.with_dftd3 is not None if with_dftd3 else grad.with_dftd3 is None
    assert isinstance(grad.grids, SkalaGrids)

    ks = grad.base
    assert isinstance(ks, SkalaRKS if mol.spin == 0 else SkalaUKS)
    assert ks.with_dftd3 is not None if with_dftd3 else ks.with_dftd3 is None
    assert isinstance(ks.grids, SkalaGrids)


def test_skala_dftd3_results(mol: gto.Mole, skala_xc: ExcFunctionalBase) -> None:
    """Test DFT-D3 energy and gradient evaluation."""
    ks = SkalaKS(mol, xc=skala_xc, with_dftd3=True)

    assert np.isfinite(ks.energy_nuc())

    gradient = ks.nuc_grad_method().grad_nuc()
    assert gradient.shape == (mol.natm, 3)
    assert np.isfinite(gradient).all()


@pytest.mark.parametrize("spin", [0, 1], ids=["rks", "uks"])
@pytest.mark.parametrize("gradient", [False, True], ids=["energy", "gradient"])
def test_skala_scanner_reset(
    skala_xc: ExcFunctionalBase,
    with_density_fit: bool,
    with_dftd3: bool,
    spin: int,
    gradient: bool,
) -> None:
    """Scanners must discard geometry-dependent state and match fresh calculations."""
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        charge=spin,
        spin=spin,
        verbose=0,
    )

    def make_ks(molecule: gto.Mole) -> SkalaRKS | SkalaUKS:
        ks = SkalaKS(
            molecule,
            xc=skala_xc,
            with_density_fit=with_density_fit,
            auxbasis="def2-universal-jkfit" if with_density_fit else None,
            with_dftd3=with_dftd3,
        )
        assert isinstance(ks, (SkalaRKS, SkalaUKS))
        ks.grids.level = 0
        ks.conv_tol = 1e-10
        return ks

    ks = make_ks(mol)
    scanner = ks.nuc_grad_method().as_scanner() if gradient else ks.as_scanner()
    moved_mol = mol.set_geom_("H 0.1 0.2 0.3; H 0.1 0.2 1.4", inplace=False)

    for geometry in (mol, moved_mol):
        reference = make_ks(geometry)
        reference_energy = reference.kernel()
        assert reference.converged

        if gradient:
            energy, nuclear_gradient = scanner(geometry)
            reference_gradient = reference.nuc_grad_method().kernel()
            assert nuclear_gradient.shape == (geometry.natm, 3)
            np.testing.assert_allclose(
                nuclear_gradient, reference_gradient, atol=1e-7, rtol=0
            )
            scanner_ks = scanner.base
            assert scanner.veff_nuc_grad_ is not None
        else:
            energy = scanner(geometry)
            scanner_ks = scanner

        assert scanner_ks.converged
        assert scanner.mol is geometry
        assert scanner_ks.mol is geometry
        assert scanner_ks.grids.mol is geometry
        assert scanner_ks.grids.coords is not None
        # Float32 model inference limits agreement between warm and fresh SCF starts.
        assert energy == pytest.approx(reference_energy, abs=1e-7, rel=0)
        if with_dftd3:
            assert scanner.with_dftd3 is not None
            assert reference.with_dftd3 is not None
            assert scanner.with_dftd3.get_energy() == pytest.approx(
                reference.with_dftd3.get_energy(), abs=1e-14, rel=0
            )
            np.testing.assert_allclose(
                scanner.with_dftd3.get_gradient(),
                reference.with_dftd3.get_gradient(),
                atol=1e-14,
                rtol=0,
            )
        else:
            assert scanner.with_dftd3 is None

    assert scanner.reset(mol) is scanner
    assert scanner.mol is mol
    assert scanner_ks.mol is mol
    assert scanner_ks.grids.mol is mol
    assert scanner_ks.grids.coords is None
    assert scanner_ks.grids.weights is None
    assert scanner_ks.grids._spatial_grid_layout is None
    if gradient:
        assert scanner.veff_nuc_grad_ is None

    assert scanner.reset() is scanner
    assert scanner.mol is mol
    reference = make_ks(mol)
    if with_dftd3:
        assert scanner.with_dftd3 is not None
        assert reference.with_dftd3 is not None
        assert scanner.with_dftd3.get_energy() == pytest.approx(
            reference.with_dftd3.get_energy(), abs=1e-14, rel=0
        )
    if gradient:
        assert scanner.base.mol is mol
        np.testing.assert_allclose(
            scanner.grad_nuc(),
            reference.nuc_grad_method().grad_nuc(),
            atol=1e-14,
            rtol=0,
        )


def test_uks_gradient_requires_initialized_grids(skala_xc: ExcFunctionalBase) -> None:
    mol = gto.M(atom="H", basis="sto-3g", spin=1)
    gradient = SkalaKS(mol, xc=skala_xc, with_dftd3=False).nuc_grad_method()
    assert isinstance(gradient, SkalaUKSGradient)
    gradient.grids = None
    dm = np.zeros((2, mol.nao_nr(), mol.nao_nr()))

    with pytest.raises(
        RuntimeError, match="UKS grids must be initialized before computing gradients"
    ):
        gradient.get_veff(dm=dm)


def test_skala_class_with_dftd3_and_native_functional_raises() -> None:
    """Test that using DFT-D3 with a native PySCF functional raises an error."""
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    with pytest.raises(
        ValueError, match="DFT-D3 dispersion correction is not supported"
    ):
        SkalaKS(mol, xc="b3lyp", with_dftd3=True)


def test_skala_class_with_native_functional_and_no_dftd3_is_allowed() -> None:
    """Native PySCF functionals should be allowed when DFT-D3 is disabled."""
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    ks = SkalaKS(mol, xc="b3lyp", with_dftd3=False)
    assert ks.xc == "b3lyp"
    assert not isinstance(ks, (SkalaRKS, SkalaUKS))


def test_initialize_grids_rejects_non_skala_grids(
    skala_xc: ExcFunctionalBase,
) -> None:
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    ks = SkalaRKS(mol, xc=skala_xc)
    ks.grids = dft.gen_grid.Grids(mol)

    with pytest.raises(TypeError, match="SkalaRKS requires .*SkalaGrids"):
        ks.initialize_grids()


def test_skala_grids_require_unit_alignment() -> None:
    mol = gto.M(atom="H", basis="sto-3g", spin=1, verbose=0)
    grids = SkalaGrids(mol)

    assert grids.alignment == 1
    grids.alignment = 1
    with pytest.raises(ValueError, match="alignment must be 1"):
        grids.alignment = 8


def test_skala_classes_disable_density_grid_pruning(
    monkeypatch: pytest.MonkeyPatch,
    skala_xc: ExcFunctionalBase,
) -> None:
    monkeypatch.setattr(dft.rks.KohnShamDFT, "small_rho_cutoff", 1e-7)
    rks_mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    uks_mol = gto.M(atom="H", basis="sto-3g", spin=1, verbose=0)

    assert SkalaRKS(rks_mol, xc=skala_xc).small_rho_cutoff == 0
    assert SkalaUKS(uks_mol, xc=skala_xc).small_rho_cutoff == 0
