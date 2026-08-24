from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
import torch
from pyscf import gto

from tests.utils import require_gpu

pytestmark = pytest.mark.gpu

require_gpu()

from skala.functional.base import ExcFunctionalBase  # noqa: E402
from skala.gpu4pyscf import SkalaKS  # noqa: E402
from skala.gpu4pyscf.dft import SkalaRKS, SkalaUKS  # noqa: E402
from skala.gpu4pyscf.gradients import SkalaRKSGradient, SkalaUKSGradient  # noqa: E402
from skala.gpu4pyscf.grids import SkalaGrids  # noqa: E402


@pytest.fixture(params=["skala-1.0", "skala-1.1"])
def skala_xc(
    request: pytest.FixtureRequest,
    load_functional_cached: Callable[..., ExcFunctionalBase | str],
) -> ExcFunctionalBase:
    """Load the Skala functional under test on GPU."""
    func = load_functional_cached(
        cast(str, request.param), device=torch.device("cuda:0")
    )
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
    """Test DFT-D3 energy and gradient evaluation on the GPU integration."""
    ks = SkalaKS(mol, xc=skala_xc, with_dftd3=True)

    assert np.isfinite(ks.energy_nuc())

    gradient = ks.nuc_grad_method().grad_nuc()
    assert gradient.shape == (mol.natm, 3)
    assert np.isfinite(gradient).all()


def test_skala_grids_require_unit_alignment() -> None:
    mol = gto.M(atom="H", basis="sto-3g", spin=1, verbose=0)
    grids = SkalaGrids(mol)

    assert grids.alignment == 1
    grids.alignment = 1
    with pytest.raises(ValueError, match="alignment must be 1"):
        grids.alignment = 256


def test_skala_grids_disable_all_gpu4pyscf_sorting(
    mol: gto.Mole, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_options: dict[str, bool] = {}

    def build(
        self: object,
        mol: gto.Mole | None = None,
        with_non0tab: bool = False,
        **kwargs: bool,
    ) -> object:
        build_options.update(kwargs)
        return self

    monkeypatch.setattr("gpu4pyscf.dft.gen_grid.Grids.build", build)

    grids = SkalaGrids(mol)
    grids.build(sort_grids=True, sort_grids_of_each_atom=True)

    assert build_options["sort_grids"] is False
    assert build_options["sort_grids_of_each_atom"] is False


def test_skala_classes_disable_density_grid_pruning(
    monkeypatch: pytest.MonkeyPatch,
    skala_xc: ExcFunctionalBase,
) -> None:
    from gpu4pyscf.dft import rks

    monkeypatch.setattr(rks.KohnShamDFT, "small_rho_cutoff", 1e-7)
    rks_mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    uks_mol = gto.M(atom="H", basis="sto-3g", spin=1, verbose=0)

    assert SkalaRKS(rks_mol, xc=skala_xc).small_rho_cutoff == 0
    assert SkalaUKS(uks_mol, xc=skala_xc).small_rho_cutoff == 0
