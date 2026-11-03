"""Shared functional, scanner, and route-control helpers for tests."""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import ExitStack, contextmanager
from types import ModuleType
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
import torch
from skala.features import Feature, FeatureMap
from skala.functional.base import ExcFunctionalBase
from skala.pyscf import xc_integrator as xc_integrator_module
from typing_extensions import override

from pyscf import gto

if TYPE_CHECKING:
    from skala.gpu4pyscf.dft import SkalaRKS as GPUSkalaRKS
    from skala.gpu4pyscf.dft import SkalaUKS as GPUSkalaUKS
    from skala.pyscf.dft import SkalaRKS, SkalaUKS


def check_skala_scanner_reset(
    make_ks: Callable[[gto.Mole], SkalaRKS | SkalaUKS | GPUSkalaRKS | GPUSkalaUKS],
    spin: int,
    gradient: bool,
    with_dftd3: bool,
) -> None:
    """Check scanner state invalidation and agreement with fresh calculations.

    Args:
        make_ks: Factory for a configured CPU or GPU Skala calculator.
        spin: Molecular spin and charge for the hydrogen dimer.
        gradient: Whether to check a gradient rather than an energy scanner.
        with_dftd3: Whether dispersion corrections should be present.
    """
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        charge=spin,
        spin=spin,
        verbose=0,
    )
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

    for reset_mol in (mol, None):
        with ExitStack() as stack:
            reset_grids = stack.enter_context(
                patch.object(scanner_ks.grids, "reset", wraps=scanner_ks.grids.reset)
            )
            reset_d3 = (
                stack.enter_context(
                    patch.object(
                        scanner.with_dftd3, "reset", wraps=scanner.with_dftd3.reset
                    )
                )
                if with_dftd3
                else None
            )
            assert scanner.reset(reset_mol) is scanner
            reset_grids.assert_called_once_with(reset_mol)
            if reset_d3 is not None:
                reset_d3.assert_called_once_with(mol)

        assert scanner.mol is mol
        assert scanner_ks.mol is mol
        assert scanner_ks.grids.mol is mol
        assert scanner_ks.grids.coords is None
        assert scanner_ks.grids.weights is None
        assert scanner_ks.grids._spatial_grid_layout is None
        if gradient:
            assert scanner.veff_nuc_grad_ is None

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


def require_gpu() -> ModuleType:
    """Require CUDA and import CuPy, skipping when either is unavailable.

    Returns:
        The imported CuPy module.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.", allow_module_level=True)
    result = pytest.importorskip("cupy", reason="CuPy is not available.")
    assert isinstance(result, ModuleType)
    return result


class QuadraticFunctional(ExcFunctionalBase):
    """Functional whose energy is a weighted sum of squared AO-derived features."""

    def __init__(
        self,
        features: Iterable[Feature] = (
            Feature.ATOMIC_GRID_SIZES,
            Feature.DENSITY,
            Feature.GRID_WEIGHTS,
        ),
    ) -> None:
        """Initialize the functional with its required model features.

        AO-derived entries contribute quadratic energy terms. Other entries declare
        metadata needed by the evaluation route but do not contribute to the energy.

        Args:
            features: Features required from the model evaluation.

        Raises:
            ValueError: If no AO-derived feature is selected.
        """
        super().__init__()
        self.features = list(features)
        self._quadratic_features = tuple(
            feature
            for feature in self.features
            if feature
            in {
                Feature.DENSITY,
                Feature.GRAD,
                Feature.KIN,
                Feature.LAPL,
            }
        )
        if not self._quadratic_features:
            raise ValueError("At least one AO-derived feature must be selected")

    @override
    def get_exc(self, mol: FeatureMap) -> torch.Tensor:
        """Return the grid-integrated quadratic feature energy.

        The vector components of the density gradient are summed before combining
        them with scalar density, kinetic, or Laplacian terms.

        Args:
            mol: Model features keyed by their feature identifiers.

        Returns:
            Scalar exchange-correlation energy.
        """
        grid_weights = mol[Feature.GRID_WEIGHTS]
        quadratic_terms = [
            (
                mol[feature].square().sum(dim=-2)
                if feature is Feature.GRAD
                else mol[feature].square()
            )
            for feature in self._quadratic_features
        ]
        energy_density = torch.stack(quadratic_terms).sum(dim=0)
        return (energy_density * grid_weights).sum()


@contextmanager
def force_ao_screening(
    enabled: bool,
    module: ModuleType = xc_integrator_module,
) -> Generator[None]:
    """Temporarily force the AO-screening route decision.

    Args:
        enabled: Whether calls should select screened AO evaluation.
        module: Module whose ``_should_screen_aos`` decision function is patched.

    Yields:
        Control while the forced decision is active. The previous function is
        restored when the context exits.
    """
    with patch.object(
        module,
        "_should_screen_aos",
        return_value=enabled,
    ):
        yield


FULL_GRAD_REF: Mapping[str, torch.Tensor] = {
    "HF:pbe": torch.tensor(
        [[0.0, 0.0, -1.0283181338840031e-01], [0.0, 0.0, 1.0283181338840475e-01]],
        dtype=torch.float64,
    ),
    "H2O:pbe": torch.tensor(
        [
            [7.3868922411540083e-02, 0.0, 0.0],
            [-3.6934461205758495e-02, 0.0, -1.3005275018782658e-01],
            [-3.6934461205764268e-02, 0.0, 1.3005275018783147e-01],
        ],
        dtype=torch.float64,
    ),
    "H2O+:pbe": torch.tensor(
        [
            [1.3766133501961964e-01, 0.0, 0.0],
            [-6.8830667509800936e-02, 0.0, -1.6302458647600626e-01],
            [-6.8830667509806709e-02, 0.0, 1.6302458647600737e-01],
        ],
        dtype=torch.float64,
    ),
    "HF:skala-1.0": torch.tensor(
        [
            [0.0, 0.0, -0.11766455110756313],
            [0.0, 0.0, 0.11766455110756091],
        ],
        dtype=torch.float64,
    ),
    "H2O:skala-1.0": torch.tensor(
        [
            [0.04761426020567949, 0.0, 0.0],
            [-0.023807130986786884, 0.0, -0.12656276817486223],
            [-0.023807129218868184, 0.0, 0.126562766972401],
        ],
        dtype=torch.float64,
    ),
    "H2O+:skala-1.0": torch.tensor(
        [
            [0.11016447311737299, 0.0, 0.0],
            [-0.055082237334041384, 0.0, -0.15564537931499212],
            [-0.05508223578332139, 0.0, 0.15564537861887162],
        ],
        dtype=torch.float64,
    ),
    "HF:skala-1.1": torch.tensor(
        [
            [0.0, 0.0, -0.11922130029704636],
            [0.0, 0.0, 0.11922130029705125],
        ],
        dtype=torch.float64,
    ),
    "H2O:skala-1.1": torch.tensor(
        [
            [0.05518685428627901, 0.0, 0.0],
            [-0.027593427632945478, 0.0, -0.12591870031741337],
            [-0.027593426653364173, 0.0, 0.12591869974465286],
        ],
        dtype=torch.float64,
    ),
    "H2O+:skala-1.1": torch.tensor(
        [
            [0.11201511304824052, 0.0, 0.0],
            [-0.05600755684684611, 0.0, -0.15729176960843216],
            [-0.056007556201392195, 0.0, 0.15729176919222287],
        ],
        dtype=torch.float64,
    ),
}
