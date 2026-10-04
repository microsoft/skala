# SPDX-License-Identifier: MIT

from typing import Any, Protocol, cast

import numpy as np
from ase.atoms import Atoms
from ase.calculators.calculator import (
    Calculator,
    CalculatorError,
    InputError,
    all_changes,
)
from ase.units import Bohr, Debye, Hartree
from typing_extensions import override

import skala.pyscf as skala_cpu
from pyscf import grad, gto
from skala.functional.base import ExcFunctionalBase
from skala.pyscf.retry import retry_scf

try:
    import skala.gpu4pyscf as skala_gpu
except ImportError as e:
    skala_gpu = None  # type: ignore[assignment]
    gpu4pyscf_import_error = e


class _SkalaParameters(Protocol):
    xc: object
    basis: object
    with_density_fit: object
    auxbasis: str | None
    with_newton: object
    with_dftd3: object
    with_retry: object
    charge: Any
    multiplicity: Any
    verbose: Any
    ks_config: dict[str, Any] | None
    device: object


class Skala(Calculator):
    """
    ASE calculator for the Skala exchange-correlation functional.

    This calculator integrates the Skala functional into ASE, allowing
    for efficient density functional theory calculations using the Skala
    neural network-based exchange-correlation functional.
    """

    atoms: Atoms | None = None
    """Atoms object associated with the calculator."""

    implemented_properties = [
        "energy",
        "forces",
        "dipole",
    ]

    default_parameters: dict[str, Any] = {
        "xc": "skala-1.1",
        "basis": None,
        "with_density_fit": False,
        "auxbasis": None,
        "with_newton": False,
        "with_dftd3": True,
        "with_retry": True,
        "charge": None,
        "multiplicity": None,
        "verbose": 0,
        "ks_config": None,
        "device": "cpu",
    }

    _mol: gto.Mole | None = None
    _ks: grad.rhf.GradientsBase | None = None

    def __init__(self, atoms: Atoms | None = None, **kwargs: Any) -> None:
        super().__init__(atoms=atoms, **kwargs)  # type: ignore[no-untyped-call]

    @override
    def set(self, **kwargs: Any) -> dict[str, Any]:
        """
        Set parameters for the Skala calculator.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to set for the calculator.
        """
        changed_parameters: dict[str, Any] = super().set(**kwargs)  # type: ignore[no-untyped-call]
        parameters = cast(_SkalaParameters, self.parameters)
        if "verbose" in changed_parameters:
            if self._mol is not None:
                self._mol.verbose = int(parameters.verbose)
            if self._ks is not None:
                verbose = int(parameters.verbose)
                self._ks.verbose = verbose
                self._ks.base.verbose = verbose

        if "ks_config" in changed_parameters and parameters.ks_config is not None:
            if self._ks is not None:
                self._ks.base(**parameters.ks_config)

        if (
            "charge" in changed_parameters
            or "multiplicity" in changed_parameters
            or "basis" in changed_parameters
        ):
            self._mol = None
            self._ks = None
            self.reset()

        if (
            "xc" in changed_parameters
            or "with_density_fit" in changed_parameters
            or "auxbasis" in changed_parameters
            or "with_newton" in changed_parameters
            or "with_dftd3" in changed_parameters
            or "device" in changed_parameters
        ):
            self._ks = None
            self.reset()

        return changed_parameters

    @override
    def reset(self) -> None:
        """
        Reset the calculator to its initial state.
        """
        super().reset()  # type: ignore[no-untyped-call]

    @override
    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] | None = None,
    ) -> None:
        """
        Perform the calculation for the given atoms.

        Parameters
        ----------
        atoms : Atoms, optional
            The atoms object to calculate properties for.
        properties : list of str, optional
            List of properties to calculate.
        system_changes : list of str, optional
            List of changes in the system that trigger recalculation.
        """
        if not properties:
            properties = ["energy"]
        if system_changes is None:
            system_changes = all_changes

        super().calculate(  # type: ignore[no-untyped-call]
            atoms=atoms, properties=properties, system_changes=system_changes
        )
        parameters = cast(_SkalaParameters, self.parameters)

        if not isinstance(basis := parameters.basis, str):
            raise InputError("Basis set must be specified in the parameters.")

        if self.atoms is None:
            raise CalculatorError("Atoms object is required for calculation.")

        if self.atoms.pbc.any():
            raise CalculatorError(
                "Skala functional does not support periodic boundary conditions (PBC) yet."
            )

        atom = [(atom.symbol, atom.position) for atom in self.atoms]
        if set(system_changes) - {"positions"}:
            self._mol = None
            self._ks = None

        if self._mol is None:
            self._mol = gto.M(
                atom=atom,
                basis=basis,
                unit="Angstrom",
                verbose=int(parameters.verbose),
                charge=_get_charge(self.atoms, parameters),
                spin=_get_uhf(self.atoms, parameters),
            )
            self._ks = None
        else:
            self._mol = self._mol.set_geom_(atom, inplace=False)

        if self._ks is None:
            dm0 = None
            if not isinstance(xc_param := parameters.xc, (ExcFunctionalBase, str)):
                raise InputError("XC functional must be a string or ExcFunctionalBase.")
            device = parameters.device
            if device == "cuda":
                if skala_gpu is None:
                    raise ImportError(
                        "gpu4pyscf is not available. Please install gpu4pyscf to use GPU acceleration."
                    ) from gpu4pyscf_import_error
                ks = skala_gpu.SkalaKS(
                    mol=self._mol,
                    xc=xc_param,
                    with_density_fit=bool(parameters.with_density_fit),
                    auxbasis=parameters.auxbasis,
                    with_newton=bool(parameters.with_newton),
                    with_dftd3=bool(parameters.with_dftd3),
                    ks_config=parameters.ks_config,
                )
            elif device == "cpu":
                ks = skala_cpu.SkalaKS(
                    mol=self._mol,
                    xc=xc_param,
                    with_density_fit=bool(parameters.with_density_fit),
                    auxbasis=parameters.auxbasis,
                    with_newton=bool(parameters.with_newton),
                    with_dftd3=bool(parameters.with_dftd3),
                    ks_config=parameters.ks_config,
                )
            else:
                raise InputError(f"Unsupported device type: {device}")

            self._ks = ks.nuc_grad_method()
        else:
            # Mimic PySCF's SCF_Scanner (hf.py:1569-1588): convert old MOs
            # into a density-matrix guess, wipe mo_coeff so Newton won't
            # reuse stale (non-orthogonal w.r.t. new overlap) orbitals,
            # and let the solver re-diagonalise the Fock matrix.
            dm0 = None
            if self._ks.base.mo_coeff is not None:
                dm0 = self._ks.base.make_rdm1()
            self._ks.reset(self._mol)
            self._ks.base.mo_coeff = None

        if parameters.with_retry and dm0 is None:
            self._ks.base, _ = retry_scf(self._ks.base)
            energy = self._ks.base.e_tot
        else:
            energy = self._ks.base.kernel(dm0=dm0)
        gradient = self._ks.kernel()

        self.results["energy"] = float(energy) * Hartree
        dipole = self._ks.base.dip_moment(unit="debye", verbose=self._mol.verbose)
        self.results["dipole"] = np.asarray(dipole) * Debye
        self.results["forces"] = -np.asarray(gradient) * Hartree / Bohr


def _get_charge(atoms: Atoms, parameters: _SkalaParameters) -> int:
    """
    Get the total charge of the system.
    If no charge is provided, the total charge of the system is calculated
    by summing the initial charges of all atoms.
    """
    if parameters.charge is None:
        charge = atoms.get_initial_charges().sum()  # type: ignore[no-untyped-call]
    else:
        charge = parameters.charge
    return int(charge)


def _get_uhf(atoms: Atoms, parameters: _SkalaParameters) -> int:
    """
    Get the number of unpaired electrons.
    If no multiplicity is provided, the number of unpaired electrons
    is calculated by summing the initial magnetic moments of all atoms.
    """
    if parameters.multiplicity is None:
        multiplicity = int(atoms.get_initial_magnetic_moments().sum().round())  # type: ignore[no-untyped-call]
        return multiplicity
    return int(parameters.multiplicity) - 1
