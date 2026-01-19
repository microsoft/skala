# SPDX-License-Identifier: MIT

from typing import Any

import numpy as np
from ase.atoms import Atoms
from ase.calculators.calculator import (
    Calculator,
    CalculatorError,
    InputError,
    Parameters,
    all_changes,
)
from ase.units import Bohr, Debye, Hartree
from pyscf import grad, gto

from skala.functional.base import ExcFunctionalBase
from skala.pyscf import SkalaKS


class Skala(Calculator):  # type: ignore[misc]
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
        "xc": "skala",
        "basis": None,
        "with_density_fit": False,
        "auxbasis": None,
        "with_newton": False,
        "with_dftd3": True,
        "charge": None,
        "multiplicity": None,
        "verbose": 0,
    }

    _mol: gto.Mole | None = None
    _ks: grad.rhf.GradientsBase | None = None

    def __init__(self, atoms: Atoms | None = None, **kwargs: Any):
        super().__init__(atoms=atoms, **kwargs)

    def set(self, **kwargs: Any) -> dict[str, Any]:
        """
        Set parameters for the Skala calculator.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to set for the calculator.
        """
        changed_parameters: dict[str, Any] = super().set(**kwargs)
        if "verbose" in changed_parameters:
            if self._mol is not None:
                self._mol.verbose = int(self.parameters.verbose)
            if self._ks is not None:
                verbose = int(self.parameters.verbose)
                self._ks.verbose = verbose
                self._ks.base.verbose = verbose

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
        ):
            self._ks = None
            self.reset()

        return changed_parameters

    def reset(self) -> None:
        """
        Reset the calculator to its initial state.
        """
        super().reset()

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

        super().calculate(
            atoms=atoms, properties=properties, system_changes=system_changes
        )

        if not isinstance(basis := self.parameters.basis, str):
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
                verbose=int(self.parameters.verbose),
                charge=_get_charge(self.atoms, self.parameters),
                spin=_get_uhf(self.atoms, self.parameters),
            )
            self._ks = None
        else:
            self._mol = self._mol.set_geom_(atom, inplace=False)

        if self._ks is None:
            if not isinstance(xc_param := self.parameters.xc, (ExcFunctionalBase, str)):
                raise InputError("XC functional must be a string or ExcFunctionalBase.")
            grad_method = SkalaKS(
                self._mol,
                xc=xc_param,
                with_density_fit=bool(self.parameters.with_density_fit),
                auxbasis=self.parameters.auxbasis,
                with_newton=bool(self.parameters.with_newton),
                with_dftd3=bool(self.parameters.with_dftd3),
            ).nuc_grad_method()
            self._ks = grad_method
        else:
            self._ks.reset(self._mol)

        energy = self._ks.base.kernel()
        gradient = self._ks.kernel()

        self.results["energy"] = float(energy) * Hartree
        dipole = self._ks.base.dip_moment(unit="debye", verbose=self._mol.verbose)
        self.results["dipole"] = np.asarray(dipole) * Debye
        self.results["forces"] = -np.asarray(gradient) * Hartree / Bohr


def _get_charge(atoms: Atoms, parameters: Parameters) -> int:
    """
    Get the total charge of the system.
    If no charge is provided, the total charge of the system is calculated
    by summing the initial charges of all atoms.
    """
    if parameters.charge is None:
        charge = atoms.get_initial_charges().sum()
    else:
        charge = parameters.charge
    return int(charge)


def _get_uhf(atoms: Atoms, parameters: Parameters) -> int:
    """
    Get the number of unpaired electrons.
    If no multiplicity is provided, the number of unpaired electrons
    is calculated by summing the initial magnetic moments of all atoms.
    """
    if parameters.multiplicity is None:
        multiplicity = int(atoms.get_initial_magnetic_moments().sum().round())
        return multiplicity
    return int(parameters.multiplicity) - 1
