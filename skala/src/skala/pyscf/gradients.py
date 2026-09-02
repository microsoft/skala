# SPDX-License-Identifier: MIT

"""Modification of PySCF nuclear gradient object to work with Skala functional."""

from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from pyscf.grad.rhf import Gradients as RHFGradient
from pyscf.grad.rks import grids_noresponse_cc, grids_response_cc
from pyscf.grad.uks import Gradients as UHFGradient
from pyscf.scf.hf import SCF

from pyscf import dft, gto
from skala.dispersion import DFTD3Dispersion
from skala.features import Feature
from skala.functional.base import ExcFunctionalBase
from skala.pyscf.gradient_core import (
    assemble_nuclear_gradient,
    evaluate_nuclear_feature_derivatives,
)


def veff_and_expl_nuc_grad(
    functional: ExcFunctionalBase,
    mol: gto.Mole,
    grid: dft.Grids,
    rdm1: torch.Tensor,
    nuc_grad_feats: set[Feature] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    returns:
    - 1st tuple argument: the effective potential as requested by PySCF for nuclear gradient
    - 2nd tuple argument: explicit contributions to the nuclear gradient
    """

    coord_list = []
    weight_list = []
    for coords, weight in grids_noresponse_cc(grid):
        coord_list.append(coords)
        weight_list.append(weight)

    grid_ = grid.copy()
    grid_.coords = np.concatenate(coord_list)
    grid_.weights = np.concatenate(weight_list)
    ao_deriv, derivatives = evaluate_nuclear_feature_derivatives(
        functional,
        mol,
        grid_,
        rdm1,
        nuc_grad_feats,
        max_memory_in_mb=2000,
    )

    def atom_grid_blocks() -> Iterator[tuple[torch.Tensor, int, torch.Tensor]]:
        for coords, weight, weight1 in grids_response_cc(grid):
            mask = dft.gen_grid.make_mask(mol, coords)
            ao = torch.from_numpy(
                dft.numint.eval_ao(
                    mol, coords, deriv=ao_deriv, non0tab=mask, cutoff=grid.cutoff
                )
            )
            if ao_deriv == 0:
                ao = ao[None, ...]
            yield ao, weight.shape[0], torch.from_numpy(weight1)

    return assemble_nuclear_gradient(derivatives, rdm1, mol.natm, atom_grid_blocks())


class SkalaRKSGradient(RHFGradient):  # type: ignore[misc]
    functional: ExcFunctionalBase
    """LivDFT functional"""
    nuc_grad_feats: set[Feature] | None
    """Which partial derivatives to take into account. None defaults to all."""
    veff_nuc_grad_: torch.Tensor
    """Contribution of the coordinate dependence of density, grad, kin, etc."""
    with_dftd3: DFTD3Dispersion | None = None
    """DFTD3 dispersion correction"""

    def __init__(
        self,
        ks: SCF,
        verbose: bool = False,
        nuc_grad_feats: set[Feature] | None = None,
    ):
        super().__init__(ks)
        self.functional = ks._numint.func
        self.grids = ks.grids
        self.nuc_grad_feats = nuc_grad_feats
        self.verbose = verbose
        self.with_dftd3 = getattr(ks, "with_dftd3", None)

    def get_veff(
        self,
        mol: gto.Mole | None = None,
        dm: np.ndarray | None = None,
    ) -> np.ndarray:
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.base.make_rdm1()

        veff, self.veff_nuc_grad_ = veff_and_expl_nuc_grad(
            self.functional,
            mol=mol,
            grid=self.grids,
            rdm1=torch.from_numpy(dm),
            nuc_grad_feats=self.nuc_grad_feats,
        )
        self.veff_nuc_grad_.detach_()
        result = veff.detach_().numpy() + self.get_j(mol, dm)
        assert isinstance(result, np.ndarray)
        return result

    def grad_elec(
        self,
        mo_energy: np.ndarray | None = None,
        mo_coeff: np.ndarray | None = None,
        mo_occ: np.ndarray | None = None,
        atmlst: list[int] | None = None,
    ) -> np.ndarray:
        if mo_energy is None:
            mo_energy = self.base.mo_energy
        if mo_occ is None:
            mo_occ = self.base.mo_occ
        if mo_coeff is None:
            mo_coeff = self.base.mo_coeff

        grad = super().grad_elec(mo_energy, mo_coeff, mo_occ, atmlst)
        result = grad + (self.veff_nuc_grad_).numpy()
        assert isinstance(result, np.ndarray)
        return result

    def grad_nuc(
        self, mol: gto.Mole | None = None, atmlst: list[int] | None = None
    ) -> np.ndarray:
        nuc_g = super().grad_nuc(mol, atmlst)
        assert isinstance(nuc_g, np.ndarray)
        if self.with_dftd3 is None:
            return nuc_g
        disp_g = self.with_dftd3.get_gradient()
        if atmlst is not None:
            disp_g = disp_g[atmlst]
        nuc_g += disp_g
        return nuc_g

    def extra_force(
        self, atom_id: int | None = None, envs: dict[str, Any] | None = None
    ) -> int:
        return 0


class SkalaUKSGradient(UHFGradient):  # type: ignore[misc]
    functional: ExcFunctionalBase
    """LivDFT functional"""
    nuc_grad_feats: set[Feature] | None
    """Which partial derivatives to take into account. None defaults to all."""
    veff_nuc_grad_: torch.Tensor
    """Contribution of the coordinate dependence of density, grad, kin, etc."""
    with_dftd3: DFTD3Dispersion | None = None
    """DFTD3 dispersion correction"""

    def __init__(
        self,
        ks: SCF,
        verbose: bool = False,
        nuc_grad_feats: set[Feature] | None = None,
    ):
        super().__init__(ks)
        self.functional = ks._numint.func
        self.grids = ks.grids
        self.nuc_grad_feats = nuc_grad_feats
        self.verbose = verbose
        self.with_dftd3 = getattr(ks, "with_dftd3", None)

    def get_veff(
        self,
        mol: gto.Mole | None = None,
        dm: np.ndarray | None = None,
    ) -> np.ndarray:
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.base.make_rdm1()

        veff, self.veff_nuc_grad_ = veff_and_expl_nuc_grad(
            self.functional,
            mol=mol,
            grid=self.grids,
            rdm1=torch.from_numpy(dm),
            nuc_grad_feats=self.nuc_grad_feats,
        )
        result = veff.detach_().numpy() + self.get_j(mol, dm).sum(0)
        assert isinstance(result, np.ndarray)
        return result

    def grad_elec(
        self,
        mo_energy: np.ndarray | None = None,
        mo_coeff: np.ndarray | None = None,
        mo_occ: np.ndarray | None = None,
        atmlst: list[int] | None = None,
    ) -> np.ndarray:
        if mo_energy is None:
            mo_energy = self.base.mo_energy
        if mo_occ is None:
            mo_occ = self.base.mo_occ
        if mo_coeff is None:
            mo_coeff = self.base.mo_coeff

        grad = super().grad_elec(mo_energy, mo_coeff, mo_occ, atmlst)
        result = grad + (self.veff_nuc_grad_).numpy()
        assert isinstance(result, np.ndarray)
        return result

    def grad_nuc(
        self, mol: gto.Mole | None = None, atmlst: list[int] | None = None
    ) -> np.ndarray:
        nuc_g = super().grad_nuc(mol, atmlst)
        assert isinstance(nuc_g, np.ndarray)
        if self.with_dftd3 is None:
            return nuc_g
        disp_g = self.with_dftd3.get_gradient()
        if atmlst is not None:
            disp_g = disp_g[atmlst]
        nuc_g += disp_g
        return nuc_g
