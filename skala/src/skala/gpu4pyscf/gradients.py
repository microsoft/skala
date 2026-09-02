# SPDX-License-Identifier: MIT

"""Modification of PySCF nuclear gradient object to work with Skala functional."""

from collections.abc import Iterator
from typing import Any

import cupy as cp
import numpy as np
import torch
from gpu4pyscf import dft
from gpu4pyscf.grad.rhf import Gradients as RHFGradient
from gpu4pyscf.grad.rks import grids_noresponse_cc, grids_response_cc
from gpu4pyscf.grad.uhf import Gradients as UHFGradient
from gpu4pyscf.scf.hf import SCF
from torch.utils.dlpack import from_dlpack

from pyscf import gto
from skala.dispersion import DFTD3Dispersion
from skala.features import Feature
from skala.functional.base import ExcFunctionalBase
from skala.pyscf.gradient_core import (
    assemble_nuclear_gradient,
    evaluate_nuclear_feature_derivatives,
)


def _veff_and_expl_nuc_grad(
    functional: ExcFunctionalBase,
    mol: gto.Mole,
    grid: dft.Grids,
    rdm1: torch.Tensor,
    nuc_grad_feats: set[Feature] | None = None,
    *,
    max_memory_in_mb: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    returns:
    - 1st tuple argument: the effective potential per atom (not the matrix!)
    - 2nd tuple argument: explicit contributions to the nuclear gradient
    """

    coord_list = []
    weight_list = []
    for coords, weight in grids_noresponse_cc(grid):
        coord_list.append(coords)
        weight_list.append(weight)

    grid_ = grid.copy()
    grid_.coords = cp.concatenate(coord_list)
    grid_.weights = cp.concatenate(weight_list)
    ao_deriv, derivatives = evaluate_nuclear_feature_derivatives(
        functional,
        mol,
        grid_,
        rdm1,
        nuc_grad_feats,
        max_memory_in_mb=max_memory_in_mb,
    )

    def atom_grid_blocks() -> Iterator[tuple[torch.Tensor, int, torch.Tensor]]:
        for coords, weight, weight1 in grids_response_cc(grid):
            mask = dft.gen_grid.make_mask(mol, coords)
            ao = from_dlpack(
                dft.numint.eval_ao(
                    mol,
                    coords,
                    deriv=ao_deriv,
                    non0tab=mask,
                )
            )
            if ao_deriv == 0:
                ao = ao[None, ...]
            yield ao, weight.shape[0], from_dlpack(weight1)

    return assemble_nuclear_gradient(derivatives, rdm1, mol.natm, atom_grid_blocks())


def nuc_grad_from_veff(
    mol: gto.Mole, veff: torch.Tensor, rdm1: torch.Tensor
) -> torch.Tensor:
    grad = torch.empty((mol.natm, 3), dtype=veff.dtype, device=veff.device)
    aoslices = mol.aoslice_by_atom()
    for iatm in range(mol.natm):
        _, _, p0, p1 = aoslices[iatm]
        grad[iatm] = torch.einsum(
            "...xij,...ij->x", veff[..., p0:p1, :], rdm1[..., p0:p1, :]
        )
    return grad


class SkalaRKSGradient(RHFGradient):  # type: ignore[misc]
    functional: ExcFunctionalBase
    """Skala functional"""
    nuc_grad_feats: set[Feature] | None
    """Which partial derivatives to take into account. None defaults to all."""
    veff_nuc_grad_: torch.Tensor | None
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
        self.veff_nuc_grad_ = None

    def energy_ee(
        self,
        mol: gto.Mole | None = None,
        dm: cp.ndarray | None = None,
    ) -> np.ndarray:
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.base.make_rdm1()

        veff, self.veff_nuc_grad_ = _veff_and_expl_nuc_grad(
            self.functional,
            mol=mol,
            grid=self.grids,
            rdm1=from_dlpack(dm),
            nuc_grad_feats=self.nuc_grad_feats,
            max_memory_in_mb=int(self.base.max_memory),
        )
        veff_grad = (
            2 * nuc_grad_from_veff(mol, veff, from_dlpack(dm)).detach().cpu().numpy()
        )
        result = veff_grad + self.jk_energy_per_atom(
            dm, k_factor=0.0, verbose=self.verbose
        )
        assert isinstance(result, np.ndarray)
        return result

    def grad_elec(
        self,
        mo_energy: cp.ndarray | None = None,
        mo_coeff: cp.ndarray | None = None,
        mo_occ: cp.ndarray | None = None,
        atmlst: list[int] | None = None,
    ) -> np.ndarray:
        if mo_energy is None:
            mo_energy = self.base.mo_energy
        if mo_occ is None:
            mo_occ = self.base.mo_occ
        if mo_coeff is None:
            mo_coeff = self.base.mo_coeff

        if self.veff_nuc_grad_ is None:
            dm = self.base.make_rdm1()
            _, self.veff_nuc_grad_ = _veff_and_expl_nuc_grad(
                self.functional,
                mol=self.mol,
                grid=self.grids,
                rdm1=from_dlpack(dm),
                nuc_grad_feats=self.nuc_grad_feats,
                max_memory_in_mb=int(self.base.max_memory),
            )
        veff_nuc_grad = self.veff_nuc_grad_
        if veff_nuc_grad is None:
            raise RuntimeError("Nuclear gradient contribution was not computed")

        grad = super().grad_elec(mo_energy, mo_coeff, mo_occ, atmlst)
        result = grad + veff_nuc_grad.detach().cpu().numpy()
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

    def reset(self, mol: gto.Mole | None = None) -> "SkalaRKSGradient":
        super().reset(mol)
        self.veff_nuc_grad_ = None
        return self


class SkalaUKSGradient(UHFGradient):  # type: ignore[misc]
    functional: ExcFunctionalBase
    """Skala functional"""
    nuc_grad_feats: set[Feature] | None
    """Which partial derivatives to take into account. None defaults to all."""
    veff_nuc_grad_: torch.Tensor | None
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
        self.veff_nuc_grad_ = None

    def energy_ee(
        self,
        mol: gto.Mole | None = None,
        dm: cp.ndarray | None = None,
    ) -> np.ndarray:
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.base.make_rdm1()

        veff, self.veff_nuc_grad_ = _veff_and_expl_nuc_grad(
            self.functional,
            mol=mol,
            grid=self.grids,
            rdm1=from_dlpack(dm),
            nuc_grad_feats=self.nuc_grad_feats,
            max_memory_in_mb=int(self.base.max_memory),
        )
        veff_grad = (
            2 * nuc_grad_from_veff(mol, veff, from_dlpack(dm)).detach().cpu().numpy()
        )
        result = veff_grad + self.jk_energy_per_atom(
            dm, k_factor=0, verbose=self.verbose
        )
        assert isinstance(result, np.ndarray)
        return result

    def grad_elec(
        self,
        mo_energy: cp.ndarray | None = None,
        mo_coeff: cp.ndarray | None = None,
        mo_occ: cp.ndarray | None = None,
        atmlst: list[int] | None = None,
    ) -> np.ndarray:
        if mo_energy is None:
            mo_energy = self.base.mo_energy
        if mo_occ is None:
            mo_occ = self.base.mo_occ
        if mo_coeff is None:
            mo_coeff = self.base.mo_coeff

        if self.veff_nuc_grad_ is None:
            dm = self.base.make_rdm1()
            _, self.veff_nuc_grad_ = _veff_and_expl_nuc_grad(
                self.functional,
                mol=self.mol,
                grid=self.grids,
                rdm1=from_dlpack(dm),
                nuc_grad_feats=self.nuc_grad_feats,
                max_memory_in_mb=int(self.base.max_memory),
            )
        veff_nuc_grad = self.veff_nuc_grad_
        if veff_nuc_grad is None:
            raise RuntimeError("Nuclear gradient contribution was not computed")

        grad = super().grad_elec(mo_energy, mo_coeff, mo_occ, atmlst)
        result = grad + veff_nuc_grad.detach().cpu().numpy()
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

    def reset(self, mol: gto.Mole | None = None) -> "SkalaUKSGradient":
        super().reset(mol)
        self.veff_nuc_grad_ = None
        return self
