# SPDX-License-Identifier: MIT

"""Modification of PySCF nuclear gradient object to work with Skala functional."""

import logging
from typing import Any

import numpy as np
import torch
from dftd3.pyscf import DFTD3Dispersion
from pyscf import dft, gto
from pyscf.grad.rhf import Gradients as RHFGradient
from pyscf.grad.rks import grids_noresponse_cc, grids_response_cc
from pyscf.grad.uks import Gradients as UHFGradient
from pyscf.scf.hf import SCF

import skala.pyscf.features as feature
from skala.functional.base import ExcFunctionalBase

LOG = logging.getLogger(__name__)


def veff_and_expl_nuc_grad(
    functional: ExcFunctionalBase,
    mol: gto.Mole,
    grid: dft.Grids,
    rdm1: torch.Tensor,
    nuc_grad_feats: set[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    returns:
    - 1st tuple argument: the effective potential as requested by PySCF for nuclear gradient
    - 2nd tuple argument: explicit contributions to the nuclear gradient
    """

    SUPPORTED_FEATS = {
        "density",
        "grad",
        "kin",
        "grid_coords",
        "grid_weights",
        "atomic_grid_weights",
        "coarse_0_atomic_coords",
    }

    if nuc_grad_feats is None:  # generate feature list from functional features
        nuc_grad_feats = set(functional.features)

    # Integer-valued features have no nuclear gradient — always discard them
    nuc_grad_feats.discard("atomic_grid_sizes")
    nuc_grad_feats.discard("atomic_grid_size_bound_shape")

    # check for unsupported features
    unsupported_feats = {feat for feat in nuc_grad_feats if feat not in SUPPORTED_FEATS}
    if unsupported_feats != set():
        raise NotImplementedError(
            f"Not supported features for nuclear gradient: {unsupported_feats}"
        )

    LOG.debug("nuc_grad_feats = %s", nuc_grad_feats)

    # determine the maximum ao derivative needed
    if "grad" in nuc_grad_feats or "kin" in nuc_grad_feats:
        ao_deriv = 2
    elif "density" in nuc_grad_feats:
        ao_deriv = 1
    else:
        ao_deriv = 0

    # Get the derivatives of the weights per atom and make sure the grid is blocked per atom, no padding, etc.
    coord_list = []
    weight_list = []
    for coords, weight in grids_noresponse_cc(grid):
        coord_list.append(coords)
        weight_list.append(weight)

    grid_ = grid.copy()
    grid_.coords = np.concatenate(coord_list)
    grid_.weights = np.concatenate(weight_list)
    mol_feats = feature.generate_features(mol, rdm1, grid_, set(functional.features))

    # Discard atomic_grid_weights from VJP features: d(atomic_grid_weights)/dR = 0
    # because they are raw quadrature weights that depend only on the radial/angular
    # grid rule, not on nuclear positions. They still pass through as other_feats.
    nuc_grad_feats.discard("atomic_grid_weights")

    # Get required derivatives
    nuc_feat_names = list(nuc_grad_feats)  # ensure specific order
    nuc_feat_tensors = [mol_feats[feat] for feat in nuc_feat_names]
    other_feats = {
        feat: mol_feats[feat] for feat in mol_feats.keys() if feat not in nuc_grad_feats
    }

    def exc_feat_func(*nuc_feat_tensors: torch.Tensor) -> torch.Tensor:
        exc_mol_feats = (
            dict(zip(nuc_feat_names, nuc_feat_tensors, strict=True)) | other_feats
        )
        return functional.get_exc(exc_mol_feats)

    _, dExc_func = torch.func.vjp(exc_feat_func, *nuc_feat_tensors)
    dExc_tuple = dExc_func(torch.tensor(1.0, dtype=rdm1.dtype))
    dExc: dict[str, torch.Tensor] = {}
    for i in range(len(dExc_tuple)):
        dExc[nuc_feat_names[i]] = dExc_tuple[i].detach()

    LOG.debug("torch.func.vjp done")

    nao = rdm1.shape[-1]
    veff = torch.zeros((2, 3, nao, nao), dtype=rdm1.dtype)
    nuc_grad = torch.zeros((mol.natm, 3), dtype=rdm1.dtype)

    atm_start = 0
    for atm_id, (coords, weight, weight1) in enumerate(grids_response_cc(grid)):
        mask = dft.gen_grid.make_mask(mol, coords)
        ao = torch.from_numpy(
            dft.numint.eval_ao(
                mol, coords, deriv=ao_deriv, non0tab=mask, cutoff=grid.cutoff
            )
        )
        if ao_deriv == 0:
            ao = ao[None, ...]
        atm_end = atm_start + weight.shape[0]

        # Calculate the contribution to veff for this atomic grid
        veff_atm = torch.zeros((2, 3, nao, nao), dtype=rdm1.dtype)

        if "density" in nuc_grad_feats:
            veff_atm += torch.einsum(
                "si, xip, iq -> sxpq",
                dExc["density"][:, atm_start:atm_end],
                ao[1:4],
                ao[0],
            )

        if "grad" in nuc_grad_feats:
            Exc_dgrad_atm = dExc["grad"][:, :, atm_start:atm_end]

            veff_atm += torch.einsum(
                "syi, xip, yiq -> sxpq", Exc_dgrad_atm, ao[1:4], ao[1:4]
            )
            # XX, XY, XZ = 4, 5, 6
            veff_atm[:, 0] += torch.einsum(
                "si, ip, iq -> spq", Exc_dgrad_atm[:, 0], ao[4], ao[0]
            )
            veff_atm[:, 0] += torch.einsum(
                "si, ip, iq -> spq", Exc_dgrad_atm[:, 1], ao[5], ao[0]
            )
            veff_atm[:, 0] += torch.einsum(
                "si, ip, iq -> spq", Exc_dgrad_atm[:, 2], ao[6], ao[0]
            )
            # YX, YY, YZ = 5, 7, 8
            veff_atm[:, 1] += torch.einsum(
                "si, ip, iq -> spq", Exc_dgrad_atm[:, 0], ao[5], ao[0]
            )
            veff_atm[:, 1] += torch.einsum(
                "si, ip, iq -> spq", Exc_dgrad_atm[:, 1], ao[7], ao[0]
            )
            veff_atm[:, 1] += torch.einsum(
                "si, ip, iq -> spq", Exc_dgrad_atm[:, 2], ao[8], ao[0]
            )
            # ZX, ZY, ZZ = 6, 8, 9
            veff_atm[:, 2] += torch.einsum(
                "si, ip, iq -> spq", Exc_dgrad_atm[:, 0], ao[6], ao[0]
            )
            veff_atm[:, 2] += torch.einsum(
                "si, ip, iq -> spq", Exc_dgrad_atm[:, 1], ao[8], ao[0]
            )
            veff_atm[:, 2] += torch.einsum(
                "si, ip, iq -> spq", Exc_dgrad_atm[:, 2], ao[9], ao[0]
            )

        if "kin" in nuc_grad_feats:
            Exc_dkin_atm = dExc["kin"][:, atm_start:atm_end]
            # XX, XY, XZ = 4, 5, 6
            veff_atm[:, 0] += (
                torch.einsum("si, ip, iq -> spq", Exc_dkin_atm, ao[4], ao[1]) / 2
            )
            veff_atm[:, 0] += (
                torch.einsum("si, ip, iq -> spq", Exc_dkin_atm, ao[5], ao[2]) / 2
            )
            veff_atm[:, 0] += (
                torch.einsum("si, ip, iq -> spq", Exc_dkin_atm, ao[6], ao[3]) / 2
            )
            # YX, YY, YZ = 5, 7, 8
            veff_atm[:, 1] += (
                torch.einsum("si, ip, iq -> spq", Exc_dkin_atm, ao[5], ao[1]) / 2
            )
            veff_atm[:, 1] += (
                torch.einsum("si, ip, iq -> spq", Exc_dkin_atm, ao[7], ao[2]) / 2
            )
            veff_atm[:, 1] += (
                torch.einsum("si, ip, iq -> spq", Exc_dkin_atm, ao[8], ao[3]) / 2
            )
            # ZX, ZY, ZZ = 6, 8, 9
            veff_atm[:, 2] += (
                torch.einsum("si, ip, iq -> spq", Exc_dkin_atm, ao[6], ao[1]) / 2
            )
            veff_atm[:, 2] += (
                torch.einsum("si, ip, iq -> spq", Exc_dkin_atm, ao[8], ao[2]) / 2
            )
            veff_atm[:, 2] += (
                torch.einsum("si, ip, iq -> spq", Exc_dkin_atm, ao[9], ao[3]) / 2
            )

        if "grid_coords" in nuc_grad_feats:
            # also add the explicit grid coordinate dependence
            nuc_grad[atm_id] += dExc["grid_coords"][atm_start:atm_end].sum(dim=0)

        if "grid_weights" in nuc_grad_feats:
            Exc_dgw = dExc["grid_weights"][atm_start:atm_end]
            nuc_grad += torch.from_numpy(weight1) @ Exc_dgw
            # add the grid coordinate dependence via the density-like quantities to the nuclear gradient
            # we get those from the veff block. This tends to largely cancel with the grid_weights derivative,
            # so that's why we include it here.
            if len(rdm1.shape) == 2:
                nuc_grad[atm_id] += torch.einsum("sxpq,qp->x", veff_atm, rdm1)
            else:
                nuc_grad[atm_id] += torch.einsum("sxpq,sqp->x", veff_atm, rdm1) * 2

        veff += veff_atm
        atm_start = atm_end

    if "coarse_0_atomic_coords" in nuc_grad_feats:
        nuc_grad += dExc["coarse_0_atomic_coords"]

    # finalize
    if len(rdm1.shape) == 2:
        veff = veff.sum(0) / 2

    LOG.debug("veff and explicit components for nuclear gradient calculated")

    return (-veff, nuc_grad)


class SkalaRKSGradient(RHFGradient):  # type: ignore[misc]
    functional: ExcFunctionalBase
    """LivDFT functional"""
    nuc_grad_feats: set[str] | None
    """Which partial derivatives to take into account. None defaults to all."""
    veff_nuc_grad_: torch.Tensor
    """Contribution of the coordinate dependence of density, grad, kin, etc."""
    with_dftd3: DFTD3Dispersion | None = None
    """DFTD3 dispersion correction"""

    def __init__(
        self,
        ks: SCF,
        verbose: bool = False,
        nuc_grad_feats: set[str] | None = None,
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
        return veff.detach_().numpy() + self.get_j(mol, dm)

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

        return grad + (self.veff_nuc_grad_).numpy()

    def grad_nuc(
        self, mol: gto.Mole | None = None, atmlst: list[int] | None = None
    ) -> np.ndarray:
        nuc_g = super().grad_nuc(mol, atmlst)
        if self.with_dftd3 is None:
            return nuc_g
        disp_g = self.with_dftd3.kernel()[1]
        if atmlst is not None:
            disp_g = disp_g[atmlst]
        nuc_g += disp_g
        return nuc_g

    def extra_force(self, atom_id: int, envs: dict[str, Any]) -> int:
        return 0


class SkalaUKSGradient(UHFGradient):  # type: ignore[misc]
    functional: ExcFunctionalBase
    """LivDFT functional"""
    nuc_grad_feats: set[str] | None
    """Which partial derivatives to take into account. None defaults to all."""
    veff_nuc_grad_: torch.Tensor
    """Contribution of the coordinate dependence of density, grad, kin, etc."""
    with_dftd3: DFTD3Dispersion | None = None
    """DFTD3 dispersion correction"""

    def __init__(
        self,
        ks: SCF,
        verbose: bool = False,
        nuc_grad_feats: set[str] | None = None,
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
        return veff.detach_().numpy() + self.get_j(mol, dm).sum(0)

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

        return grad + (self.veff_nuc_grad_).numpy()

    def grad_nuc(
        self, mol: gto.Mole | None = None, atmlst: list[int] | None = None
    ) -> np.ndarray:
        nuc_g = super().grad_nuc(mol, atmlst)
        if self.with_dftd3 is None:
            return nuc_g
        disp_g = self.with_dftd3.kernel()[1]
        if atmlst is not None:
            disp_g = disp_g[atmlst]
        nuc_g += disp_g
        return nuc_g
