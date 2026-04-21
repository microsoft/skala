# SPDX-License-Identifier: MIT

"""
Retry mechanism for SCF calculations in PySCF.

This module provides a retry mechanism for self-consistent field (SCF)
calculations in PySCF, allowing for robust convergence handling.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from pyscf.scf.hf import SCF
from pyscf.soscf.newton_ah import _CIAH_SOSCF

SMALL_GAP = 0.1


def _min_gap(gaps: list[float | None]) -> float | None:
    finite_gaps = [gap for gap in gaps[2:-1] if gap is not None]
    return min(finite_gaps) if finite_gaps else None


@dataclass
class SCFState:
    ntries: int
    cycles: list[int]
    grid_size: int | None
    e_tot_per_cycle: list[float]
    gradient_norm_per_cycle: list[float | None]
    dm_change_per_cycle: list[float | None]
    homo_lumo_gap_up_per_cycle: list[float | None]
    homo_lumo_gap_down_per_cycle: list[float | None]

    @classmethod
    def empty(cls) -> "SCFState":
        return cls(
            ntries=0,
            cycles=[],
            grid_size=None,
            e_tot_per_cycle=[],
            gradient_norm_per_cycle=[],
            dm_change_per_cycle=[],
            homo_lumo_gap_up_per_cycle=[],
            homo_lumo_gap_down_per_cycle=[],
        )

    def add_callback(self, scf: SCF) -> None:
        scf.pre_kernel = self.pre_kernel_callback
        scf.callback = self.post_cycle_callback
        scf.post_kernel = self.post_kernel_callback

    def pre_kernel_callback(self, envs: dict[str, Any]) -> None:
        scf: SCF = envs["mf"]
        e_tot = envs["e_tot"]

        self.ntries += 1
        self.cycles = [0]
        self.grid_size = scf.grids.size if hasattr(scf, "grids") else None
        self.e_tot_per_cycle = [e_tot]
        self.gradient_norm_per_cycle = [None]
        self.dm_change_per_cycle = [None]
        self.homo_lumo_gap_up_per_cycle = [None]
        self.homo_lumo_gap_down_per_cycle = [None]

    def post_cycle_callback(self, envs: dict[str, Any]) -> None:
        scf: SCF = envs["mf"]
        e_tot = envs["e_tot"]
        norm_gorb = envs["norm_gorb"]

        if "cycle" in envs:  # Default SCF
            self.cycles.append(int(envs["cycle"] + 1))
            mo_energy = envs["mo_energy"]
        elif "imacro" in envs:  # Second-order SCF
            self.cycles.append(envs["imacro"] + 1)
            # In second-order SCF the orbital energies are not available every step
            mo_energy, _ = scf._scf.canonicalize(
                envs["mo_coeff"], envs["mo_occ"], envs["fock"]
            )
        else:  # Unknown SCF implementation, hope for the best
            mo_energy = envs["mo_energy"]

        self.e_tot_per_cycle.append(e_tot)
        self.gradient_norm_per_cycle.append(norm_gorb)
        if "norm_ddm" not in envs:
            envs["norm_ddm"] = np.linalg.norm(envs["dm"] - envs["dm_last"])

        norm_ddm = envs["norm_ddm"]
        assert isinstance(norm_ddm, (float, int)), (
            f"Expected norm_ddm to be a float, got {type(norm_ddm)}"
        )
        self.dm_change_per_cycle.append(norm_ddm)

        if not isinstance(mo_energy, list) and len(mo_energy.shape) == 1:
            self.homo_lumo_gap_up_per_cycle.append(
                np.min(mo_energy[envs["mo_occ"] == 0])
                - np.max(mo_energy[envs["mo_occ"] > 0])
            )
            self.homo_lumo_gap_down_per_cycle.append(
                self.homo_lumo_gap_up_per_cycle[-1]
            )
        else:
            gap = [None, None]
            for spin in (0, 1):
                if np.any(envs["mo_occ"][spin] > 0):
                    gap[spin] = np.min(
                        mo_energy[spin][envs["mo_occ"][spin] == 0]
                    ) - np.max(mo_energy[spin][envs["mo_occ"][spin] > 0])
            self.homo_lumo_gap_up_per_cycle.append(gap[0])
            self.homo_lumo_gap_down_per_cycle.append(gap[1])

    def post_kernel_callback(self, envs: dict[str, Any]) -> None:
        scf: SCF = envs["mf"]

        if scf.conv_check:
            envs["cycle"] += 1
            scf.callback(envs)

    def get_gap(self) -> tuple[float | None, float | None]:
        return (
            _min_gap(self.homo_lumo_gap_up_per_cycle),
            _min_gap(self.homo_lumo_gap_down_per_cycle),
        )


def retry_scf(
    scf: SCF,
) -> tuple[SCF, SCFState]:
    """
    Retry the SCF calculation if it fails due to convergence issues.

    Parameters
    ----------
    scf : SCF
        The Kohn-Sham object to perform the SCF calculation on.

    Returns
    -------
    (SCF, SCFState)
         The SCF object after retrying and the state of the SCF calculation.
    """

    state = SCFState.empty()
    state.add_callback(scf)

    scf.kernel()
    if scf.converged:
        return scf, state

    # If SOSCF was used already, don't retry
    if isinstance(scf, _CIAH_SOSCF):
        return scf, state

    init_config: dict[str, float | int] = {
        "damp": scf.damp,
        "diis_start_cycle": scf.diis_start_cycle,
    }
    scf = scf.set(
        damp=0.5,
        diis_start_cycle=7,
    )

    scf.kernel()
    if scf.converged:
        return scf, state

    # Only try level shift if gap is small
    gaps = state.get_gap()
    if any(gap is not None and gap < SMALL_GAP for gap in gaps):
        while scf.level_shift != increment_level_shift(scf.level_shift):
            scf.set(**init_config, level_shift=increment_level_shift(scf.level_shift))
            scf.kernel()
            gaps = state.get_gap()
            # Ensure that gap actually opened from level shift to avoid misconverged results
            sufficient_gap = all(gap is not None and gap >= SMALL_GAP for gap in gaps)
            scf.converged = scf.converged and sufficient_gap
            if scf.converged:
                return scf, state
            if sufficient_gap:
                break

    # Reset level shift to zero
    scf.set(**init_config, level_shift=0.0)

    # Try second-order SCF with Newton's method
    scf = scf.newton()
    scf.kernel()
    if scf.converged:
        return scf, state

    # If still not converged, return the last attempt
    return scf, state


def increment_level_shift(
    level_shift: float,
    max_level_shift: float = 0.5,
    level_shift_init: float = 0.1,
    level_shift_increment: float = 0.2,
) -> float:
    return (
        min(level_shift + level_shift_increment, max_level_shift)
        if level_shift > 0
        else level_shift_init
    )
