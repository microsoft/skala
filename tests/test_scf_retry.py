# SPDX-License-Identifier: MIT

import pytest
from pyscf import dft, gto
from pyscf.soscf.newton_ah import _CIAH_SOSCF

from skala.pyscf.retry import retry_scf

SCF_CONFIG = {
    "conv_tol": 5e-6,
    "conv_tol_grad": 0.001,
    "max_cycle": 30,
    "damp": 0.0,
    "diis_start_cycle": 1,
    "level_shift": 0.0,
    "auxbasis": "def2-svp-jkfit",
}


@pytest.fixture
def mol() -> gto.Mole:
    return gto.M(
        atom=[
            ("P", [0.3166375, -0.096706, 0.63765773]),
            ("Cl", [-1.63988274, 1.74055197, -2.03795662]),
            ("Cl", [3.28655457, 1.58141858, 2.29179879]),
            ("Cl", [-0.67829555, -3.60509951, 1.66529955]),
        ],
        verbose=4,
        unit="Bohr",
        charge=0,
        spin=0,
        basis="def2-svp",
    )


def test_retry_newton(mol: gto.Mole) -> None:
    ks = dft.KS(mol, xc="revpbe")(**SCF_CONFIG).density_fit()
    ks, state = retry_scf(ks)

    assert state.ntries > 1, "SCF should have been retried"
    assert ks.converged, "SCF did not converge with retry mechanism"
    assert isinstance(
        ks, _CIAH_SOSCF
    ), "SCF should have used Newton's method after retries"
    assert ks.level_shift == 0, "Level shift should be zero after Newton's method retry"
