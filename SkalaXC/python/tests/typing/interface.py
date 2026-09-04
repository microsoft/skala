from __future__ import annotations

import numpy as np
import skalaxc
from typing_extensions import assert_type  # noqa: UP035

Matrix = np.ndarray[tuple[int, int], np.dtype[np.float64]]


def check_evaluation_types(
    integrator: skalaxc.XCIntegrator,
    density: Matrix,
) -> None:
    energy, scalar_potential, spin_potential = integrator.eval_exc_vxc(density, density)
    assert_type(energy, float)
    assert_type(scalar_potential, Matrix)
    assert_type(spin_potential, Matrix)
    assert_type(
        integrator.eval_exc_grad(density, density),
        Matrix,
    )
