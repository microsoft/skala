# SPDX-License-Identifier: MIT

"""Fixtures for the GauXC export utilities."""

from collections.abc import Iterator

import pytest
from pyscf.scf import hf


@pytest.fixture(scope="session", autouse=True)
def mute_pyscf_temporary_checkpoints() -> Iterator[None]:
    """Disable implicit PySCF checkpoint files for the test session."""
    previous = hf.MUTE_CHKFILE
    hf.MUTE_CHKFILE = True
    try:
        yield
    finally:
        hf.MUTE_CHKFILE = previous
