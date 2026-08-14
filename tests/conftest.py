# SPDX-License-Identifier: MIT

"""Shared test fixtures."""

import functools
from collections.abc import Callable, Iterator

import pytest
from pyscf.scf import hf

from skala.functional import ExcFunctionalBase, load_functional


@pytest.fixture(scope="session", autouse=True)
def mute_pyscf_temporary_checkpoints() -> Iterator[None]:
    """Disable implicit PySCF checkpoint files for the test session.

    PySCF opens a ``NamedTemporaryFile`` whenever it constructs an SCF object,
    even when a test never requests checkpointing. Some transformed SCF objects
    participate in reference cycles, so their temporary files can remain open
    until cyclic garbage collection. Python then emits ``ResourceWarning``, and
    pytest 9 re-emits it as ``PytestUnraisableExceptionWarning``. Because this
    project treats warnings as errors, whichever test triggers collection can
    fail even if a different test created the file.

    Preventing the unused files through PySCF's ``MUTE_CHKFILE`` switch avoids
    the resource lifetime problem instead of filtering platform-specific warning
    messages. The previous setting is restored after the session, and production
    behavior is unchanged.
    """
    previous = hf.MUTE_CHKFILE
    hf.MUTE_CHKFILE = True
    try:
        yield
    finally:
        hf.MUTE_CHKFILE = previous


@pytest.fixture(scope="session")
def load_functional_cached() -> Callable[..., ExcFunctionalBase | str]:
    """Load each functional from the Hub at most once per test session."""
    return functools.lru_cache(maxsize=None)(load_functional)
