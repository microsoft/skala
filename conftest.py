# SPDX-License-Identifier: MIT

"""Repository-wide pytest fixtures and collection hooks."""

import functools
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import torch
from pyscf.scf import hf

from skala.functional import ExcFunctionalBase, load_functional


@pytest.fixture(scope="session", autouse=True)
def mute_pyscf_temporary_checkpoints() -> Iterator[None]:
    """Disable implicit PySCF checkpoint files for the test session.

    PySCF opens temporary checkpoint files for SCF objects even when tests do
    not use checkpointing. Reference cycles can delay their cleanup until a
    later test, where pytest reports the resulting ``ResourceWarning`` as a
    failure. Prevent the unused files and restore the previous global setting
    after the session.
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


CUDA_AVAILABLE = torch.cuda.is_available()


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """Skip CUDA-only source doctests when CUDA is unavailable."""
    if CUDA_AVAILABLE:
        return None

    try:
        relative_path = collection_path.resolve().relative_to(config.rootpath.resolve())
    except ValueError:
        return None

    return relative_path == Path(
        "src/skala/utils/torch_allocator.py"
    ) or relative_path.is_relative_to("src/skala/gpu4pyscf")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Iterator[None]:
    """Surface asynchronous CUDA failures at the test that launched them."""
    try:
        yield
    finally:
        if CUDA_AVAILABLE and item.get_closest_marker("gpu") is not None:
            try:
                torch.cuda.synchronize()
            except RuntimeError as error:
                pytest.fail(
                    f"CUDA synchronization failed after {item.nodeid}: {error}",
                    pytrace=True,
                )
