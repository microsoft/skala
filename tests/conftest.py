# SPDX-License-Identifier: MIT

"""Shared test fixtures."""

import functools
from collections.abc import Callable

import pytest

from skala.functional import ExcFunctionalBase, load_functional


@pytest.fixture(scope="session")
def load_functional_cached() -> Callable[..., ExcFunctionalBase | str]:
    """Load each functional from the Hub at most once per test session."""
    return functools.lru_cache(maxsize=None)(load_functional)
