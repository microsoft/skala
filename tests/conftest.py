# SPDX-License-Identifier: MIT

"""Shared test fixtures."""

import functools
import typing as ty

import pytest

from skala.functional import ExcFunctionalBase, load_functional


@pytest.fixture(scope="session")
def load_functional_cached() -> ty.Callable[..., ExcFunctionalBase | str]:
    """Load each functional from the Hub at most once per test session.

    ``load_functional`` validates the cached model file's ETag against the
    Hugging Face Hub on every call, i.e. one network round-trip per invocation.
    The parametrized suite would otherwise issue hundreds of these and could trip
    the Hub's 429 rate limit.

    Two pieces work together to fix this, each doing a distinct job:

    - ``lru_cache`` does the actual deduplication. It memoizes by argument, so
      each distinct ``(name, device)`` runs ``load_functional`` (and its ETag
      check) exactly once and later calls reuse the loaded object. A plain
      fixture cannot do this: a fixture returns a single value and takes no
      call-time arguments, but tests need many functionals (``skala-1.0``,
      ``skala-1.1``, ``pbe``, ...), some chosen dynamically. Caching on the
      name lets one loader serve them all while loading each only once.
    - ``scope="session"`` gives the cache a single instance that lives for the
      whole run, so deduplication spans every test file (a function-scoped
      fixture would rebuild the cache per test). The fixture also injects the
      loader into tests via the usual dependency-injection mechanism, keeping
      this test-only: production ``load_functional`` is unchanged and the cache
      is discarded when the session ends.
    """
    return functools.lru_cache(maxsize=None)(load_functional)
