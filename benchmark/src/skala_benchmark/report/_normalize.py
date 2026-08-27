# SPDX-License-Identifier: MIT

"""Shared value normalization for benchmark reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def as_mapping(value: Any) -> dict[str, Any]:
    """Return a string-keyed mapping, or an empty mapping for other values."""
    return dict(value) if isinstance(value, Mapping) else {}


def coerce_int(value: Any) -> int | None:
    """Return an integer value when possible."""
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def string_list(value: Any) -> list[str]:
    """Normalize a scalar or sequence into a list of strings."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return [str(value)]
