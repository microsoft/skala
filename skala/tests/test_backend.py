# SPDX-License-Identifier: MIT

import numpy as np
import pytest
from skala.pyscf.backend import from_numpy_or_cupy


def test_from_numpy_or_cupy_shares_writable_numpy_memory() -> None:
    array = np.arange(3, dtype=np.float64)

    tensor = from_numpy_or_cupy(array)
    array[0] = -1

    assert tensor[0].item() == -1


def test_from_numpy_or_cupy_restores_read_only_numpy_flag() -> None:
    array = np.arange(3, dtype=np.float64)
    array.setflags(write=False)

    tensor = from_numpy_or_cupy(array)
    with pytest.raises(ValueError, match="read-only"):
        array[0] = -1

    tensor[0] = -1

    assert array[0] == -1
