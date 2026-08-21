# SPDX-License-Identifier: MIT


from typing import TypeAlias

import numpy as np

D1: TypeAlias = tuple[int]
D2: TypeAlias = tuple[int, int]
D3: TypeAlias = tuple[int, int, int]

F64: TypeAlias = np.dtype[np.float64]
I64: TypeAlias = np.dtype[np.int64]
U8: TypeAlias = np.dtype[np.uint8]

__all__ = [
    "D1",
    "D2",
    "D3",
    "F64",
    "I64",
    "U8",
]
