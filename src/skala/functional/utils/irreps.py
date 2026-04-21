# SPDX-License-Identifier: MIT

"""
Compile-compatible implementation of e3nn.o3.Irreps.

This module provides a reimplementation of e3nn's Irrep, MulIr, and Irreps classes
that is fully compatible with torch.compile. The original e3nn implementation
inherits from tuple and raises NotImplementedError for __len__ on Irrep, which
causes issues with torch.compile's graph tracing.

Key differences from e3nn.o3.Irreps:
    - Uses __slots__ instead of inheriting from tuple (for Irrep and MulIr)
    - Uses an internal list instead of inheriting from tuple (for Irreps)
    - Uses __init__ instead of __new__ for construction
    - Omits Wigner D-matrix methods (D_from_angles, D_from_quaternion, etc.)
    - Omits some convenience methods (randn, filter, regroup, count, etc.)

The implemented subset is sufficient for defining irreducible representations
and their direct sums for use in equivariant neural network architectures.

Example:
    >>> from skala.functional.utils.irreps import Irrep, Irreps
    >>> Irrep("1o")
    1o
    >>> Irreps("16x0e + 8x1o + 4x2e")
    16x0e+8x1o+4x2e
    >>> Irreps.spherical_harmonics(3)
    1x0e+1x1o+1x2e+1x3o

See Also:
    e3nn.o3.Irreps: The original implementation from the e3nn library.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import NamedTuple, overload


class Irrep:
    __slots__ = ("_l", "_p")
    _l: int
    _p: int

    def __init__(self, l: int | str | tuple[int, int] | Irrep, p: int | None = None):
        if p is None:
            if isinstance(l, Irrep):
                self._l = l._l
                self._p = l._p
                return
            if isinstance(l, str):
                name = l.strip()
                self._l = int(name[:-1])
                assert self._l >= 0
                self._p = {"e": 1, "o": -1, "y": (-1) ** self._l}[name[-1]]
                return
            if isinstance(l, tuple):
                l, p = l

        if not isinstance(l, int) or l < 0:
            raise ValueError(f"l must be non-negative integer, got {l}")
        if p not in (-1, 1):
            raise ValueError(f"parity must be -1 or 1, got {p}")
        self._l = l
        self._p = p

    @property
    def l(self) -> int:  # noqa: E743
        return self._l

    @property
    def p(self) -> int:
        return self._p

    @property
    def dim(self) -> int:
        return 2 * self._l + 1

    def __repr__(self) -> str:
        p = {1: "e", -1: "o"}[self._p]
        return f"{self._l}{p}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Irrep):
            return self._l == other._l and self._p == other._p
        if isinstance(other, tuple) and len(other) == 2:
            return self._l == other[0] and self._p == other[1]
        return False

    def __hash__(self) -> int:
        return hash((self._l, self._p))

    def __mul__(self, other: Irrep) -> Iterator[Irrep]:
        other = Irrep(other)
        p = self._p * other._p
        lmin = abs(self._l - other._l)
        lmax = self._l + other._l
        for l in range(lmin, lmax + 1):
            yield Irrep(l, p)

    def __iter__(self) -> Iterator[int]:
        yield self._l
        yield self._p

    def __getitem__(self, i: int) -> int:
        if i == 0:
            return self._l
        if i == 1:
            return self._p
        raise IndexError(i)


class MulIr:
    __slots__ = ("_mul", "_ir")
    _mul: int
    _ir: Irrep

    def __init__(
        self,
        mul: int | MulIr | tuple[int, Irrep | str | tuple[int, int]],
        ir: Irrep | str | tuple[int, int] | None = None,
    ):
        if ir is None:
            if isinstance(mul, MulIr):
                self._mul = mul._mul
                self._ir = mul._ir
                return
            if isinstance(mul, tuple) and len(mul) == 2:
                mul, ir = mul
        if not isinstance(mul, int) or mul < 0:
            raise ValueError(f"mul must be non-negative integer, got {mul}")
        if ir is None:
            raise ValueError("ir must be provided")
        self._mul = mul
        self._ir = Irrep(ir) if not isinstance(ir, Irrep) else ir

    @property
    def mul(self) -> int:
        return self._mul

    @property
    def ir(self) -> Irrep:
        return self._ir

    @property
    def dim(self) -> int:
        return self._mul * self._ir.dim

    def __repr__(self) -> str:
        return f"{self._mul}x{self._ir}"

    def __iter__(self) -> Iterator[int | Irrep]:
        yield self._mul
        yield self._ir

    def __getitem__(self, i: int) -> int | Irrep:
        if i == 0:
            return self._mul
        if i == 1:
            return self._ir
        raise IndexError(i)


class Irreps:
    """Compile-compatible direct sum of irreducible representations of O(3)."""

    def __init__(
        self,
        irreps: str
        | Irreps
        | Irrep
        | Sequence[MulIr | str | Irrep | tuple[int, Irrep | str | tuple[int, int]]]
        | None = None,
    ):
        self._data: list[MulIr] = []

        if irreps is None:
            return
        if isinstance(irreps, Irreps):
            self._data = list(irreps._data)
            return
        if isinstance(irreps, Irrep):
            self._data = [MulIr(1, irreps)]
            return
        if isinstance(irreps, str):
            if irreps.strip() == "":
                return
            for mul_ir in irreps.split("+"):
                mul_ir = mul_ir.strip()
                if "x" in mul_ir:
                    mul_str, ir_str = mul_ir.split("x")
                    mul = int(mul_str)
                    ir = Irrep(ir_str)
                else:
                    mul = 1
                    ir = Irrep(mul_ir)
                self._data.append(MulIr(mul, ir))
            return
        for item in irreps:
            if isinstance(item, MulIr):
                self._data.append(item)
            elif isinstance(item, str):
                self._data.append(MulIr(1, Irrep(item)))
            elif isinstance(item, Irrep):
                self._data.append(MulIr(1, item))
            elif isinstance(item, tuple) and len(item) == 2:
                mul, ir_like = item
                self._data.append(MulIr(mul, Irrep(ir_like)))
            else:
                raise ValueError(f"Unable to interpret {item!r} as an irrep")

    @staticmethod
    def spherical_harmonics(lmax: int, p: int = -1) -> Irreps:
        return Irreps([(1, (l, p**l)) for l in range(lmax + 1)])

    def slices(self) -> list[slice]:
        s = []
        i = 0
        for mul_ir in self._data:
            s.append(slice(i, i + mul_ir.dim))
            i += mul_ir.dim
        return s

    @property
    def dim(self) -> int:
        return sum(mul_ir.dim for mul_ir in self._data)

    @property
    def num_irreps(self) -> int:
        return sum(mul_ir.mul for mul_ir in self._data)

    @property
    def ls(self) -> list[int]:
        return [mul_ir.ir.l for mul_ir in self._data for _ in range(mul_ir.mul)]

    @property
    def lmax(self) -> int:
        if len(self._data) == 0:
            raise ValueError("Cannot get lmax of empty Irreps")
        return max(mul_ir.ir.l for mul_ir in self._data)

    def simplify(self) -> Irreps:
        out: list[tuple[int, Irrep]] = []
        for mul_ir in self._data:
            mul, ir = mul_ir.mul, mul_ir.ir
            if out and out[-1][1] == ir:
                out[-1] = (out[-1][0] + mul, ir)
            elif mul > 0:
                out.append((mul, ir))
        return Irreps(out)

    class _SortResult(NamedTuple):
        irreps: Irreps
        p: tuple[int, ...]
        inv: tuple[int, ...]

    def sort(self) -> Irreps._SortResult:
        indexed = [(mul_ir.ir, i, mul_ir.mul) for i, mul_ir in enumerate(self._data)]
        indexed.sort(key=lambda x: (x[0].l, x[0].p))
        inv = tuple(i for _, i, _ in indexed)
        p_list = [0] * len(inv)
        for i, j in enumerate(inv):
            p_list[j] = i
        p = tuple(p_list)
        irreps = Irreps([(mul, ir) for ir, _, mul in indexed])
        return Irreps._SortResult(irreps, p, inv)

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[MulIr]:
        return iter(self._data)

    @overload
    def __getitem__(self, i: int) -> MulIr: ...

    @overload
    def __getitem__(self, i: slice) -> Irreps: ...

    def __getitem__(self, i: int | slice) -> MulIr | Irreps:
        if isinstance(i, slice):
            return Irreps([(m.mul, m.ir) for m in self._data[i]])
        return self._data[i]

    def __contains__(self, ir: Irrep) -> bool:
        ir = Irrep(ir)
        return any(mul_ir.ir == ir for mul_ir in self._data)

    def __add__(self, other: Irreps) -> Irreps:
        other = Irreps(other)
        return Irreps(
            [(m.mul, m.ir) for m in self._data] + [(m.mul, m.ir) for m in other._data]
        )

    def __mul__(self, n: int) -> Irreps:
        if not isinstance(n, int):
            raise NotImplementedError("Use o3.TensorProduct for irrep multiplication")
        return Irreps([(m.mul, m.ir) for m in self._data] * n)

    def __rmul__(self, n: int) -> Irreps:
        return self.__mul__(n)

    def __repr__(self) -> str:
        return "+".join(str(mul_ir) for mul_ir in self._data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Irreps):
            return False
        if len(self._data) != len(other._data):
            return False
        return all(
            a.mul == b.mul and a.ir == b.ir
            for a, b in zip(self._data, other._data, strict=True)
        )

    def __hash__(self) -> int:
        return hash(tuple((m.mul, m.ir.l, m.ir.p) for m in self._data))
