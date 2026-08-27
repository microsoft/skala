# SPDX-License-Identifier: MIT

# Based on original MACE code: https://github.com/ACEsuit/mace
# See algorithm 1 in the appendix of https://arxiv.org/pdf/2206.07697

"""Clebsch–Gordan coefficient utilities for the symmetric contraction."""

from __future__ import annotations

import torch
from e3nn import o3


def _wigner_nj(
    irreps_list: list[o3.Irreps],
    filter_irs: list[o3.Irrep] | None = None,
    normalization: str = "component",
    dtype: torch.dtype | None = None,
) -> list[tuple[o3.Irrep, torch.Tensor]]:
    ret: list[tuple[o3.Irrep, torch.Tensor]] = []

    if len(irreps_list) == 1:
        irreps = irreps_list[0]
        e = torch.eye(irreps.dim, dtype=dtype)
        i = 0
        for mul, ir in irreps:
            for _ in range(mul):
                sl = slice(i, i + ir.dim)
                ret.append((ir, e[sl]))
                i += ir.dim
        return ret

    *irreps_list_left, irreps_right = irreps_list
    for ir_left, C_left in _wigner_nj(
        irreps_list_left,
        normalization=normalization,
        filter_irs=filter_irs,
        dtype=dtype,
    ):
        i = 0
        for mul, ir in irreps_right:
            for ir_out in ir_left * ir:
                if filter_irs is not None and ir_out not in filter_irs:
                    continue

                C = o3.wigner_3j(ir_out.l, ir_left.l, ir.l, dtype=dtype)

                if normalization == "component":
                    C *= ir_out.dim**0.5
                if normalization == "norm":
                    C *= ir_left.dim**0.5 * ir.dim**0.5

                C = torch.einsum("jk,ijl->ikl", C_left.flatten(1), C)
                C = C.reshape(
                    ir_out.dim,
                    *(irreps.dim for irreps in irreps_list_left),
                    ir.dim,
                )

                for u in range(mul):
                    E = torch.zeros(
                        ir_out.dim,
                        *(irreps.dim for irreps in irreps_list_left),
                        irreps_right.dim,
                        dtype=dtype,
                    )
                    sl = slice(i + u * ir.dim, i + (u + 1) * ir.dim)
                    E[..., sl] = C

                    ret.append((ir_out, E))
            i += mul * ir.dim

    return sorted(ret, key=lambda x: x[0])


def u_matrix_real(
    irreps_in: o3.Irreps,
    irreps_out: o3.Irreps,
    correlation: int,
    normalization: str = "component",
    filter_irs: list[o3.Irrep] | None = None,
    dtype: torch.dtype | None = None,
) -> list[torch.Tensor]:
    """Compute the real U-matrix for the symmetric contraction.

    Args:
        irreps_in: Input irreps (will be repeated ``correlation`` times).
        irreps_out: Target output irreps.
        correlation: Correlation order.
        normalization: Either ``"component"`` or ``"norm"``.
        filter_irs: Optional filter on intermediate irreps.
        dtype: Tensor dtype.

    Returns:
        List of U-matrix tensors, one per output irrep.
    """
    irreps_list = [irreps_in] * correlation

    if correlation == 4:
        filter_irs = [o3.Irrep(l, (-1) ** l) for l in range(12)]

    wigner_njs = _wigner_nj(
        irreps_list,
        filter_irs=filter_irs,
        normalization=normalization,
        dtype=dtype,
    )

    current_ir = wigner_njs[0][0]

    out: list[torch.Tensor] = []
    stack = torch.tensor([])

    for irrep, base_o3 in wigner_njs:
        if irrep in irreps_out and irrep == current_ir:
            stack = torch.cat((stack, base_o3.squeeze().unsqueeze(-1)), dim=-1)
        elif irrep in irreps_out and irrep != current_ir:
            if len(stack) != 0:
                out.append(stack)
            stack = base_o3.squeeze().unsqueeze(-1)
            current_ir = irrep
        else:
            current_ir = irrep

    if len(stack) != 0:
        out.append(stack)
    return out
