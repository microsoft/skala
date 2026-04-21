# SPDX-License-Identifier: MIT

# Based on original MACE code: https://github.com/ACEsuit/mace
# See algorithm 1 in the appendix of https://arxiv.org/pdf/2206.07697

import opt_einsum_fx
import torch
import torch.fx
from e3nn import o3

from skala.functional.utils.cg import u_matrix_real

ALPHABET = ["w", "x", "v", "n", "z", "r", "t", "y", "u", "o", "p", "s"]


def get_alphabet_string(i: int) -> str:
    if i == -1:
        return ""
    return "".join(ALPHABET[:i])


class SymmetricContraction(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        correlation: int,
        sketch: bool = False,
    ) -> None:
        super().__init__()

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.correlation = correlation
        self.sketch = sketch

        hidden_nfs = [mul for mul, _ in irreps_in]
        assert len(set(hidden_nfs)) == 1, (
            "All irreps need to have the same number of channels"
        )
        hidden_nf = hidden_nfs[0]

        self.contractions = torch.nn.ModuleList(
            [
                Contraction(
                    irreps_in=irreps_in,
                    irrep_out=o3.Irreps(str(irrep_out.ir)),
                    correlation=correlation,
                    hidden_nf=hidden_nf,
                )
                for irrep_out in irreps_out
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x_packed = pack(x, self.irreps_in)
        if self.sketch:
            xs = torch.chunk(x_packed, self.correlation, dim=1)
        else:
            xs = (x_packed,) * self.correlation
        outs = [contraction(*xs) for contraction in self.contractions]
        return torch.cat(outs, dim=-1)


class Contraction(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irrep_out: o3.Irreps,
        correlation: int,
        hidden_nf: int,
    ) -> None:
        super().__init__()

        self.correlation = correlation

        for nu in range(correlation, 0, -1):
            u_matrix = u_matrix_real(
                irreps_in=o3.Irreps("+".join(str(irrep.ir) for irrep in irreps_in)),
                irreps_out=irrep_out,
                correlation=nu,
                dtype=torch.get_default_dtype(),
            )[-1]
            self.register_buffer(f"u_matrix_{nu}", u_matrix)

        self.contractions_weighting = torch.nn.ModuleList()
        self.contractions_features = torch.nn.ModuleList()

        self.weights = torch.nn.ParameterList()

        for i in range(correlation, 0, -1):
            u_tensor = self.get_u_tensor(i)
            dim_nu = u_tensor.shape[-1]
            dim_lm = u_tensor.shape[-2]
            dim_M = 2 * irrep_out.lmax + 1
            dim_x = 11
            dim_c = hidden_nf

            if i == correlation:
                indices = get_alphabet_string(i + min(irrep_out.lmax, 1) - 1)

                def _graph_module_main(
                    C: torch.Tensor, W: torch.Tensor, x: torch.Tensor
                ) -> torch.Tensor:
                    return torch.einsum(
                        indices + "ik,kc,bci -> bc" + indices,  # noqa: B023
                        C,
                        W,
                        x,
                    )

                graph_module_main = torch.fx.symbolic_trace(_graph_module_main)

                graph_opt_main = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_main,
                    example_inputs=(
                        torch.randn([dim_M] + [dim_lm] * i + [dim_nu]).squeeze(0),
                        torch.randn((dim_nu, dim_c)),
                        torch.randn((dim_x, dim_c, dim_lm)),
                    ),
                )
                assert isinstance(graph_opt_main, torch.fx.GraphModule)
                self.graph_opt_main = graph_opt_main

                self.weights_max = torch.nn.Parameter(
                    torch.randn((dim_nu, dim_c)) / dim_nu
                )

            else:
                indices = get_alphabet_string(i + min(irrep_out.lmax, 1))

                def _graph_module_weighting(
                    C: torch.Tensor, W: torch.Tensor
                ) -> torch.Tensor:
                    return torch.einsum(
                        indices + "k,kc->c" + indices,  # noqa: B023
                        C,
                        W,
                    )

                graph_module_weighting = torch.fx.symbolic_trace(
                    _graph_module_weighting
                )
                graph_opt_weighting = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_weighting,
                    example_inputs=(
                        torch.randn([dim_M] + [dim_lm] * i + [dim_nu]).squeeze(0),
                        torch.randn((dim_nu, dim_c)),
                    ),
                )
                assert isinstance(graph_opt_weighting, torch.fx.GraphModule)
                self.contractions_weighting.append(graph_opt_weighting)

                indices = get_alphabet_string(i - 1 + min(irrep_out.lmax, 1))

                def _graph_module_features(
                    c: torch.Tensor, x: torch.Tensor
                ) -> torch.Tensor:
                    return torch.einsum(
                        "bc" + indices + "i,bci->bc" + indices,  # noqa: B023
                        c,
                        x,
                    )

                graph_module_features = torch.fx.symbolic_trace(_graph_module_features)
                graph_opt_features = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_features,
                    example_inputs=(
                        torch.randn([dim_x, dim_c, dim_M] + [dim_lm] * i).squeeze(2),
                        torch.randn((dim_x, dim_c, dim_lm)),
                    ),
                )
                assert isinstance(graph_opt_features, torch.fx.GraphModule)
                self.contractions_features.append(graph_opt_features)

                self.weights.append(
                    torch.nn.Parameter(torch.randn((dim_nu, dim_c)) / dim_nu)
                )

    def get_u_tensor(self, nu: int) -> torch.Tensor:
        return dict(self.named_buffers())[f"u_matrix_{nu}"]

    def forward(
        self,
        *xs: list[torch.Tensor],
    ) -> torch.Tensor:
        out = self.graph_opt_main(
            self.get_u_tensor(self.correlation), self.weights_max, xs[0]
        )
        for i, (x, weights, contract_weights, contract_features) in enumerate(
            zip(
                xs[1:],
                self.weights,
                self.contractions_weighting,
                self.contractions_features,
                strict=False,
            )
        ):
            c_tensor = contract_weights(
                self.get_u_tensor(self.correlation - i - 1), weights
            )
            c_tensor = c_tensor + out
            out = contract_features(c_tensor, x)

        return out.view(out.shape[0], -1)


def pack(x: torch.Tensor, irreps: o3.Irreps) -> torch.Tensor:
    return torch.cat(
        [
            x[..., slice].view(*x.shape[:-1], mul, ir.dim)
            for (mul, ir), slice in zip(irreps, irreps.slices(), strict=False)
        ],
        dim=-1,
    )
