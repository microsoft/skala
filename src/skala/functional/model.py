# SPDX-License-Identifier: MIT

import math

import torch
from e3nn import o3
from opt_einsum_fx import jitable, optimize_einsums_full
from torch import fx, nn

from skala.functional.base import ExcFunctionalBase, enhancement_density_inner_product
from skala.functional.layers import ScaledSigmoid
from skala.utils.scatter import scatter_sum

# 0.32 and 2.32 are the smallest and largest covalent radius estimates
# from Pyykko and Atsumi, Chem. Eur. J. 15, 2009, 188-197
ANGSTROM_TO_BOHR = 1.88973
MIN_COV_RAD = 0.32 * ANGSTROM_TO_BOHR
MAX_COV_RAD = 2.32 * ANGSTROM_TO_BOHR


def _prepare_features(
    mol: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.cat(
        [
            mol["density"].T,
            (mol["grad"] ** 2).sum(1).T,
            mol["kin"].T,
            (mol["grad"].sum(0) ** 2).sum(0).view(-1, 1),
        ],
        dim=1,
    )
    x = x.double()

    features = torch.log(torch.abs(x) + 1e-5)

    features_ab = features
    features_ba = features[:, [1, 0, 3, 2, 5, 4, 6]]
    return features_ab, features_ba


class SkalaFunctional(ExcFunctionalBase):
    features = [
        "density",
        "kin",
        "grad",
        "grid_coords",
        "grid_weights",
        "coarse_0_atomic_coords",
    ]

    def __init__(
        self,
        lmax: int = 3,  # max angular momentum order of the spherical harmonics
        non_local: bool = True,
        non_local_hidden_nf: int = 16,
        radius_cutoff: float = float("inf"),
    ) -> None:
        super().__init__()

        self.num_scalar_features = 7
        self.non_local = non_local
        self.lmax = lmax

        self.num_feats = 256
        self.input_model = torch.nn.Sequential(
            nn.Linear(self.num_scalar_features, self.num_feats),
            nn.SiLU(),
            nn.Linear(self.num_feats, self.num_feats),  # layer 1
            nn.SiLU(),
        )

        if self.non_local:
            self.non_local_model = NonLocalModel(
                input_nf=self.num_feats,
                hidden_nf=non_local_hidden_nf,
                lmax=self.lmax,
                radius_cutoff=radius_cutoff,
            )
            self.num_non_local_contributions = non_local_hidden_nf
        else:
            self.num_non_local_contributions = 0

        # concatenate the non-local contributions to the input layer if non-local is enabled
        self.output_model = torch.nn.Sequential(
            nn.Linear(
                self.num_feats + self.num_non_local_contributions, self.num_feats
            ),  # layer 2
            nn.SiLU(),
            nn.Linear(self.num_feats, self.num_feats),  # layer 3
            nn.SiLU(),
            nn.Linear(self.num_feats, self.num_feats),  # layer 4
            nn.SiLU(),
            nn.Linear(self.num_feats, 1),
            ScaledSigmoid(scale=2.0),
        )

        self.reset_parameters()

    def get_exc_density(self, mol: dict[str, torch.Tensor]) -> torch.Tensor:
        grid_coords = mol["grid_coords"]
        grid_weights = mol["grid_weights"]
        coarse_coords = mol["coarse_0_atomic_coords"]
        features_ab, features_ba = _prepare_features(mol)

        # Learned symmetrized features
        spin_feats = torch.cat([features_ab, features_ba], dim=0)
        spin_feats = spin_feats.to(self.dtype)
        spin_feats = self.input_model(spin_feats)
        features = torch.add(*torch.chunk(spin_feats, 2, dim=0)) / 2

        # Non-local model
        if self.non_local:
            h_grid_non_local = self.non_local_model(
                features,
                grid_coords,
                coarse_coords,
                grid_weights,
            )
            h_grid_non_local = h_grid_non_local * torch.exp(
                -mol["density"].sum(0).view(-1, 1)
            ).to(self.dtype)

            features = torch.cat([features, h_grid_non_local], dim=-1)

        enhancement_factor = self.output_model(features)
        return enhancement_density_inner_product(
            enhancement_factor=enhancement_factor, density=mol["density"]
        )

    def reset_parameters(self) -> None:
        for layer in self.input_model:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

        for layer in self.output_model:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    @property
    def dtype(self) -> torch.dtype:
        return self.input_model[0].weight.dtype


class NonLocalModel(nn.Module):  # type: ignore[misc]
    def __init__(
        self,
        input_nf: int,
        hidden_nf: int,
        lmax: int,
        radius_cutoff: float = float("inf"),
    ):
        super().__init__()

        self.input_nf = input_nf
        self.hidden_nf = hidden_nf
        self.in_irreps = o3.Irreps(f"{self.hidden_nf}x0e")
        self.out_irreps = o3.Irreps(f"{self.hidden_nf}x0e")
        self.lmax = lmax
        self.hidden_irreps = o3.Irreps(
            "+".join([f"{hidden_nf}x{i}e" for i in range(self.lmax + 1)])
        )
        self.sph_irreps = o3.Irreps.spherical_harmonics(self.lmax, p=1)
        self.edge_irreps = self.sph_irreps
        self.spherical_harmonics = o3.SphericalHarmonics(
            irreps_out=self.sph_irreps,
            normalize=False,
            normalization="norm",
        )
        self.radius_cutoff = radius_cutoff

        self.pre_down_layer = torch.nn.Sequential(
            nn.Linear(self.input_nf, self.hidden_nf),
            torch.nn.SiLU(),
        )
        torch.nn.init.xavier_uniform_(self.pre_down_layer[0].weight)
        torch.nn.init.zeros_(self.pre_down_layer[0].bias)

        self.tp_down = TensorProduct(
            self.in_irreps,
            self.edge_irreps,
            self.hidden_irreps,
        )

        self.tp_up = TensorProduct(
            self.hidden_irreps,
            self.edge_irreps,
            self.out_irreps,
        )

        self.post_up_layer = torch.nn.Sequential(
            nn.Linear(self.hidden_nf, self.hidden_nf),
            torch.nn.SiLU(),
        )
        torch.nn.init.xavier_uniform_(self.post_up_layer[0].weight)
        torch.nn.init.zeros_(self.post_up_layer[0].bias)

    def forward(
        self,
        h: torch.Tensor,  # (num_fine, feats)
        grid_coords: torch.Tensor,
        coarse_coords: torch.Tensor,
        grid_weights: torch.Tensor,
    ) -> torch.Tensor:
        h = self.pre_down_layer(h)  # (num_fine, hidden_nf)

        directions, distances = vect_cdist(grid_coords, coarse_coords)
        directions = directions.to(self.dtype)  # (num_fine, num_coarse, 3)
        distances = distances.to(self.dtype)  # (num_fine, num_coarse)
        if self.radius_cutoff != float("inf"):
            up_weight = normalization_envelope(distances, self.radius_cutoff)
        else:
            up_weight = torch.ones_like(distances)

        # Find edges within the radius cutoff.
        radius_mask = distances <= self.radius_cutoff
        edge_directions = directions[radius_mask]  # (num_edges, 3)
        edge_distances = distances[radius_mask]  # (num_edges,)
        up_weight = up_weight[radius_mask]  # (num_edges,)
        edge_indices = radius_mask.nonzero()  # (num_edges, 2)
        edge_fine_idx = edge_indices[:, 0]  # (num_edges,)
        edge_coarse_idx = edge_indices[:, 1]  # (num_edges,)

        # For each edge, form a feature vector of size (hidden_nf,)
        # based on the distance between the fine and coarse points.
        edge_dist_ft = exp_radial_func(
            edge_distances, self.hidden_nf
        )  # (num_edges, hidden_nf)
        if self.radius_cutoff != float("inf"):
            # Make the cutoff smooth using a polynomial that goes from 1 at distance 0 to 0 at the
            # cutoff distance.
            envelope = polynomial_envelope(edge_distances, self.radius_cutoff, 8)
            edge_dist_ft *= envelope.unsqueeze(-1)
        else:
            envelope = torch.ones_like(edge_distances)

        # For each edge, compute a feature vector of size (hidden_nf,)
        # based on the direction.
        edge_direction_ft = self.spherical_harmonics(
            edge_directions
        )  # (num_edges, (lmax+1)^2)

        # Process (fine -> coarse) features on each edge.
        edge_h = h[edge_fine_idx]  # (num_edges, hidden_nf)
        down = self.tp_down(
            edge_h, edge_direction_ft
        )  # (num_edges, hidden_nf * (lmax+1)^2)
        down = self._mul_repeat(
            edge_dist_ft, down, self.hidden_irreps
        )  # (num_edges, hidden_nf * (lmax+1)^2)

        # Sum data from incoming edges into each coarse point.
        h_coarse = scatter_sum(
            down.double() * grid_weights.double()[edge_fine_idx].view(-1, 1),
            edge_coarse_idx,
            dim=0,
            dim_size=coarse_coords.size(0),
        ).to(self.dtype)  # (num_coarse, hidden_nf * (lmax+1)^2)

        # Process (coarse -> fine) features on each edge.
        edge_coarse_ft = h_coarse[edge_coarse_idx]
        up = self.tp_up(edge_coarse_ft, edge_direction_ft)
        # Compute the normalization factor as the sum of envelope from each coarse point
        denom = scatter_sum(
            up_weight,
            edge_fine_idx,
            dim=0,
            dim_size=grid_coords.size(0),
        )[edge_fine_idx]
        up_weight = up_weight / (denom + 0.1)
        up = self._mul_repeat(
            edge_dist_ft * up_weight.unsqueeze(-1), up, self.out_irreps
        )

        # Broadcast coarse point information back to fine points.
        h_fine = scatter_sum(
            up,
            edge_fine_idx,
            dim=0,
            dim_size=grid_coords.size(0),
        )  # (num_fine, hidden_nf)

        # Process the fine points.
        h_fine = self.post_up_layer(h_fine)  # (num_fine, hidden_nf)

        return h_fine

    @staticmethod
    def _mul_repeat(
        mul_by: torch.Tensor, edge_attrs: torch.Tensor, irreps: o3.Irreps
    ) -> torch.Tensor:
        # `edge_attrs` is spherical tensor features
        # this function multiplies `edge_attrs` with `mul_by` channels-wise per tensor order
        # (repeating over all irreps)
        mul_by_shape = mul_by.size()[:-1]
        product = torch.cat(
            [
                # (..., v, 1)  *  (..., v, j) -> (..., (v*j))
                (
                    mul_by.unsqueeze(-1)
                    * edge_attrs[..., slices].view(*mul_by_shape, mul, ir.dim)
                ).view(*mul_by_shape, -1)
                for slices, (mul, ir) in zip(irreps.slices(), irreps, strict=True)
            ],
            dim=-1,
        )
        return product

    @property
    def dtype(self) -> torch.dtype:
        return self.pre_down_layer[0].weight.dtype


def vect_cdist(c1: torch.Tensor, c2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    direction = c1[:, None] - c2[None, :]
    dist = (direction**2 + 1e-20).sum(-1).sqrt()
    return direction / dist[:, :, None], dist


def exp_radial_func(dist: torch.Tensor, num_basis: int, dim: int = 3) -> torch.Tensor:
    """
    This ensures two standard deviations of the Gaussian kernel would reach
    the desired covalent radius value (95% of the Gaussian mass).
    """
    min_std = MIN_COV_RAD / 2
    max_std = MAX_COV_RAD / 2
    s = torch.linspace(min_std, max_std, num_basis, device=dist.device)

    temps = 2 * s**2
    x2 = dist[..., None] ** 2
    emb = (
        torch.exp(-x2 / temps) * 2 / dim * x2 / temps / (math.pi * temps) ** (0.5 * dim)
    )

    return emb


def polynomial_envelope(r: torch.Tensor, cutoff: float, p: int) -> torch.Tensor:
    """
    This smoothly maps the domain r=[0, cutoff] to the range [1, 0] using a polynomial function.
    Every r >= cutoff is mapped to 0.
    """
    # from DimeNet (https://arxiv.org/abs/2003.03123)
    assert p >= 2
    r = r / cutoff
    r = torch.clamp(r, 0, 1)
    x = r - 1
    x2 = x * x
    poly = p * (p + 1) * x2 - 2 * p * x + 2
    return torch.relu(1 - 0.5 * r.pow(p) * poly)


def normalization_envelope(r: torch.Tensor, cutoff: float) -> torch.Tensor:
    r = r / cutoff
    r = torch.clamp(r, 0, 1)
    return 1 - torch.where(r < 0.5, 2 * r**2, -2 * r**2 + 4 * r - 1)


class TensorProduct(nn.Module):  # type: ignore[misc]
    optimize_einsums = True
    script_codegen = True

    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
    ):
        super().__init__()

        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = irreps_out

        self.instr = [
            (i_1, i_2, i_out)
            for i_1, (_, ir_1) in enumerate(irreps_in1)
            for i_2, (_, ir_2) in enumerate(irreps_in2)
            for i_out, (_, ir_out) in enumerate(irreps_out)
            if ir_out in ir_1 * ir_2
        ]

        self.weight_numel = sum(
            self.irreps_in1[i_1].mul
            * self.irreps_in2[i_2].mul
            * self.irreps_out[i_out].mul
            for i_1, i_2, i_out in self.instr
        )

        for i_1, i_2, i_out in self.instr:
            self.register_parameter(
                f"weight_{i_1}_{i_2}_{i_out}",
                nn.Parameter(
                    torch.randn(
                        self.irreps_in1[i_1].mul,
                        self.irreps_in2[i_2].mul,
                        self.irreps_out[i_out].mul,
                    )
                ),
            )

        self.slices = [irreps_in1.slices(), irreps_in2.slices(), irreps_out.slices()]
        for i_1, i_2, i_out in self.instr:
            w3j = o3.wigner_3j(
                irreps_in1[i_1].ir.l, irreps_in2[i_2].ir.l, irreps_out[i_out].ir.l
            ).permute(2, 0, 1)  # ijk -> kij
            self.register_buffer(f"w3j_{i_1}_{i_2}_{i_out}", w3j)

        self.reset_parameters()
        self._sparse_tp = self.generate_sparse_tp_code()

    def generate_sparse_tp_code(self) -> fx.GraphModule:
        graphmod = _sparse_tensor_product_codegen(*self.tp_params)

        if self.optimize_einsums:
            m = 3

            weight_list = self.weight_list
            example_inputs = (
                torch.randn(m, self.irreps_in1.dim),
                torch.randn(m, self.irreps_in2.dim),
                *self.w3j_list,
                *weight_list,
            )
            graphmod = optimize_einsums_full(graphmod, example_inputs)

        if self.script_codegen:
            graphmod = torch.jit.script(jitable(graphmod))

        return graphmod

    @property
    def tp_params(
        self,
    ) -> tuple[
        list[tuple[int, int, int]],
        list[tuple[int, int, int]],
        list[tuple[int, int, int]],
        list[tuple[int, int, int]],
        list[list[tuple[int, int]]],
    ]:
        return (
            self.instr,
            convert_irreps(self.irreps_in1),
            convert_irreps(self.irreps_in2),
            convert_irreps(self.irreps_out),
            [[(ss.start, ss.stop) for ss in s] for s in self.slices],
        )

    def reset_parameters(self) -> None:
        def num_elements(ins: tuple[int, int, int]) -> int:
            # assuming uvw connectivity
            return int(self.irreps_in1[ins[0]].mul * self.irreps_in2[ins[1]].mul)

        for ins in self.instr:
            i_1, i_2, i_out = ins
            num_in = sum(num_elements(ins_) for ins_ in self.instr if ins_[2] == ins[2])
            num_out = self.irreps_out[ins[2]].mul
            x = (6 / (num_in + num_out)) ** 0.5
            getattr(self, f"weight_{i_1}_{i_2}_{i_out}").data.uniform_(-x, x)

    @property
    def weight_list(self) -> list[torch.Tensor]:
        return [
            getattr(self, f"weight_{i_1}_{i_2}_{i_out}")
            for i_1, i_2, i_out in self.instr
        ]

    @property
    def w3j_list(self) -> list[torch.Tensor]:
        return [
            getattr(self, f"w3j_{i_1}_{i_2}_{i_out}") for i_1, i_2, i_out in self.instr
        ]

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self._sparse_tp(x1, x2, *self.w3j_list, *self.weight_list)


# irreps_format: list of (mul, ir.l, ir.dim)
def convert_irreps(irreps: o3.Irreps) -> list[tuple[int, int, int]]:
    return [(mul, ir.l, ir.dim) for mul, ir in irreps]


def convert_slices(slices: list[slice]) -> list[tuple[int, int]]:
    return [(s.start, s.stop) for s in slices]


def _sparse_tensor_product_codegen(
    instr: list[tuple[int, int, int]],
    irreps_in1: list[tuple[int, int, int]],
    irreps_in2: list[tuple[int, int, int]],
    irreps_out: list[tuple[int, int, int]],
    slices: list[list[tuple[int, int]]],
) -> fx.GraphModule:
    # x1: m, (u, i)
    # x2: m, (v, j)  # v is always 1

    graph = fx.Graph()
    tracer = fx.proxy.GraphAppendingTracer(graph)
    x1 = fx.Proxy(graph.placeholder("x1", torch.Tensor), tracer=tracer)
    x2 = fx.Proxy(graph.placeholder("x2", torch.Tensor), tracer=tracer)
    m = x2.size(0)

    w3js = [
        fx.Proxy(
            graph.placeholder(f"w3j_{i_1}_{i_2}_{i_out}", torch.Tensor), tracer=tracer
        )
        for i_1, i_2, i_out in instr
    ]
    weights = [
        fx.Proxy(
            graph.placeholder(f"weight_{i_1}_{i_2}_{i_out}", torch.Tensor),
            tracer=tracer,
        )
        for i_1, i_2, i_out in instr
    ]

    outs = list()
    for (i_1, i_2, i_out), w, w3j in zip(instr, weights, w3js, strict=True):
        irrep_in1 = irreps_in1[i_1]
        irrep_in2 = irreps_in2[i_2]
        irrep_out = irreps_out[i_out]

        l1l2l3 = (irrep_in1[1], irrep_in2[1], irrep_out[1])

        x1_i = x1[..., slices[0][i_1][0] : slices[0][i_1][1]]
        x2_i = x2[..., slices[1][i_2][0] : slices[1][i_2][1]]

        if l1l2l3 == (0, 0, 0):
            outs.append(torch.einsum("mu,uvw,mv->mw", x1_i, w, x2_i))
        elif l1l2l3[0] == 0:
            outs.append(
                torch.einsum(
                    "mu,uvw,mvj->mwj", x1_i, w, x2_i.view(m, irrep_in2[0], irrep_in2[2])
                ).reshape(m, irrep_out[0] * irrep_out[2])
                / math.sqrt(irrep_out[2])
            )
        elif l1l2l3[1] == 0:
            outs.append(
                torch.einsum(
                    "mui,uvw,mv->mwi", x1_i.view(m, irrep_in1[0], irrep_in1[2]), w, x2_i
                ).reshape(m, irrep_out[0] * irrep_out[2])
                / math.sqrt(irrep_out[2])
            )
        elif l1l2l3[2] == 0:
            outs.append(
                torch.einsum(
                    "mui,uvw,mvi->mw",
                    x1_i.view(m, irrep_in1[0], irrep_in1[2]),
                    w,
                    x2_i.view(m, irrep_in2[0], irrep_in2[2]),
                )
                / math.sqrt(irrep_in1[2])
            )
        else:
            outs.append(
                torch.einsum(
                    "mui,uvw,mvj,kij->mwk",
                    x1_i.view(m, irrep_in1[0], irrep_in1[2]),
                    w,
                    x2_i.view(m, irrep_in2[0], irrep_in2[2]),
                    w3j,
                ).reshape(m, irrep_out[0] * irrep_out[2])
            )

    out = [
        sum(out for ins, out in zip(instr, outs, strict=False) if ins[2] == i_out)
        for i_out, (mul, *_) in enumerate(irreps_out)
        if mul > 0
    ]
    if len(out) > 1:
        concatenated = torch.cat(out, dim=-1)
    else:
        concatenated = out[0]

    graph.output(concatenated.node, torch.Tensor)
    graph.lint()

    graphmod = fx.GraphModule(torch.nn.Module(), graph)

    return graphmod
