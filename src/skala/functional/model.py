# SPDX-License-Identifier: MIT

"""
Skala neural exchange-correlation functional.

This module implements the Skala model (``SkalaFunctional``), which uses a
packed per-atom grid layout with multiple non-local equivariant message-passing
layers and symmetric contraction for higher-order body correlations.
"""

import math
from typing import Any, cast

import torch
from e3nn import o3
from opt_einsum_fx import jitable, optimize_einsums_full
from torch import fx, nn

from skala.functional.base import ExcFunctionalBase, enhancement_density_inner_product
from skala.functional.layers import ScaledSigmoid
from skala.functional.utils.irreps import Irreps
from skala.functional.utils.pad_ragged import pad_ragged, unpad_ragged
from skala.functional.utils.symmetric_contraction import SymmetricContraction

ANGSTROM_TO_BOHR = 1.88973


def _prepare_features_raw(
    mol: dict[str, torch.Tensor], eps: float = 1e-5
) -> torch.Tensor:
    """Compute log-space semi-local features from packed density data.

    Args:
        mol: Dictionary of packed molecular features (grid_per_atom, atoms, …).
        eps: Small constant for numerical stability in log.

    Returns:
        Tensor of shape ``(grid_per_atom, atoms, 7)``.
    """
    x = torch.cat(
        [
            mol["density"].permute(1, 2, 0),
            (mol["grad"] ** 2).sum(1).permute(1, 2, 0),
            mol["kin"].permute(1, 2, 0),
            (mol["grad"].sum(0) ** 2).sum(0).unsqueeze(-1),
        ],
        dim=-1,
    )

    # Cast to double to work around a TorchScript gradient bug with torch.abs.
    # See PR #15759 in the livdft repository for details.
    x = x.double()

    return torch.log(torch.abs(x) + eps)


class SemiLocalFeatures(nn.Module):
    """Compute semi-local (ab, ba) feature pairs with a pre-buffered permutation index."""

    _PERM = [1, 0, 3, 2, 5, 4, 6]
    _feature_perm: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "_feature_perm",
            torch.tensor(self._PERM, dtype=torch.long),
            persistent=False,
        )

    def forward(
        self, mol: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = _prepare_features_raw(mol)
        features_ab = features
        features_ba = features.index_select(-1, self._feature_perm)
        return features_ab, features_ba


class ExpRadialScaleModel(nn.Module):
    """Learnable radial basis using exponentially-spaced Gaussians.

    Args:
        embedding_size: Number of radial basis functions.
    """

    temps: torch.Tensor

    def __init__(self, embedding_size: int = 8) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        min_std = 0.32 * ANGSTROM_TO_BOHR / 2
        max_std = 2.32 * ANGSTROM_TO_BOHR / 2
        self.register_buffer(
            "temps", 2 * torch.linspace(min_std, max_std, embedding_size) ** 2
        )

    def forward(self, dist2: torch.Tensor) -> torch.Tensor:
        """Compute radial basis values.

        Args:
            dist2: Squared distances, shape ``(…, 1)``.

        Returns:
            Radial basis values, shape ``(…, embedding_size)``.
        """
        t = self.temps
        dim = 3
        return (
            2 / (dim * t * (math.pi * t) ** (0.5 * dim)) * torch.exp(-dist2 / t) * dist2
        )


class SkalaFunctional(ExcFunctionalBase):
    """Skala neural exchange-correlation functional.

    This model operates on per-atom packed grid features and uses multiple
    equivariant non-local message-passing layers with symmetric contraction.

    Args:
        lmax: Maximum angular momentum order for spherical harmonics.
        num_mid_layers: Total number of mid-layers (local + non-local).
        num_non_local_layers: How many mid-layers are non-local.
        non_local_hidden_nf: Number of channels in the non-local model.
        coarse_linear_type: Type of coarse linear layer.
        correlation: Correlation order for the symmetric contraction.
    """

    features = [
        "density",
        "kin",
        "grad",
        "grid_coords",
        "grid_weights",
        "atomic_grid_weights",
        "atomic_grid_sizes",
        "coarse_0_atomic_coords",
        "atomic_grid_size_bound_shape",
    ]

    def __init__(
        self,
        lmax: int = 3,
        num_mid_layers: int = 3,
        num_non_local_layers: int = 2,
        non_local_hidden_nf: int = 16,
        coarse_linear_type: str | None = "decomp-identity",
        correlation: int = 3,
    ) -> None:
        super().__init__()

        assert 0 <= num_non_local_layers <= num_mid_layers

        self.num_scalar_features = 7
        self.lmax = lmax
        self.num_mid_layers = num_mid_layers
        self.num_non_local_layers = num_non_local_layers
        self.non_local_hidden_nf = non_local_hidden_nf
        self.num_feats = 256

        self.semi_local_features = SemiLocalFeatures()

        self.input_model = torch.nn.Sequential(
            nn.Linear(self.num_scalar_features, self.num_feats),
            nn.SiLU(),
            nn.Linear(self.num_feats, self.num_feats),
            nn.SiLU(),
        )

        if self.num_non_local_layers > 0:
            self.sph_irreps = Irreps.spherical_harmonics(self.lmax, p=1)
            self.spherical_harmonics = o3.SphericalHarmonics(
                irreps_out=str(self.sph_irreps),
                normalize=False,
                normalization="norm",
            )
            self.non_local_layers = torch.nn.ModuleList(
                [
                    NonLocalModel(
                        input_nf=self.num_feats,
                        hidden_nf=self.non_local_hidden_nf,
                        lmax=self.lmax,
                        edge_irreps=self.sph_irreps,
                        coarse_linear_type=coarse_linear_type,
                        correlation=correlation,
                    )
                    for _ in range(self.num_non_local_layers)
                ]
            )
            self.radial_basis = ExpRadialScaleModel(self.non_local_hidden_nf)
        else:
            raise NotImplementedError("Non-local model must be enabled.")

        self.output_model = torch.nn.Sequential(
            *[
                module
                for _ in range(self.num_mid_layers - self.num_non_local_layers)
                for module in [
                    nn.Linear(self.num_feats, self.num_feats),
                    nn.SiLU(),
                ]
            ],
            nn.Linear(self.num_feats, 1),
            ScaledSigmoid(scale=2.0),
        )

        self._init_weights()

    # Keys introduced after the original checkpoint format (deterministic buffers
    # that can be reconstructed from __init__ args).
    _RECONSTRUCTABLE_BUFFER_PREFIXES = ("radial_basis.", "semi_local_features.")

    def load_state_dict(  # type: ignore  # needs mutable dict
        self,
        state_dict: dict[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Load state_dict with backward compatibility for older checkpoints."""
        if strict:
            current_sd = self.state_dict()
            # Add missing reconstructable buffers from the current model.
            for key in current_sd:
                if key not in state_dict and any(
                    key.startswith(p) for p in self._RECONSTRUCTABLE_BUFFER_PREFIXES
                ):
                    state_dict[key] = current_sd[key]
            # Remove extra reconstructable buffers that the checkpoint carries but
            # the model does not expose (e.g. non-persistent buffers).
            extra = set(state_dict) - set(current_sd)
            for key in extra:
                if any(
                    key.startswith(p) for p in self._RECONSTRUCTABLE_BUFFER_PREFIXES
                ):
                    del state_dict[key]
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _init_weights(self) -> None:
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
        return cast(nn.Linear, self.input_model[0]).weight.dtype

    def pack_features(
        self, mol_feats: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Pack flat features into dense (grid_per_atom, atoms, …) layout.

        Args:
            mol_feats: Flat molecular features from grid evaluation.

        Returns:
            Packed features dictionary.
        """
        atomic_grid_sizes = mol_feats["atomic_grid_sizes"]
        size_bound = mol_feats["atomic_grid_size_bound_shape"].shape[0]

        packed_mol_feats: dict[str, torch.Tensor] = {}
        for key in self.features:
            if key == "atomic_grid_weights":
                packed_mol_feats[key] = pad_ragged(
                    mol_feats[key], atomic_grid_sizes, size_bound
                ).T  # (max_grid_size, num_atoms)
            elif key == "grid_weights":
                continue
            elif key == "grid_coords":
                packed_mol_feats[key] = pad_ragged(
                    mol_feats[key], atomic_grid_sizes, size_bound
                ).permute(1, 0, 2)  # (max_grid_size, num_atoms, 3)
            elif key == "coarse_0_atomic_coords":
                packed_mol_feats[key] = mol_feats[key]
            elif key == "density":
                packed_mol_feats[key] = pad_ragged(
                    mol_feats[key].T, atomic_grid_sizes, size_bound
                ).permute(2, 1, 0)  # (2, max_grid_size, num_atoms)
            elif key == "grad":
                packed_mol_feats[key] = pad_ragged(
                    mol_feats[key].permute(2, 0, 1), atomic_grid_sizes, size_bound
                ).permute(2, 3, 1, 0)  # (2, 3, max_grid_size, num_atoms)
            elif key == "kin":
                packed_mol_feats[key] = pad_ragged(
                    mol_feats[key].T, atomic_grid_sizes, size_bound
                ).permute(2, 1, 0)  # (2, max_grid_size, num_atoms)
            elif key in ("atomic_grid_sizes", "atomic_grid_size_bound_shape"):
                continue
            else:
                raise ValueError(f"Unexpected key: {key}")

        return packed_mol_feats

    def get_exc(self, mol: dict[str, torch.Tensor]) -> torch.Tensor:
        exc_density = self._get_exc_density_padded(mol).double()
        grid_weights = (
            pad_ragged(
                mol["grid_weights"],
                mol["atomic_grid_sizes"],
                mol["atomic_grid_size_bound_shape"].shape[0],
            )
            .T.double()
            .reshape(-1)
        )

        return (exc_density * grid_weights).sum()

    def get_exc_density(self, mol: dict[str, torch.Tensor]) -> torch.Tensor:
        padded = self._get_exc_density_padded(mol)
        sizes = mol["atomic_grid_sizes"]
        size_bound = mol["atomic_grid_size_bound_shape"].shape[0]
        num_atoms = sizes.shape[0]
        total_grid_points = mol["grid_weights"].shape[0]
        padded_2d = padded.reshape(size_bound, num_atoms).T
        return unpad_ragged(padded_2d, sizes, total_grid_points)

    def _get_exc_density_padded(self, mol: dict[str, torch.Tensor]) -> torch.Tensor:
        mol = self.pack_features(mol)
        grid_coords = mol["grid_coords"]
        atomic_grid_weights = mol["atomic_grid_weights"]
        coarse_coords = mol["coarse_0_atomic_coords"]
        features_ab, features_ba = self.semi_local_features(mol)

        # Learned symmetrized features
        spin_feats = torch.stack([features_ab, features_ba], dim=0)
        spin_feats = spin_feats.to(self.dtype)
        spin_feats = self.input_model(spin_feats)
        features = spin_feats.sum(0) / 2

        # Non-local model
        if self.num_non_local_layers > 0:
            # grid_coords: (num_fine, num_coarse, 3)
            # coarse_coords: (num_coarse, 3)
            directions = grid_coords.double() - coarse_coords.double()
            distances = (directions**2 + 1e-20).sum(-1) ** 0.5
            directions = directions / distances[:, :, None]

            directions = directions.to(self.dtype)
            distances = distances.to(self.dtype)

            distance_ft = self.radial_basis(
                distances.unsqueeze(-1) ** 2
            )  # (#fine, #coarse, hidden_nf)

            direction_ft = self.spherical_harmonics(
                directions
            )  # (num_fine, num_coarse, (lmax+1)^2)

            exp_m1_rho_total = torch.exp(-mol["density"].sum(0).unsqueeze(-1)).to(
                self.dtype
            )

            for non_local_layer in self.non_local_layers:
                features = non_local_layer(
                    features,
                    distance_ft,
                    direction_ft,
                    atomic_grid_weights,
                    exp_m1_rho_total,
                )

        enhancement_factor = self.output_model(features)
        return enhancement_density_inner_product(
            enhancement_factor=enhancement_factor.view(-1, 1),
            density=mol["density"].reshape(2, -1),
        )

    def reset_parameters(self) -> None:
        self._init_weights()


class NonLocalModel(nn.Module):
    """Equivariant non-local message-passing layer.

    Args:
        input_nf: Number of input scalar features.
        hidden_nf: Number of hidden channels per irrep.
        lmax: Maximum angular momentum.
        edge_irreps: Irreps for edge features (spherical harmonics).
        coarse_linear_type: Type of O3-equivariant linear on coarse features.
        correlation: Correlation order for symmetric contraction.
    """

    def __init__(
        self,
        input_nf: int,
        hidden_nf: int,
        lmax: int,
        edge_irreps: Irreps,
        coarse_linear_type: str | None = None,
        correlation: int = 1,
    ):
        super().__init__()

        self.input_nf = input_nf
        self.hidden_nf = hidden_nf
        self.in_irreps = Irreps(f"{self.hidden_nf}x0e")
        self.out_irreps = Irreps(f"{self.hidden_nf}x0e")
        self.lmax = lmax
        self.hidden_irreps = Irreps(
            "+".join([f"{hidden_nf}x{i}e" for i in range(self.lmax + 1)])
        )
        self.edge_irreps = edge_irreps
        self.coarse_linear_type = coarse_linear_type
        assert correlation >= 1
        self.correlation = correlation

        self.pre_down_layer = torch.nn.Sequential(
            nn.Linear(self.input_nf, self.hidden_nf),
            torch.nn.SiLU(),
        )
        torch.nn.init.xavier_uniform_(self.pre_down_layer[0].weight)  # type: ignore
        torch.nn.init.zeros_(self.pre_down_layer[0].bias)  # type: ignore

        self.tp_down = TensorProduct(
            self.in_irreps,
            self.edge_irreps,
            self.hidden_irreps,
        )

        if coarse_linear_type is not None:
            if coarse_linear_type == "decomp":
                self.coarse_linear = O3Linear(self.hidden_irreps, self.hidden_irreps)
            elif coarse_linear_type == "decomp-identity":
                self.coarse_linear = O3Linear(self.hidden_irreps, self.hidden_irreps)
                o3_identity_init(self.coarse_linear)
            elif coarse_linear_type == "sketch":
                self.coarse_linear = O3Linear(
                    self.hidden_irreps,
                    (self.hidden_irreps * correlation).sort().irreps.simplify(),
                )
            elif coarse_linear_type == "sketch-identity":
                self.coarse_linear = O3Linear(
                    self.hidden_irreps,
                    (self.hidden_irreps * correlation).sort().irreps.simplify(),
                )
                o3_identity_init(self.coarse_linear, out_dim_multiplier=correlation)
            else:
                raise ValueError(f"Unknown coarse_linear method: {coarse_linear_type}")

        if correlation > 1:
            self.symmetric_product = SymmetricContraction(
                irreps_in=self.hidden_irreps,
                irreps_out=self.hidden_irreps,
                correlation=correlation,
            )

        self.tp_up = TensorProduct(
            self.hidden_irreps,
            self.edge_irreps,
            self.out_irreps,
            x1_contains_r=False,
        )

        self.post_up_layer = torch.nn.Sequential(
            nn.Linear(self.hidden_nf, self.hidden_nf),
            torch.nn.SiLU(),
        )
        torch.nn.init.xavier_uniform_(self.post_up_layer[0].weight)  # type: ignore
        torch.nn.init.zeros_(self.post_up_layer[0].bias)  # type: ignore

        self.concat_layer = torch.nn.Sequential(
            nn.Linear(self.input_nf + self.hidden_nf, self.input_nf),
            nn.SiLU(),
        )

    def forward(
        self,
        h: torch.Tensor,  # (num_fine, num_coarse, input_nf)
        distance_ft: torch.Tensor,
        direction_ft: torch.Tensor,
        grid_weights: torch.Tensor,
        exp_m1_rho_total: torch.Tensor,
    ) -> torch.Tensor:
        features = h  # skip connection
        h = self.pre_down_layer(h)

        # Process (fine -> coarse) features on each edge.
        down = self.tp_down(h, direction_ft, distance_weights=distance_ft)

        # Sum data from incoming edges into each coarse point.
        h_coarse = torch.einsum("gck,gc->ck", down.double(), grid_weights.double()).to(
            self.dtype
        )

        # Correlate the coarse features.
        if self.coarse_linear_type is not None:
            h_coarse = self.coarse_linear(h_coarse)

        if self.correlation > 1:
            h_coarse = self.symmetric_product(h_coarse)

        # Process (coarse -> fine) features on each edge.
        h_fine = self.tp_up(h_coarse, direction_ft, distance_weights=distance_ft)

        # Process the fine points.
        h_fine = self.post_up_layer(h_fine)

        # Non-linear transform with skip connection
        features = torch.cat([features, h_fine * exp_m1_rho_total], dim=-1)
        return self.concat_layer(features)

    @property
    def dtype(self) -> torch.dtype:
        return self.pre_down_layer[0].weight.dtype  # type: ignore


class TensorProduct(nn.Module):
    """Equivariant tensor product with learned weights.

    Uses batched gather-index operations for efficient ``tp_down`` (all-scalar
    input) and ``tp_up`` (all-scalar output) patterns.

    Args:
        irreps_in1: Irreps for the first input.
        irreps_in2: Irreps for the second input (edge features).
        irreps_out: Output irreps.
        x1_contains_r: If True, x1 has a spatial (r) dimension.
    """

    _tp_down_xw_idx: torch.Tensor
    _tp_down_sph_idx: torch.Tensor
    _tp_down_norm: torch.Tensor

    _tp_up_x1_gather: torch.Tensor
    _tp_up_instr_idx: torch.Tensor
    _tp_up_norm: torch.Tensor

    def __init__(
        self,
        irreps_in1: Irreps,
        irreps_in2: Irreps,
        irreps_out: Irreps,
        x1_contains_r: bool = True,
    ):
        super().__init__()

        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = irreps_out
        self.x1_contains_r = x1_contains_r

        self.instr = [
            (i_1, i_2, i_out)
            for i_1, (_, ir_1) in enumerate(irreps_in1)
            for i_2, (_, ir_2) in enumerate(irreps_in2)
            for i_out, (_, ir_out) in enumerate(irreps_out)
            if ir_out in ir_1 * ir_2  # type: ignore  # Irrep.__mul__ not in stubs
        ]

        self.slices = [irreps_in1.slices(), irreps_in2.slices(), irreps_out.slices()]

        self._setup_batched_tp()
        assert self._batched_tp_mode is not None, (
            f"TensorProduct requires batched TP support but got incompatible irreps: "
            f"in1={irreps_in1}, in2={irreps_in2}, out={irreps_out}"
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        def num_elements(ins: tuple[int, int, int]) -> int:
            return int(self.irreps_in1[ins[0]].mul * self.irreps_in2[ins[1]].mul)

        for idx, ins in enumerate(self.instr):
            num_in = sum(num_elements(ins_) for ins_ in self.instr if ins_[2] == ins[2])
            num_out = self.irreps_out[ins[2]].mul
            x = (6 / (num_in + num_out)) ** 0.5
            self._batched_W.data[idx].uniform_(-x, x)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Backward-compatible loading: convert old per-instruction weight keys to _batched_W."""
        if self._batched_tp_mode is not None:
            old_keys = [f"{prefix}weight_{i1}_{i2}_{io}" for i1, i2, io in self.instr]
            has_old = all(k in state_dict for k in old_keys)
            has_new = f"{prefix}_batched_W" in state_dict

            if has_old and not has_new:
                stacked = torch.stack([state_dict.pop(k).squeeze(1) for k in old_keys])
                state_dict[f"{prefix}_batched_W"] = stacked

            # Drop legacy w3j buffers that are no longer registered.
            for i1, i2, io in self.instr:
                state_dict.pop(f"{prefix}w3j_{i1}_{i2}_{io}", None)

            # Fill reconstructable buffers from current model state.
            current_sd = dict(self.named_buffers())
            for buf_name, buf_val in current_sd.items():
                full_key = f"{prefix}{buf_name}"
                if full_key not in state_dict:
                    state_dict[full_key] = buf_val

        super()._load_from_state_dict(  # type: ignore[no-untyped-call]
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _setup_batched_tp(self) -> None:
        """Detect batched mode and pre-compute gather indices.

        Supports two patterns:

        - ``tp_down``: all-scalar input (l1=0), single input irrep, v=1 edge
          multiplicities.  Batches weight matmuls into a single einsum, then
          scatter-multiplies with x2.
        - ``tp_up``: all-scalar output (l_out=0), single output irrep, v=1 edge
          multiplicities.  Computes per-instruction dot products, then batches
          weight matmuls + sum.
        """
        self._batched_tp_mode: str | None = None

        if len(self.instr) <= 1:
            return

        # Common checks: all v=1 and all (u, w) same shape.
        if not all(self.irreps_in2[i_2].mul == 1 for _, i_2, _ in self.instr):
            return
        shapes = {
            (self.irreps_in1[i_1].mul, self.irreps_out[i_out].mul)
            for i_1, _, i_out in self.instr
        }
        if len(shapes) != 1:
            return
        u, w = next(iter(shapes))

        if self.x1_contains_r:
            # tp_down: all l1=0, single input irrep group.
            i1_values = {i_1 for i_1, _, _ in self.instr}
            if len(i1_values) == 1 and self.irreps_in1[next(iter(i1_values))].ir.l == 0:
                self._batched_tp_mode = "tp_down"
                self._tp_batch_u = u
                self._tp_batch_w = w
                self._tp_down_x2_info = [
                    (
                        self.slices[1][i_2].start,
                        self.slices[1][i_2].stop,
                        self.irreps_out[i_out].ir.l,
                    )
                    for _, i_2, i_out in self.instr
                ]
                # Pre-compute gather indices for vectorized tp_down.
                xw_idx = []
                sph_idx = []
                norm_vals = []
                for b_idx in range(len(self.instr)):
                    x2_start, _, l_out = self._tp_down_x2_info[b_idx]
                    dim_l = 2 * l_out + 1
                    nf = 1.0 if l_out == 0 else 1.0 / math.sqrt(dim_l)
                    for c in range(w):
                        for m in range(dim_l):
                            xw_idx.append(b_idx * w + c)
                            sph_idx.append(x2_start + m)
                            norm_vals.append(nf)
                self.register_buffer(
                    "_tp_down_xw_idx",
                    torch.tensor(xw_idx, dtype=torch.long),
                )
                self.register_buffer(
                    "_tp_down_sph_idx",
                    torch.tensor(sph_idx, dtype=torch.long),
                )
                self.register_buffer("_tp_down_norm", torch.tensor(norm_vals))
        else:
            # tp_up: all l_out=0, single output irrep group.
            i_out_values = {i_out for _, _, i_out in self.instr}
            if len(i_out_values) == 1 and all(
                self.irreps_out[i_out].ir.l == 0 for _, _, i_out in self.instr
            ):
                self._batched_tp_mode = "tp_up"
                self._tp_batch_u = u
                self._tp_batch_w = w
                self._tp_up_x1_info = [
                    (
                        self.slices[0][i_1].start,
                        self.slices[0][i_1].stop,
                        self.irreps_in1[i_1].ir.dim,
                    )
                    for i_1, _, _ in self.instr
                ]
                self._tp_up_x2_info = [
                    (self.slices[1][i_2].start, self.slices[1][i_2].stop)
                    for _, i_2, _ in self.instr
                ]
                # Pre-compute gather index to rearrange x1 from interleaved
                # layout to sph-major layout aligned with x2.
                sph_total = self.irreps_in2.dim
                x1_gather = torch.zeros(sph_total * u, dtype=torch.long)
                instr_idx = torch.zeros(sph_total, dtype=torch.long)
                norm_vals_t = torch.zeros(sph_total)
                for b_idx in range(len(self.instr)):
                    x1_start, _, dim_b = self._tp_up_x1_info[b_idx]
                    x2_start, _ = self._tp_up_x2_info[b_idx]
                    nf = 1.0 / math.sqrt(dim_b)
                    for m in range(dim_b):
                        s = x2_start + m
                        instr_idx[s] = b_idx
                        norm_vals_t[s] = nf
                        for ch in range(u):
                            x1_gather[s * u + ch] = x1_start + ch * dim_b + m
                self.register_buffer("_tp_up_x1_gather", x1_gather)
                self.register_buffer("_tp_up_instr_idx", instr_idx)
                self.register_buffer("_tp_up_norm", norm_vals_t)

        if self._batched_tp_mode is not None:
            self._batched_W = nn.Parameter(torch.empty(len(self.instr), u, w))

    def _forward_batched_tp_down(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        distance_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Batched forward for scalar-input tensor product (tp_down)."""
        W = self._batched_W  # (B, u, w)
        xw = torch.einsum("sru, buw -> bsrw", x1, W)  # (B, s, r, w)
        if distance_weights is not None:
            xw = xw * distance_weights  # broadcasts over B
        # Flatten instruction & channel dims: (B, s, r, w) -> (s, r, B*w)
        xw_flat = xw.permute(1, 2, 0, 3).reshape(xw.shape[1], xw.shape[2], -1)
        return (
            xw_flat[:, :, self._tp_down_xw_idx]
            * x2[:, :, self._tp_down_sph_idx]
            * self._tp_down_norm.to(x2.dtype)
        )

    def _forward_batched_tp_up(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        distance_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Batched forward for scalar-output tensor product (tp_up)."""
        W = self._batched_W  # (B, u, w)
        u = self._tp_batch_u
        c = x1.shape[0]
        sph = x2.shape[-1]

        # Rearrange x1: (c, hidden) -> (c, sph, u) aligned with x2's sph layout
        x1_r = x1[:, self._tp_up_x1_gather].reshape(c, sph, u)

        # Expand W per sph position with normalization: (B, u, w) -> (sph, u, w)
        W_sph = W[self._tp_up_instr_idx] * self._tp_up_norm[:, None, None].to(W.dtype)

        # Linear transform on x1: (c, sph, u) @ (sph, u, w) -> (c, sph, w)
        x1_W = torch.einsum("csu, suw -> csw", x1_r, W_sph)

        # Inner product with x2: sum_l <x1_l, x2_l> = <x1, x2>
        out = torch.einsum("csw, fcs -> fcw", x1_W, x2)
        if distance_weights is not None:
            out = distance_weights * out
        return out

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        *,
        distance_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert (len(x1.size()) == 2 and not self.x1_contains_r) or (
            len(x1.size()) == 3 and self.x1_contains_r
        ), "x1 must be 2D or 3D"
        assert len(x2.size()) == 3, "x2 must be 3D"
        if self._batched_tp_mode == "tp_down":
            return self._forward_batched_tp_down(x1, x2, distance_weights)
        return self._forward_batched_tp_up(x1, x2, distance_weights)


class O3Linear(nn.Module):
    """Equivariant linear layer operating on irreps."""

    optimize_einsums = True
    script_codegen = False

    def __init__(self, irreps_in: Irreps, irreps_out: Irreps):
        super().__init__()

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out

        self.instr = [
            (i_in, i_out)
            for i_in, (_, ir_in) in enumerate(irreps_in)
            for i_out, (_, ir_out) in enumerate(irreps_out)
            if ir_in == ir_out
        ]

        self.weight_numel = sum(
            self.irreps_in[i_in].mul * self.irreps_out[i_out].mul
            for i_in, i_out in self.instr
        )

        for i_in, i_out in self.instr:
            self.register_parameter(
                f"weight_{i_in}_{i_out}",
                nn.Parameter(
                    torch.randn(
                        self.irreps_in[i_in].mul,
                        self.irreps_out[i_out].mul,
                    )
                ),
            )

        self.slices = [irreps_in.slices(), irreps_out.slices()]
        self.reset_parameters()
        self._o3_linear = self.generate_o3_linear_code()

    def generate_o3_linear_code(self) -> fx.GraphModule:
        graphmod = _o3_linear_codegen(*self.linear_params)

        if self.optimize_einsums:
            example_inputs = (
                torch.randn(3, self.irreps_in.dim),
                *self.weight_list,
            )
            graphmod = optimize_einsums_full(graphmod, example_inputs)

        if self.script_codegen:
            graphmod = torch.jit.script(jitable(graphmod))

        return graphmod

    @property
    def linear_params(self) -> tuple[Any, ...]:
        return (
            self.instr,
            convert_irreps(self.irreps_in),
            convert_irreps(self.irreps_out),
            [[(ss.start, ss.stop) for ss in s] for s in self.slices],
        )

    def reset_parameters(self) -> None:
        def num_elements(ins: tuple[int, int]) -> int:
            return int(self.irreps_in[ins[0]].mul)

        for ins in self.instr:
            i_in, i_out = ins
            num_in = sum(num_elements(ins_) for ins_ in self.instr if ins_[1] == ins[1])
            num_out = self.irreps_out[ins[1]].mul
            x = (6 / (num_in + num_out)) ** 0.5
            getattr(self, f"weight_{i_in}_{i_out}").data.uniform_(-x, x)

    @property
    def weight_list(self) -> list[torch.Tensor]:
        return [getattr(self, f"weight_{i_in}_{i_out}") for i_in, i_out in self.instr]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._o3_linear(x, *self.weight_list)


def o3_identity_init(linear: O3Linear, out_dim_multiplier: int = 1) -> None:
    """Initialize an O3Linear layer as an identity mapping."""
    for ins in linear.instr:
        if (
            linear.irreps_in[ins[0]].mul * out_dim_multiplier
            != linear.irreps_out[ins[1]].mul
        ):
            raise ValueError(
                f"Input and output irreps must match, got "
                f"{linear.irreps_in[ins[0]].mul} * {out_dim_multiplier} "
                f"!= {linear.irreps_out[ins[1]].mul}"
            )
        getattr(linear, f"weight_{ins[0]}_{ins[1]}").data.copy_(
            torch.eye(linear.irreps_in[ins[0]].mul).repeat(1, out_dim_multiplier)
        )


# irreps_format: list of (mul, ir.l, ir.dim)
def convert_irreps(irreps: Irreps) -> list[tuple[int, int, int]]:
    return [(mulir.mul, mulir.ir.l, mulir.ir.dim) for mulir in irreps]


def _o3_linear_codegen(
    instr: list[tuple[int, int]],
    irreps_in: list[tuple[int, int, int]],
    irreps_out: list[tuple[int, int, int]],
    slices: list[list[tuple[int, int]]],
) -> fx.GraphModule:
    graph = fx.Graph()
    tracer = fx.proxy.GraphAppendingTracer(graph)
    x1 = fx.Proxy(graph.placeholder("x1", torch.Tensor), tracer=tracer)

    weights = [
        fx.Proxy(
            graph.placeholder(f"weight_{i_in}_{i_out}", torch.Tensor),
            tracer=tracer,
        )
        for i_in, i_out in instr
    ]

    outs: list[Any] = list()
    for (i_in, i_out), w in zip(instr, weights, strict=True):
        x1_i = x1[:, slices[0][i_in][0] : slices[0][i_in][1]]  # type: ignore
        outs.append(
            torch.einsum(
                "sui,uv->svi",
                x1_i.view(-1, irreps_in[i_in][0], irreps_in[i_in][2]),
                w,
            ).reshape(-1, irreps_out[i_out][0] * irreps_out[i_out][2])
        )

    out: list[Any] = [
        sum(out for ins, out in zip(instr, outs, strict=True) if ins[1] == i_out)
        for i_out, (mul, *_) in enumerate(irreps_out)
        if mul > 0
    ]
    if len(out) > 1:
        concatenated: Any = torch.cat(out, dim=-1)
    else:
        concatenated = out[0]

    graph.output(concatenated.node, torch.Tensor)
    graph.lint()  # type: ignore[no-untyped-call]

    return fx.GraphModule(torch.nn.Module(), graph)
