# SPDX-License-Identifier: MIT

"""Tests for the pad_ragged, irreps, and symmetric_contraction utilities."""

from collections.abc import Iterator

import pytest
import torch

from skala.functional.utils.irreps import Irrep, Irreps, MulIr
from skala.functional.utils.pad_ragged import pad_ragged, unpad_ragged


class TestPadRagged:
    def test_round_trip_uniform(self) -> None:
        """pad then unpad recovers the original flat tensor."""
        sizes = torch.tensor([5, 5, 5])
        flat = torch.randn(15, 3)
        padded = pad_ragged(flat, sizes, 5)
        assert padded.shape == (3, 5, 3)
        recovered = unpad_ragged(padded, sizes, 15)
        torch.testing.assert_close(recovered, flat)

    def test_round_trip_variable(self) -> None:
        """pad then unpad works with variable sizes."""
        sizes = torch.tensor([3, 7, 5])
        total = sizes.sum().item()
        assert isinstance(total, int)
        flat = torch.randn(total, 2)
        padded = pad_ragged(flat, sizes, 7)
        assert padded.shape == (3, 7, 2)
        recovered = unpad_ragged(padded, sizes, total)
        torch.testing.assert_close(recovered, flat)

    def test_single_sequence_fast_path(self) -> None:
        """Single sequence (1 atom) takes the fast path."""
        sizes = torch.tensor([10])
        flat = torch.randn(10, 4)
        padded = pad_ragged(flat, sizes, 10)
        assert padded.shape == (1, 10, 4)
        recovered = unpad_ragged(padded, sizes, 10)
        torch.testing.assert_close(recovered, flat)

    def test_1d_input(self) -> None:
        """pad_ragged works with 1D inputs."""
        sizes = torch.tensor([3, 2])
        flat = torch.randn(5)
        padded = pad_ragged(flat, sizes, 3)
        assert padded.shape == (2, 3)
        recovered = unpad_ragged(padded, sizes, 5)
        torch.testing.assert_close(recovered, flat)

    def test_padding_is_zero(self) -> None:
        """Padded elements should be zero."""
        sizes = torch.tensor([2, 5])
        flat = torch.randn(7, 3) + 10  # shift away from zero
        padded = pad_ragged(flat, sizes, 5)
        # Atom 0 has 2 elements, so positions [2:5] should be zero
        assert (padded[0, 2:5, :] == 0).all()

    def test_negative_sizes_raises(self) -> None:
        """pad_ragged should reject negative sizes."""
        sizes = torch.tensor([2, -1])
        flat = torch.randn(1, 3)
        with pytest.raises(ValueError, match="non-negative"):
            pad_ragged(flat, sizes, 5)

    def test_mismatched_data_length_raises(self) -> None:
        """pad_ragged should reject data length != sum(sizes)."""
        sizes = torch.tensor([2, 3])
        flat = torch.randn(6, 3)  # should be 5
        with pytest.raises(ValueError, match="data length"):
            pad_ragged(flat, sizes, 5)

    def test_unpad_negative_sizes_raises(self) -> None:
        """unpad_ragged should reject negative sizes."""
        padded = torch.randn(2, 5, 3)
        sizes = torch.tensor([2, -1])
        with pytest.raises(ValueError, match="non-negative"):
            unpad_ragged(padded, sizes, 1)


class TestIrreps:
    def test_irrep_from_string(self) -> None:
        ir = Irrep("1e")
        assert ir.l == 1
        assert ir.p == 1
        assert ir.dim == 3

    def test_irrep_from_tuple(self) -> None:
        ir = Irrep(2, -1)
        assert ir.l == 2
        assert ir.p == -1
        assert ir.dim == 5

    def test_irreps_from_string(self) -> None:
        irreps = Irreps("3x0e+2x1o")
        assert len(irreps) == 2
        assert irreps[0].mul == 3
        assert irreps[0].ir.l == 0
        assert irreps[1].mul == 2
        assert irreps[1].ir.l == 1
        assert irreps[1].ir.p == -1

    def test_irreps_dim(self) -> None:
        irreps = Irreps("3x0e+2x1e")
        assert irreps.dim == 3 * 1 + 2 * 3  # 9

    def test_irreps_spherical_harmonics(self) -> None:
        irreps = Irreps.spherical_harmonics(2, p=1)
        # 1x0e + 1x1e + 1x2e
        assert irreps.dim == 1 + 3 + 5

    def test_irreps_slices(self) -> None:
        irreps = Irreps("2x0e+3x1e")
        slices = irreps.slices()
        assert slices[0] == slice(0, 2)
        assert slices[1] == slice(2, 11)

    def test_irrep_multiplication(self) -> None:
        ir0 = Irrep(0, 1)
        ir1 = Irrep(1, -1)
        products = list(ir0 * ir1)
        assert len(products) == 1
        assert products[0].l == 1

    def test_mul_ir_equality(self) -> None:
        a = MulIr(3, Irrep(0, 1))
        b = MulIr(3, Irrep(0, 1))
        assert a.mul == b.mul and a.ir == b.ir

    def test_irreps_simplify(self) -> None:
        irreps = Irreps("2x0e+3x0e+1x1e")
        simplified = irreps.simplify()
        assert simplified[0].mul == 5
        assert simplified[0].ir.l == 0

    def test_irreps_sort(self) -> None:
        irreps = Irreps("1x1e+1x0e")
        sorted_irreps = irreps.sort()
        assert sorted_irreps.irreps[0].ir.l == 0
        assert sorted_irreps.irreps[1].ir.l == 1

    def test_irreps_lmax(self) -> None:
        irreps = Irreps("1x0e+1x1e+1x3o")
        assert irreps.lmax == 3

    def test_irreps_mul_scalar(self) -> None:
        irreps = Irreps("2x0e+1x1e")
        scaled = irreps * 3
        # Irreps * int repeats the whole sequence 3 times
        assert len(scaled) == 6  # 3 copies of [2x0e, 1x1e]


class TestSymmetricContraction:
    @pytest.fixture(autouse=True)
    def _isolated_rng(self) -> Iterator[None]:
        with torch.random.fork_rng():
            yield

    def test_output_shape(self) -> None:
        from e3nn import o3 as e3nn_o3

        from skala.functional.utils.symmetric_contraction import SymmetricContraction

        torch.manual_seed(42)
        irreps_in = e3nn_o3.Irreps("3x0e+3x1e")
        irreps_out = e3nn_o3.Irreps("3x0e+3x1e")
        sc = SymmetricContraction(irreps_in, irreps_out, correlation=2)

        x = torch.randn(5, irreps_in.dim)
        out = sc(x)
        assert out.shape == (5, irreps_out.dim)

    def test_output_deterministic(self) -> None:
        from e3nn import o3 as e3nn_o3

        from skala.functional.utils.symmetric_contraction import SymmetricContraction

        torch.manual_seed(42)
        irreps_in = e3nn_o3.Irreps("3x0e+3x1e")
        irreps_out = e3nn_o3.Irreps("3x0e+3x1e")
        sc = SymmetricContraction(irreps_in, irreps_out, correlation=2)

        torch.manual_seed(123)
        x = torch.randn(5, irreps_in.dim)
        out1 = sc(x)
        out2 = sc(x)
        torch.testing.assert_close(out1, out2)
