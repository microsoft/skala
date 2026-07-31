from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import NamedTuple, cast

import numpy as np
import pytest
import torch
from pyscf import dft, gto, lib
from pyscf.dft import numint as pyscf_numint
from pytest_benchmark.fixture import BenchmarkFixture

from skala.functional.base import ExcFunctionalBase
from skala.pyscf.numint import SkalaNumInt, _should_screen_aos

THREAD_COUNT = 4
MAX_MEMORY_MB = 2000

NAPHTHALENE = """
C -1.2280  0.7090 0.0
C -1.2280 -0.7090 0.0
C  0.0000  1.4180 0.0
C  0.0000 -1.4180 0.0
C  1.2280  0.7090 0.0
C  1.2280 -0.7090 0.0
C  2.4560  1.4180 0.0
C  2.4560 -1.4180 0.0
C  3.6840  0.7090 0.0
C  3.6840 -0.7090 0.0
H -2.1700  1.2530 0.0
H -2.1700 -1.2530 0.0
H  0.0000  2.5060 0.0
H  0.0000 -2.5060 0.0
H  2.4560  2.5060 0.0
H  2.4560 -2.5060 0.0
H  4.6260  1.2530 0.0
H  4.6260 -1.2530 0.0
"""

ANTHRACENE = """
C -1.2280  0.7090 0.0
C -1.2280 -0.7090 0.0
C  0.0000  1.4180 0.0
C  0.0000 -1.4180 0.0
C  1.2280  0.7090 0.0
C  1.2280 -0.7090 0.0
C  2.4560  1.4180 0.0
C  2.4560 -1.4180 0.0
C  3.6840  0.7090 0.0
C  3.6840 -0.7090 0.0
C  4.9120  1.4180 0.0
C  4.9120 -1.4180 0.0
C  6.1400  0.7090 0.0
C  6.1400 -0.7090 0.0
H -2.1700  1.2530 0.0
H -2.1700 -1.2530 0.0
H  0.0000  2.5060 0.0
H  0.0000 -2.5060 0.0
H  2.4560  2.5060 0.0
H  2.4560 -2.5060 0.0
H  4.9120  2.5060 0.0
H  4.9120 -2.5060 0.0
H  7.0820  1.2530 0.0
H  7.0820 -1.2530 0.0
"""

TETRACENE = """
C -1.2280  0.7090 0.0
C -1.2280 -0.7090 0.0
C  0.0000  1.4180 0.0
C  0.0000 -1.4180 0.0
C  1.2280  0.7090 0.0
C  1.2280 -0.7090 0.0
C  2.4560  1.4180 0.0
C  2.4560 -1.4180 0.0
C  3.6840  0.7090 0.0
C  3.6840 -0.7090 0.0
C  4.9120  1.4180 0.0
C  4.9120 -1.4180 0.0
C  6.1400  0.7090 0.0
C  6.1400 -0.7090 0.0
C  7.3680  1.4180 0.0
C  7.3680 -1.4180 0.0
C  8.5960  0.7090 0.0
C  8.5960 -0.7090 0.0
H -2.1700  1.2530 0.0
H -2.1700 -1.2530 0.0
H  0.0000  2.5060 0.0
H  0.0000 -2.5060 0.0
H  2.4560  2.5060 0.0
H  2.4560 -2.5060 0.0
H  4.9120  2.5060 0.0
H  4.9120 -2.5060 0.0
H  7.3680  2.5060 0.0
H  7.3680 -2.5060 0.0
H  9.5380  1.2530 0.0
H  9.5380 -1.2530 0.0
"""


class BenchmarkSpec(NamedTuple):
    name: str
    atoms: str


class BenchmarkCase(NamedTuple):
    mol: gto.Mole
    grids: dft.Grids
    dm: np.ndarray
    numint: SkalaNumInt[np.ndarray]


@pytest.fixture(scope="module")
def fixed_cpu_threads() -> Iterator[None]:
    previous_pyscf_threads = lib.num_threads()
    previous_torch_threads = torch.get_num_threads()
    lib.num_threads(THREAD_COUNT)
    torch.set_num_threads(THREAD_COUNT)
    try:
        yield
    finally:
        torch.set_num_threads(previous_torch_threads)
        lib.num_threads(previous_pyscf_threads)


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(BenchmarkSpec("naphthalene", NAPHTHALENE), id="naphthalene"),
        pytest.param(BenchmarkSpec("anthracene", ANTHRACENE), id="anthracene"),
        pytest.param(BenchmarkSpec("tetracene", TETRACENE), id="tetracene"),
    ],
)
def benchmark_case(
    request: pytest.FixtureRequest,
    fixed_cpu_threads: None,
    load_functional_cached: Callable[..., ExcFunctionalBase | str],
) -> BenchmarkCase:
    spec = cast(BenchmarkSpec, request.param)
    mol = gto.M(atom=spec.atoms, basis="def2-qzvpp", verbose=0)
    grids = dft.Grids(mol)
    grids.level = 1
    grids.build(sort_grids=False)
    dm = dft.RKS(mol).get_init_guess()
    functional = load_functional_cached("skala-1.1")
    assert isinstance(functional, ExcFunctionalBase)
    return BenchmarkCase(mol, grids, dm, SkalaNumInt(functional))


@pytest.fixture
def screened_case(benchmark_case: BenchmarkCase) -> BenchmarkCase:
    assert _should_screen_aos(benchmark_case.mol)
    return benchmark_case


@pytest.fixture
def dense_case(
    benchmark_case: BenchmarkCase, monkeypatch: pytest.MonkeyPatch
) -> BenchmarkCase:
    monkeypatch.setattr(pyscf_numint, "SWITCH_SIZE", benchmark_case.mol.nao_nr())
    assert not _should_screen_aos(benchmark_case.mol)
    return benchmark_case


def _run_xc(case: BenchmarkCase) -> tuple[float, float, np.ndarray]:
    return case.numint.nr_rks(
        case.mol,
        case.grids,
        None,
        case.dm,
        max_memory=MAX_MEMORY_MB,
    )


def _benchmark_xc(benchmark: BenchmarkFixture, case: BenchmarkCase) -> None:
    pedantic = cast(Callable[..., object], benchmark.pedantic)
    pedantic(
        _run_xc,
        args=(case,),
        rounds=1,
        iterations=2,
    )


@pytest.mark.profiling
def test_screened_and_dense_values_agree(
    benchmark_case: BenchmarkCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _should_screen_aos(benchmark_case.mol)
    screened = _run_xc(benchmark_case)

    monkeypatch.setattr(pyscf_numint, "SWITCH_SIZE", benchmark_case.mol.nao_nr())
    assert not _should_screen_aos(benchmark_case.mol)
    dense = _run_xc(benchmark_case)

    assert np.allclose(dense[0], screened[0], rtol=1e-10, atol=1e-11)
    assert np.isclose(dense[1], screened[1], rtol=1e-10, atol=1e-11)
    vxc_difference = dense[2] - screened[2]
    vxc_max_abs_difference = np.max(np.abs(vxc_difference))
    vxc_relative_l2_difference = np.linalg.norm(vxc_difference) / np.linalg.norm(
        dense[2]
    )
    assert vxc_max_abs_difference < 5e-8 and vxc_relative_l2_difference < 1e-8, (
        f"N: dense={dense[0]:.16g}, screened={screened[0]:.16g}, "
        f"abs_diff={abs(dense[0] - screened[0]):.3e}; "
        f"E_xc: dense={dense[1]:.16g}, screened={screened[1]:.16g}, "
        f"abs_diff={abs(dense[1] - screened[1]):.3e}; "
        f"V_xc: max_abs_diff={vxc_max_abs_difference:.3e}, "
        f"relative_l2_diff={vxc_relative_l2_difference:.3e}"
    )


@pytest.mark.benchmark(group="def2-qzvpp")
def test_with_natural_ao_screening(
    benchmark: BenchmarkFixture, screened_case: BenchmarkCase
) -> None:
    _benchmark_xc(benchmark, screened_case)


@pytest.mark.benchmark(group="def2-qzvpp")
def test_without_ao_screening_by_patching_threshold(
    benchmark: BenchmarkFixture, dense_case: BenchmarkCase
) -> None:
    _benchmark_xc(benchmark, dense_case)


@pytest.mark.profiling
def test_profile_with_natural_ao_screening(
    screened_case: BenchmarkCase,
) -> None:
    _run_xc(screened_case)


@pytest.mark.profiling
def test_profile_without_ao_screening_by_patching_threshold(
    dense_case: BenchmarkCase,
) -> None:
    _run_xc(dense_case)
