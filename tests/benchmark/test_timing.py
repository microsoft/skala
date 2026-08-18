# SPDX-License-Identifier: MIT

"""Tests for direct per-cycle SCF timing.

A deterministic clock is substituted for the real timeline so the assertions are
about the accounting, not about how long anything actually took.
"""

import pytest

from skala.benchmark import timing
from skala.benchmark.timing import (
    HostTimeline,
    ScfInstrumentation,
    Timeline,
    instrument,
    make_timeline,
)

#: Cost charged by the stand-in ``torch.autograd.grad``.
BACKWARD_COST = 5.0


class FakeTimeline(Timeline):
    """A clock that advances only when a test says so."""

    def __init__(self) -> None:
        self.now = 0.0
        self.resolved = 0

    def advance(self, milliseconds: float) -> None:
        self.now += milliseconds

    def mark(self) -> float:
        return self.now

    def resolve(self) -> None:
        self.resolved += 1

    def elapsed_ms(self, start: float, end: float) -> float:  # type: ignore[override]
        return float(end) - float(start)


class FakeFunctional:
    """Stands in for a neural functional: a forward pass and a backward pass."""

    def __init__(self, clock: FakeTimeline, cost: float) -> None:
        self._clock = clock
        self._cost = cost
        self.calls = 0

    def get_exc(self, features: object = None) -> str:
        self.calls += 1
        self._clock.advance(self._cost)
        return "exc"

    def backward(self) -> None:
        """Stands in for ``torch.autograd.grad``, which ``instrument`` patches."""
        import torch

        torch.autograd.grad(None, None)


class FakeNumInt:
    """Numerical integration that spends part of its time in the network."""

    def __init__(
        self, clock: FakeTimeline, around: float, functional: FakeFunctional | None
    ) -> None:
        self._clock = clock
        self._around = around
        self._functional = functional
        self.calls = 0

    def nr_uks(self, *args: object, **kwargs: object) -> str:
        self.calls += 1
        self._clock.advance(self._around)
        if self._functional is not None:
            self._functional.get_exc()
            self._functional.backward()
        self._clock.advance(self._around)
        return "vxc"

    def nr_rks(self, *args: object, **kwargs: object) -> str:
        return self.nr_uks(*args, **kwargs)


class FakeMeanField:
    """A mean field whose get_veff wraps numint, as pyscf's DFT path does."""

    callback = None

    def __init__(
        self, numint: FakeNumInt, clock: FakeTimeline, jk: float = 0.0
    ) -> None:
        self._numint = numint
        self._clock = clock
        self._jk = jk

    def get_veff(self, *args: object, **kwargs: object) -> str:
        self._clock.advance(self._jk)
        return self._numint.nr_uks()


@pytest.fixture(autouse=True)
def fake_autograd(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stand-in for torch.autograd.grad that costs a known amount of time."""
    torch = pytest.importorskip("torch")
    holder: dict[str, FakeTimeline] = {}

    def grad(*args: object, **kwargs: object) -> tuple:
        clock = holder.get("clock")
        if clock is not None:
            clock.advance(BACKWARD_COST)
        return ()

    monkeypatch.setattr(torch.autograd, "grad", grad)
    FakeTimeline._holder = holder  # type: ignore[attr-defined]


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeTimeline:
    """Install a deterministic clock behind :func:`instrument`."""
    fake = FakeTimeline()
    getattr(FakeTimeline, "_holder", {})["clock"] = fake
    monkeypatch.setattr(timing, "make_timeline", lambda device: fake)
    return fake


def build(
    clock: FakeTimeline,
    *,
    forward_cost: float = 3.0,
    around: float = 2.0,
    jk: float = 0.0,
):
    """Numint costs ``2*around + forward_cost + BACKWARD_COST`` per call."""
    functional = FakeFunctional(clock, cost=forward_cost)
    numint = FakeNumInt(clock, around, functional)
    return FakeMeanField(numint, clock, jk=jk), functional


def drive(
    clock: FakeTimeline,
    mf: FakeMeanField,
    measurement,
    *,
    setup: float,
    cycles: int,
    other_per_cycle: float,
    finalize: float,
    initial_build: bool = True,
) -> None:
    """Play out one SCF the way pyscf does.

    ``initial_build`` emits the pre-loop effective-potential build that pyscf and
    gpu4pyscf both perform on the initial guess before entering the SCF loop.
    """
    clock.advance(setup)
    if initial_build:
        mf.get_veff()
    for cycle in range(cycles):
        mf.get_veff()
        clock.advance(other_per_cycle)
        measurement._on_cycle_end({"cycle": cycle})
    clock.advance(finalize)


def test_make_timeline_selects_the_host_clock_for_cpu() -> None:
    assert isinstance(make_timeline("cpu"), HostTimeline)


def test_host_timeline_measures_non_negative_intervals() -> None:
    timeline = HostTimeline()
    start = timeline.mark()
    end = timeline.mark()
    timeline.resolve()
    assert timeline.elapsed_ms(start, end) >= 0.0


def test_cuda_timeline_is_selected_for_cuda_devices() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")
    assert isinstance(make_timeline("cuda"), timing.CudaTimeline)


def test_phases_partition_the_kernel_exactly(clock: FakeTimeline) -> None:
    mf, functional = build(clock)
    with instrument(mf, device="cpu", functional=functional) as measurement:
        drive(
            clock, mf, measurement,
            setup=5.0, cycles=3, other_per_cycle=2.0, finalize=7.0,
        )  # fmt: skip
    result = measurement.result()
    cycles_ms = sum(cycle.wall_ms for cycle in result.cycles)
    assert result.setup_ms + cycles_ms + result.finalize_ms == pytest.approx(
        result.total_ms
    )


def test_the_pre_loop_fock_build_is_charged_to_setup(clock: FakeTimeline) -> None:
    """pyscf builds the initial-guess Fock matrix before the SCF loop starts.

    That build is expensive (it carries the one-time density-fitting integral
    build) and happens before the first callback, so charging it to cycle 0 would
    make the first iteration look about twice as costly as the rest.
    """
    mf, functional = build(clock, jk=10.0)
    with instrument(mf, device="cpu", functional=functional) as measurement:
        drive(
            clock, mf, measurement,
            setup=5.0, cycles=2, other_per_cycle=2.0, finalize=0.0,
        )  # fmt: skip
    result = measurement.result()
    # 5 ms of setup plus the whole 22 ms pre-loop build (10 J/K + 12 numint).
    assert result.setup_ms == pytest.approx(27.0)
    # Every cycle now contains exactly one build, so cycle 0 matches the rest.
    assert result.cycles[0].wall_ms == pytest.approx(24.0)
    assert result.cycles[1].wall_ms == pytest.approx(24.0)
    assert [c.veff_calls for c in result.cycles] == [1, 1]
    assert [c.numint_calls for c in result.cycles] == [1, 1]
    assert [c.xc_eval_calls for c in result.cycles] == [1, 1]


def test_a_run_without_a_pre_loop_build_keeps_its_first_cycle(
    clock: FakeTimeline,
) -> None:
    """The pre-loop build is detected structurally, not assumed."""
    mf, functional = build(clock, jk=10.0)
    with instrument(mf, device="cpu", functional=functional) as measurement:
        drive(
            clock, mf, measurement,
            setup=5.0, cycles=2, other_per_cycle=2.0, finalize=0.0,
            initial_build=False,
        )  # fmt: skip
    result = measurement.result()
    assert result.setup_ms == pytest.approx(5.0)
    assert result.cycles[0].wall_ms == pytest.approx(24.0)
    assert [c.veff_calls for c in result.cycles] == [1, 1]


def test_layers_nest_within_each_cycle(clock: FakeTimeline) -> None:
    mf, functional = build(clock, forward_cost=3.0, around=2.0, jk=10.0)
    with instrument(mf, device="cpu", functional=functional) as measurement:
        drive(
            clock, mf, measurement,
            setup=1.0, cycles=3, other_per_cycle=2.0, finalize=1.0,
        )  # fmt: skip
    for cycle in measurement.result().cycles:
        assert cycle.xc_eval_ms <= cycle.numint_ms <= cycle.veff_ms <= cycle.wall_ms
        assert cycle.forward_ms == pytest.approx(3.0)
        assert cycle.backward_ms == pytest.approx(BACKWARD_COST)
        # The functional evaluation is the forward and the backward together.
        assert cycle.xc_eval_ms == pytest.approx(3.0 + BACKWARD_COST)
        assert cycle.numint_ms == pytest.approx(12.0)
        assert cycle.veff_ms == pytest.approx(22.0)
        assert cycle.wall_ms == pytest.approx(24.0)
        assert cycle.veff_calls == 1
        assert cycle.numint_calls == 1
        assert cycle.xc_eval_calls == 1


def test_xc_eval_is_summed_over_chunks(clock: FakeTimeline) -> None:
    """Skala evaluates the grid in chunks; every chunk must be charged."""
    functional = FakeFunctional(clock, cost=3.0)
    numint = FakeNumInt(clock, around=1.0, functional=None)

    def chunked(*args: object, **kwargs: object) -> str:
        for _ in range(4):
            functional.get_exc()
            functional.backward()
        return "vxc"

    numint.nr_uks = chunked  # type: ignore[method-assign]
    mf = FakeMeanField(numint, clock)
    with instrument(mf, device="cpu", functional=functional) as measurement:
        drive(
            clock, mf, measurement,
            setup=0.0, cycles=1, other_per_cycle=0.0, finalize=0.0,
        )  # fmt: skip
    cycle = measurement.result().cycles[0]
    assert cycle.xc_eval_calls == 4
    assert cycle.forward_ms == pytest.approx(12.0)
    assert cycle.backward_ms == pytest.approx(4 * BACKWARD_COST)
    assert cycle.xc_eval_ms == pytest.approx(12.0 + 4 * BACKWARD_COST)


def test_work_after_the_last_cycle_is_finalize(clock: FakeTimeline) -> None:
    mf, functional = build(clock)
    with instrument(mf, device="cpu", functional=functional) as measurement:
        drive(
            clock, mf, measurement,
            setup=0.0, cycles=1, other_per_cycle=0.0, finalize=42.0,
        )  # fmt: skip
    result = measurement.result()
    assert len(result.cycles) == 1
    assert result.finalize_ms == pytest.approx(42.0)


def test_steady_state_drops_the_first_cycle(clock: FakeTimeline) -> None:
    mf, functional = build(clock)
    with instrument(mf, device="cpu", functional=functional) as measurement:
        drive(
            clock, mf, measurement,
            setup=0.0, cycles=4, other_per_cycle=1.0, finalize=0.0,
        )  # fmt: skip
    assert [cycle.cycle for cycle in measurement.result().steady_state()] == [1, 2, 3]


def test_steady_state_falls_back_for_a_single_cycle(clock: FakeTimeline) -> None:
    mf, functional = build(clock)
    with instrument(mf, device="cpu", functional=functional) as measurement:
        drive(
            clock, mf, measurement,
            setup=0.0, cycles=1, other_per_cycle=0.0, finalize=0.0,
        )  # fmt: skip
    assert [cycle.cycle for cycle in measurement.result().steady_state()] == [0]


def test_result_resolves_the_timeline(clock: FakeTimeline) -> None:
    mf, functional = build(clock)
    with instrument(mf, device="cpu", functional=functional) as measurement:
        drive(
            clock, mf, measurement,
            setup=0.0, cycles=1, other_per_cycle=0.0, finalize=0.0,
        )  # fmt: skip
    measurement.result()
    assert clock.resolved == 1


def test_instrument_restores_every_patch(clock: FakeTimeline) -> None:
    mf, functional = build(clock)
    numint = mf._numint
    original_uks, original_rks = numint.nr_uks, numint.nr_rks
    original_get_exc = functional.get_exc

    with instrument(mf, device="cpu", functional=functional):
        assert numint.nr_uks is not original_uks
        assert functional.get_exc is not original_get_exc

    assert numint.nr_uks == original_uks
    assert numint.nr_rks == original_rks
    assert functional.get_exc == original_get_exc
    assert mf.callback is None


def test_classical_functionals_have_no_separable_forward(clock: FakeTimeline) -> None:
    mf = FakeMeanField(FakeNumInt(clock, around=2.0, functional=None), clock)
    # A native functional is a plain string with no network to time.
    with instrument(mf, device="cpu", functional="b3lyp5") as measurement:
        drive(
            clock, mf, measurement,
            setup=0.0, cycles=1, other_per_cycle=0.0, finalize=0.0,
        )  # fmt: skip
    cycle = measurement.result().cycles[0]
    # libxc returns energy and derivative together, so there is no split.
    assert cycle.forward_ms == 0.0
    assert cycle.backward_ms == 0.0
    assert cycle.numint_ms == pytest.approx(4.0)


def test_a_run_with_no_cycles_reports_empty(clock: FakeTimeline) -> None:
    mf, functional = build(clock)
    with instrument(mf, device="cpu", functional=functional) as measurement:
        clock.advance(11.0)  # died before completing an iteration
    result = measurement.result()
    assert result.cycles == ()
    assert result.total_ms == pytest.approx(11.0)


def test_an_unclosed_region_is_ignored(clock: FakeTimeline) -> None:
    """A worker killed inside numint must not produce a nonsensical duration."""
    instrumentation = ScfInstrumentation(clock)
    instrumentation._start = clock.mark()
    instrumentation._numint._intervals.append(
        timing._Interval(cycle=0, start=clock.mark(), opened=0)
    )
    clock.advance(5.0)
    instrumentation._on_cycle_end({"cycle": 0})
    instrumentation._end = clock.mark()
    cycle = instrumentation.result().cycles[0]
    assert cycle.numint_ms == 0.0
    assert cycle.numint_calls == 0
