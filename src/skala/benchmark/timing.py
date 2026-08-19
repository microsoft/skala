# SPDX-License-Identifier: MIT

"""Direct, per-cycle timing of an SCF calculation.

Four nested layers are measured without sampling::

    xc_eval  in  numint  in  veff  in  cycle  in  kernel

``xc_eval`` is the functional evaluated to first order -- ``eval_xc_eff`` for
libxc, or ``get_exc`` plus ``torch.autograd.grad`` for a neural functional.
``numint`` is the whole exchange-correlation quadrature (feature construction,
the functional itself, and Vxc assembly), ``veff`` adds the J/K build, and
``cycle`` is one SCF iteration.

On CUDA the layers cannot be timed with a host clock. ``get_exc`` returns once
its kernels are *queued*, and the backward pass is queued immediately after with
no synchronization in between, so a host timer measures launch overhead rather
than work; the error grows with chunk size. :class:`Timeline` therefore takes
durations from CUDA events, which are ordered within the stream that executes the
work. Marks are still *created* in host program order, so attributing an interval
to a cycle stays a host-side decision; only the durations come from the device.
"""

from __future__ import annotations

import itertools
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import torch
    from pyscf.scf.hf import SCF

    from skala.functional.base import ExcFunctionalBase

#: Cycles from this index on are treated as steady state. Cycle 0 is excluded so
#: that anything specific to a first iteration cannot distort a scaling fit;
#: :func:`skala.benchmark.metrics.warmup_ratio` reports how much that matters.
STEADY_STATE_FROM_CYCLE = 1


class Mark(Protocol):
    """An opaque point in time on either the host or the device timeline."""


@dataclass(frozen=True, slots=True)
class CycleTiming:
    """Directly measured costs of one SCF iteration, in milliseconds."""

    cycle: int
    wall_ms: float
    veff_ms: float
    numint_ms: float
    #: The functional evaluated to first order: ``forward + backward`` for a
    #: neural functional, libxc's combined energy-and-derivative call otherwise.
    xc_eval_ms: float
    #: Neural functional only; zero for a classical one, which has no separable
    #: forward and backward.
    forward_ms: float
    backward_ms: float
    veff_calls: int
    numint_calls: int
    xc_eval_calls: int


@dataclass(frozen=True, slots=True)
class ScfTiming:
    """Per-cycle timings for one SCF calculation.

    The phases partition the instrumented span exactly::

        total_ms == setup_ms + sum(cycle.wall_ms) + finalize_ms
    """

    #: The whole instrumented span, on the device timeline for GPU runs.
    total_ms: float
    #: Kernel entry until the SCF loop starts: grid construction, one-electron
    #: integrals, the initial guess, and the Fock build on that guess. That
    #: build is a full effective-potential evaluation, so setup contains one
    #: exchange-correlation quadrature in addition to the one-time work.
    setup_ms: float
    #: Work after the last cycle, chiefly pyscf's post-loop convergence check.
    finalize_ms: float
    cycles: tuple[CycleTiming, ...]

    def steady_state(self) -> tuple[CycleTiming, ...]:
        """Cycles excluding the warmup-dominated first one.

        Falls back to all cycles when the calculation converged too quickly for
        a steady state to exist.
        """
        tail = self.cycles[STEADY_STATE_FROM_CYCLE:]
        return tail or self.cycles


class Timeline(ABC):
    """Records marks and measures the intervals between them.

    Durations are only valid after :meth:`resolve`, which blocks until any
    pending device work referenced by a mark has finished.
    """

    @abstractmethod
    def mark(self) -> Mark:
        """Return a mark for the current point in the timeline."""
        raise NotImplementedError

    def resolve(self) -> None:
        """Block until every recorded mark has a readable timestamp."""
        return None

    @abstractmethod
    def elapsed_ms(self, start: Mark, end: Mark) -> float:
        """Return the milliseconds between two resolved marks."""
        raise NotImplementedError


class HostTimeline(Timeline):
    """Host-clock timeline, exact when all work is synchronous (CPU runs)."""

    def mark(self) -> Mark:
        return time.perf_counter()

    def elapsed_ms(self, start: Mark, end: Mark) -> float:
        return 1e3 * (float(end) - float(start))  # type: ignore[arg-type]


class CudaTimeline(Timeline):
    """Device timeline built from CUDA events recorded on the current stream."""

    def __init__(self) -> None:
        import torch

        self._torch = torch

    def mark(self) -> Mark:
        event = self._torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        event.record()  # type: ignore[no-untyped-call]
        return event

    def resolve(self) -> None:
        self._torch.cuda.synchronize()

    def elapsed_ms(self, start: Mark, end: Mark) -> float:
        return float(start.elapsed_time(end))  # type: ignore[attr-defined]


def make_timeline(device: torch.device | str) -> Timeline:
    """Return the timeline appropriate for the device the run executes on."""
    device_type = device if isinstance(device, str) else device.type
    return CudaTimeline() if device_type == "cuda" else HostTimeline()


@dataclass
class _Interval:
    """One timed region, tagged with the cycle it was recorded in.

    ``opened`` and ``closed`` are positions in a counter shared by every layer.
    Because the layers nest, they let a region be recognized as lying inside
    another without comparing marks, which is not possible for CUDA events.
    """

    cycle: int
    start: Mark
    opened: int
    end: Mark | None = None
    closed: int | None = None

    def contains(self, other: _Interval) -> bool:
        """Return whether ``other`` was recorded entirely within this region."""
        return self.closed is not None and self.opened < other.opened < self.closed


class _Accumulator:
    """Collects intervals for one instrumented layer."""

    def __init__(self, timeline: Timeline, sequence: Callable[[], int]) -> None:
        self._timeline = timeline
        self._sequence = sequence
        self._intervals: list[_Interval] = []

    @contextmanager
    def measure(self, cycle: int) -> Iterator[None]:
        interval = _Interval(
            cycle=cycle, start=self._timeline.mark(), opened=self._sequence()
        )
        self._intervals.append(interval)
        try:
            yield
        finally:
            interval.end = self._timeline.mark()
            interval.closed = self._sequence()

    def first_start(self) -> Mark | None:
        return self._intervals[0].start if self._intervals else None

    def initial_build(self) -> _Interval | None:
        """Return the pre-loop interval, if this layer recorded one.

        Both pyscf and gpu4pyscf evaluate the exchange-correlation potential once
        on the initial guess *before* entering the SCF loop, to build the starting
        Fock matrix. That call is recorded against cycle 0 because the callback
        closing cycle 0 has not fired yet, which would make the first iteration
        look twice as expensive as the rest.

        It is identified structurally rather than by position: a pre-loop build
        exists only when cycle 0 recorded more calls than one iteration needs. A
        closed interval is required, so a run that died inside the initial build
        keeps that time in the cycle rather than silently losing it.
        """
        in_first_cycle = [item for item in self._intervals if item.cycle == 0]
        if len(in_first_cycle) < 2:
            return None
        candidate = in_first_cycle[0]
        return candidate if candidate.end is not None else None

    def totals(self, outside: _Interval | None = None) -> dict[int, tuple[float, int]]:
        """Return ``{cycle: (milliseconds, call count)}`` for closed intervals.

        Regions inside ``outside`` (and ``outside`` itself) are left out, which
        drops the pre-loop initial-guess build from the first cycle along with
        any nested work it performed.
        """
        totals: dict[int, tuple[float, int]] = {}
        for interval in self._intervals:
            if interval.end is None:  # the run died inside this region
                continue
            if outside is not None and (
                interval is outside or outside.contains(interval)
            ):
                continue
            elapsed = self._timeline.elapsed_ms(interval.start, interval.end)
            total, count = totals.get(interval.cycle, (0.0, 0))
            totals[interval.cycle] = (total + elapsed, count + 1)
        return totals


class ScfInstrumentation:
    """Instruments a mean-field object to record per-cycle timings.

    The mean-field object is modified in place, so use :func:`instrument` rather
    than constructing this directly.
    """

    def __init__(self, timeline: Timeline) -> None:
        self._timeline = timeline
        self._counter = itertools.count()
        self._veff = _Accumulator(timeline, lambda: next(self._counter))
        self._numint = _Accumulator(timeline, lambda: next(self._counter))
        self._forward = _Accumulator(timeline, lambda: next(self._counter))
        self._backward = _Accumulator(timeline, lambda: next(self._counter))
        self._cycle = 0
        self._neural = False
        self._start: Mark | None = None
        self._end: Mark | None = None
        self._boundaries: list[tuple[int, Mark]] = []

    def _on_cycle_end(self, envs: Mapping[str, Any]) -> None:
        """Close the current cycle. Installed as ``mf.callback``."""
        cycle = envs.get("cycle")
        self._boundaries.append(
            (cycle if isinstance(cycle, int) else self._cycle, self._timeline.mark())
        )
        self._cycle += 1

    def result(self) -> ScfTiming:
        """Resolve every mark and assemble the per-cycle timings.

        The first cycle starts where the SCF loop does: after grid construction,
        one-electron integrals, the initial guess, and the initial-guess Fock
        build. All of those are charged to setup, so every reported cycle
        contains the same work and cycle 0 is comparable to the rest.
        """
        self._timeline.resolve()
        initial_build = self._veff.initial_build()
        veff = self._veff.totals(outside=initial_build)
        numint = self._numint.totals(outside=initial_build)
        forward = self._forward.totals(outside=initial_build)
        backward = self._backward.totals(outside=initial_build)

        start, end = self._start, self._end
        if start is None or end is None:
            return ScfTiming(0.0, 0.0, 0.0, ())
        total_ms = self._timeline.elapsed_ms(start, end)

        if initial_build is not None:
            cycles_begin = initial_build.end
        else:
            first_veff = self._veff.first_start()
            cycles_begin = first_veff if first_veff is not None else start
        setup_ms = self._timeline.elapsed_ms(start, cycles_begin)

        cycles: list[CycleTiming] = []
        previous = cycles_begin
        for cycle, boundary in self._boundaries:
            veff_ms, veff_calls = veff.get(cycle, (0.0, 0))
            numint_ms, numint_calls = numint.get(cycle, (0.0, 0))
            forward_ms, forward_calls = forward.get(cycle, (0.0, 0))
            backward_ms, _ = backward.get(cycle, (0.0, 0))
            cycles.append(
                CycleTiming(
                    cycle=cycle,
                    wall_ms=self._timeline.elapsed_ms(previous, boundary),
                    veff_ms=veff_ms,
                    numint_ms=numint_ms,
                    xc_eval_ms=forward_ms + backward_ms,
                    forward_ms=forward_ms if self._neural else 0.0,
                    backward_ms=backward_ms,
                    veff_calls=veff_calls,
                    numint_calls=numint_calls,
                    xc_eval_calls=forward_calls,
                )
            )
            previous = boundary
        return ScfTiming(
            total_ms=total_ms,
            setup_ms=setup_ms,
            finalize_ms=self._timeline.elapsed_ms(previous, end),
            cycles=tuple(cycles),
        )


@contextmanager
def instrument(
    mf: SCF,
    *,
    device: torch.device | str,
    functional: ExcFunctionalBase | str | None = None,
) -> Iterator[ScfInstrumentation]:
    """Instrument ``mf`` in place for the duration of the context.

    Wraps the effective-potential and numerical-integration entry points and,
    for a neural functional, its ``get_exc``. Every patch is applied to the
    instance (or, for the functional, bound on the object we constructed), so
    nothing leaks into the pyscf, gpu4pyscf, or skala modules themselves.

    Args:
        mf: The mean-field object whose ``kernel()`` is about to run.
        device: The device the calculation runs on; selects the timeline.
        functional: The neural functional to time the forward pass of. Native
            (string) functionals have no network and are skipped.

    Yields:
        The instrumentation, whose :meth:`~ScfInstrumentation.result` is valid
        once the kernel has returned.
    """
    instrumentation = ScfInstrumentation(make_timeline(device))
    numint: Any = getattr(mf, "_numint", None)
    restore: list[tuple[object, str, Any]] = []

    original_veff = getattr(mf, "get_veff", None)
    if callable(original_veff):
        restore.append((mf, "get_veff", original_veff))
        mf.get_veff = _timed(original_veff, instrumentation, instrumentation._veff)

    for name in ("nr_uks", "nr_rks"):
        original = getattr(numint, name, None)
        if callable(original):
            restore.append((numint, name, original))
            setattr(
                numint,
                name,
                _timed(original, instrumentation, instrumentation._numint),
            )

    if functional is not None and not isinstance(functional, str):
        instrumentation._neural = True
        original_get_exc = functional.get_exc
        restore.append((functional, "get_exc", original_get_exc))
        functional.get_exc = _timed(  # type: ignore[method-assign]
            original_get_exc, instrumentation, instrumentation._forward
        )
        # The network's derivative is a separate call, so the two halves of the
        # functional evaluation must be timed separately and added. Patching the
        # module attribute is restored on exit, and the worker runs one
        # calculation at a time, so nothing else is caught by it.
        import torch

        original_grad = torch.autograd.grad
        restore.append((torch.autograd, "grad", original_grad))
        torch.autograd.grad = _timed(
            original_grad, instrumentation, instrumentation._backward
        )
    else:
        # The classical counterpart of ``get_exc``: pyscf and gpu4pyscf both call
        # it from inside the quadrature, once per grid block, to turn the density
        # into an energy density. Timing it makes the neural and classical
        # functionals comparable at the same layer.
        original_eval_xc = getattr(numint, "eval_xc_eff", None)
        if callable(original_eval_xc):
            restore.append((numint, "eval_xc_eff", original_eval_xc))
            numint.eval_xc_eff = _timed(
                original_eval_xc, instrumentation, instrumentation._forward
            )

    previous_callback = mf.callback
    mf.callback = instrumentation._on_cycle_end
    instrumentation._start = instrumentation._timeline.mark()
    try:
        yield instrumentation
    finally:
        instrumentation._end = instrumentation._timeline.mark()
        mf.callback = previous_callback
        for owner, name, original in restore:
            setattr(owner, name, original)


def _timed(
    function: Callable[..., Any],
    instrumentation: ScfInstrumentation,
    accumulator: _Accumulator,
) -> Callable[..., Any]:
    """Wrap ``function`` so each call is charged to the current cycle."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with accumulator.measure(instrumentation._cycle):
            return function(*args, **kwargs)

    return wrapper
