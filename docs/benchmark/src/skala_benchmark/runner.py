# SPDX-License-Identifier: MIT

"""Run one benchmark calculation in an isolated worker process."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from skala_benchmark.models import Molecule
from skala_benchmark.protocol import Device, FunctionalKind, FunctionalSpec
from skala_benchmark.timing import instrument

#: When this module finished loading, on the same clock the orchestrator uses.
#: ``time.perf_counter`` is ``CLOCK_MONOTONIC`` on Linux, which counts from boot
#: and is therefore comparable across processes on one machine. Comparing this
#: against the parent's timestamp just before ``Popen`` measures process
#: creation, interpreter start-up, and this module's imports -- the part a
#: worker cannot time from the inside because it has not started yet.
_MODULE_READY_AT = time.perf_counter()

#: The system stage 1 warms the process on. Small enough that the calculation
#: itself is negligible (about 330 ms of a 13 s stage on an A100), but a real
#: SCF, so it exercises the grid, J/K, density-fitting and quadrature paths.
_WARMUP_MOLECULE = Molecule(
    atomic_numbers=[1, 1], geometry_bohr=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]
)
_WARMUP_BASIS = "def2-svp"

if TYPE_CHECKING:
    import torch
    from pyscf import gto
    from pyscf.scf.hf import SCF
    from skala.functional.base import ExcFunctionalBase


@dataclass(frozen=True, slots=True)
class RunConfig:
    """One point in the ``(basis, functional, molecule)`` benchmark grid."""

    molecule: Molecule
    basis: str
    functional: FunctionalSpec
    device: Device
    ansatz: str = "UKS"  # "RKS" | "UKS"; benchmark runs unrestricted by default
    density_fit: bool = True
    auxbasis: str | None = None
    grid_level: int = 3
    conv_tol: float = 5e-6  # Skala's loose-precision default
    conv_tol_grad: float | None = None  # None => PySCF default sqrt(conv_tol)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable worker configuration."""
        return {
            **dataclasses.asdict(self),
            "functional": {
                "name": self.functional.name,
                "kind": self.functional.kind.value,
            },
            "device": self.device.value,
        }

    @classmethod
    def from_json(cls, path: str | Path) -> RunConfig:
        with Path(path).open(encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(
            **{
                **data,
                "molecule": Molecule(**data["molecule"]),
                "functional": FunctionalSpec(
                    name=data["functional"]["name"],
                    kind=FunctionalKind(data["functional"]["kind"]),
                ),
                "device": Device(data["device"]),
            }
        )


@dataclass(frozen=True, slots=True)
class RunResult:
    """DFT result + timing for one successful :class:`RunConfig`."""

    total_energy: float
    is_converged: bool
    num_scf_iterations: int
    num_atoms: int
    num_electrons: int
    num_atomic_orbitals: int
    num_aux_basis_functions: (
        int | None
    )  # density-fitting aux basis size (None if no DF)
    grid_size: int
    conv_tol_grad: float  # effective value actually used
    device: str  # "cpu" | "gpu" -- whichever the run actually used
    #: Everything the worker timed: load + warmup + build + kernel. The
    #: difference from the orchestrator's ``process_wall_ms`` is interpreter
    #: startup, imports, and teardown.
    worker_ms: float
    #: Process creation, interpreter start-up, and importing this module.
    #: ``None`` when the worker was started without ``--launched-at``.
    boot_ms: float | None
    load_ms: float  # loading the functional (model weights onto the device)
    #: The three warmup stages. ``warmup_ms`` is their sum, so the worker phases
    #: still partition ``worker_ms``.
    warmup_ms: float
    #: A converged SCF on H2/def2-svp: CUDA context, CuPy kernel compilation,
    #: and every other cost that is paid once per process rather than per
    #: system. Only a few hundred milliseconds of it is the calculation.
    process_warmup_ms: float
    #: A single-cycle SCF on the real system: grids, sorted-orbital tables and
    #: density-fitting integrals, which are keyed on the molecule.
    target_warmup_ms: float
    #: Repeated exchange-correlation evaluations until the cost settles.
    settle_ms: float
    #: What those evaluations cost above their own steady state, which is where
    #: TorchScript compiles for this system's shapes. Understates a cold compile,
    #: because stage 1 already compiled the parts that do not depend on shape.
    jit_compile_ms: float
    #: One duration per stage-2 evaluation, in order. The shape of the ramp is
    #: what identifies the compile: the cost falls, spikes where TorchScript
    #: builds the optimized graph, then flattens.
    settle_evaluations_ms: list[float]
    build_ms: float  # measured molecule + mean-field construction
    wall_time_ms: float  # build_ms + kernel_time_ms
    kernel_time_ms: float  # mf.kernel() call only
    setup_ms: float  # kernel entry until the first numerical-integration call
    finalize_ms: float  # after the last cycle: pyscf's post-loop convergence check
    #: One entry per SCF iteration; see :class:`skala_benchmark.timing.
    #: CycleTiming`. Cycle 0 carries the one-time warmup cost.
    cycles: list[dict[str, float | int]]


@dataclass(frozen=True, slots=True)
class RunError:
    """A :class:`RunConfig` that failed to run."""

    error: str


#: Outcome of a single run: either a result or an error.
RunOutcome = RunResult | RunError


def run_worker(config: RunConfig, launched_at: float | None = None) -> RunOutcome:
    """Run one instrumented SCF calculation."""
    try:
        return _run_scf(config, launched_at)
    finally:
        _release_gpu4pyscf_global_cache(config.device)


def _run_scf(config: RunConfig, launched_at: float | None = None) -> RunOutcome:
    """Warm first-use paths, then run and report one instrumented SCF.

    Thread counts are taken from the environment (``OMP_NUM_THREADS`` etc.),
    which must be set before this process starts.

    The warmup runs in three stages, so that what it pays for can be told apart.
    Stage 1 is a converged SCF on H2/def2-svp, which is almost entirely
    process-wide cost: the CUDA context, CuPy kernel compilation (compiled per
    source, so a two-electron system compiles nearly all of them), and first-use
    paths through pyscf and torch. Stage 2 repeats a single cycle on the real
    system, which stage 1 cannot cover because gpu4pyscf keys grids,
    sorted-orbital tables and density-fitting integrals on the molecule. Stage 3
    repeats the exchange-correlation evaluation until its cost settles, which is
    where TorchScript compiles for this system's shapes.

    Skipping any of them moves cost into the measurement rather than removing
    it: without stage 2 the first measured iteration grows by about half, and
    without stage 3 the first evaluation of the measured kernel carries the
    compile.

    The warmup is timed: ``load_ms``, ``warmup_ms``, and ``build_ms`` account
    for everything before the measured kernel, so the phases sum to
    ``worker_ms`` and, with the orchestrator's startup residual, to the whole
    subprocess.
    """
    device = config.device
    t_start = time.perf_counter()
    try:
        functional = _resolve_functional(config.functional, device)
        t_loaded = time.perf_counter()

        # Stage 1: a converged SCF on the smallest possible system. Almost none
        # of this is the calculation -- it is the CUDA context, the CuPy kernels
        # (compiled per source, so a two-electron system compiles nearly all of
        # them), and the first-use paths through pyscf and torch.
        process_mol = _build_mol(_WARMUP_MOLECULE, _WARMUP_BASIS)
        process_mf = _build_mf(
            process_mol,
            dataclasses.replace(config, molecule=_WARMUP_MOLECULE, basis=_WARMUP_BASIS),
            device,
            functional=functional,
        )
        process_mf.kernel()
        if device is Device.GPU:
            _cuda_sync()
        process_mf.reset()
        del process_mf, process_mol
        _release_warmup_memory(device)
        t_process = time.perf_counter()

        # Stage 2: the exchange-correlation evaluation, repeated on the real
        # system until its cost settles. This runs before anything else touches
        # the target, so the TorchScript recompilation for its shapes is
        # isolated here rather than hidden inside an SCF.
        warmup_mol = _build_mol(config.molecule, config.basis)
        warmup_mf = _build_mf(warmup_mol, config, device, functional=functional)
        warmup_mf.grids.build()
        settle = _settle_xc_evaluation(warmup_mf, device)
        if device is Device.GPU:
            _cuda_sync()
        t_settle = time.perf_counter()

        # Stage 3: one SCF cycle on the real system, for what stage 1 cannot
        # cover because gpu4pyscf keys grids, sorted-orbital tables and
        # density-fitting integrals on the molecule.
        warmup_mf.max_cycle = 1
        warmup_mf.kernel()
        if device is Device.GPU:
            _cuda_sync()
        warmup_mf.reset()
        del warmup_mf, warmup_mol
        _release_warmup_memory(device)
        t0 = time.perf_counter()
        mol = _build_mol(config.molecule, config.basis)
        mf = _build_mf(mol, config, device, functional=functional)
        t_kernel = time.perf_counter()
        with instrument(
            mf, device=_timeline_device(device), functional=functional
        ) as measurement:
            energy = float(mf.kernel())
            if device is Device.GPU:
                _cuda_sync()
        t_end = time.perf_counter()
        timing = measurement.result()
        conv_tol_grad = float(
            getattr(mf, "conv_tol_grad", None) or math.sqrt(config.conv_tol)
        )
        return RunResult(
            total_energy=energy,
            is_converged=bool(mf.converged),
            num_scf_iterations=len(timing.cycles),
            num_atoms=int(mf.mol.natm),
            num_electrons=int(mf.mol.nelectron),
            num_atomic_orbitals=int(mf.mol.nao_nr()),
            num_aux_basis_functions=(
                _num_aux_basis_functions(mf) if config.density_fit else None
            ),
            grid_size=int(mf.grids.weights.shape[0]),
            conv_tol_grad=conv_tol_grad,
            device=device.value,
            worker_ms=1e3 * (t_end - t_start),
            boot_ms=_boot_ms(launched_at),
            load_ms=1e3 * (t_loaded - t_start),
            warmup_ms=1e3 * (t0 - t_loaded),
            process_warmup_ms=1e3 * (t_process - t_loaded),
            target_warmup_ms=1e3 * (t0 - t_settle),
            settle_ms=1e3 * (t_settle - t_process),
            jit_compile_ms=compilation_excess(settle),
            settle_evaluations_ms=list(settle),
            build_ms=1e3 * (t_kernel - t0),
            wall_time_ms=1e3 * (t_end - t0),
            kernel_time_ms=1e3 * (t_end - t_kernel),
            setup_ms=timing.setup_ms,
            finalize_ms=timing.finalize_ms,
            cycles=[dataclasses.asdict(cycle) for cycle in timing.cycles],
        )
    except Exception as exc:  # noqa: BLE001 - report failures as a row, don't crash
        return RunError(error=repr(exc))


def _settle_xc_evaluation(
    mf: SCF,
    device: Device,
    *,
    max_evaluations: int = 10,
    tolerance: float = 0.2,
) -> list[float]:
    """Repeat XC evaluations until the compiles are done and the cost is stable.

    TorchScript does not compile a graph on first use. It runs an instrumented
    graph for ``num_profiled_runs`` executions, counting down once per execution,
    and builds the optimized plan only once that count reaches zero
    (``ProfilingRecord::ready`` in the pytorch source). A differentiable module
    pays this twice, because the derivative graph is a separate executor that
    only starts counting when the forward one is done.

    Measured on an A100 with ``n = num_profiled_runs``, the expensive
    evaluations are the 1st, the ``n + 1``-th, and the ``2n + 1``-th, and the
    evaluations in between run at full speed::

        n=1:  450  216  888   50   50   50   49   49   50   49   48   48
        n=2:  455   48  222   54  918   50   49   49   48   46   48   48
        n=4:  442   48   47   47  210   54   54   52  892   48   47   48

    Waiting only for two consecutive evaluations to agree is therefore not
    enough: at ``n = 4`` the pair at the 3rd and 4th would stop the warmup and
    leave the compiles at the 5th and 9th to the measured run. So the number of
    profiled runs is read from torch and at least ``2n + 1`` evaluations are
    performed, with the stability test kept as a backstop for anything else
    that warms up lazily.

    Args:
        mf: A warmed mean field, whose grids and initial guess are reused.
        device: The device the calculation runs on.
        max_evaluations: Cap, so a noisy machine cannot loop indefinitely.
        tolerance: Relative agreement between consecutive evaluations.

    Returns:
        One duration in milliseconds per evaluation performed. Their excess over
        the cheapest of them is what compilation cost.
    """
    numint = getattr(mf, "_numint", None)
    density_matrix = mf.get_init_guess(mf.mol, mf.init_guess)
    unrestricted = getattr(density_matrix, "ndim", 2) == 3
    evaluate = getattr(numint, "nr_uks" if unrestricted else "nr_rks", None)
    if not callable(evaluate):
        return []

    minimum = 2 * _num_profiled_runs() + 1
    timings: list[float] = []
    for count in range(1, max(minimum, max_evaluations) + 1):
        start = time.perf_counter()
        evaluate(mf.mol, mf.grids, mf.xc, density_matrix)
        if device is Device.GPU:
            _cuda_sync()
        timings.append(1e3 * (time.perf_counter() - start))
        if count < 2:
            continue
        # Agreement has to be symmetric. A one-sided test would stop on the
        # falling edge before the compile, which arrives as a later spike.
        previous, elapsed = timings[-2], timings[-1]
        settled = max(elapsed, previous) <= min(elapsed, previous) * (1.0 + tolerance)
        if settled and count >= minimum:
            break
    return timings


def compilation_excess(timings: Sequence[float]) -> float:
    """Return what repeated evaluation cost above its own steady state.

    The evaluations differ only in that the earlier ones compile, so charging
    every evaluation at the cheapest observed cost and keeping the remainder
    isolates compilation without needing to know which call compiled.
    """
    if not timings:
        return 0.0
    return max(sum(timings) - len(timings) * min(timings), 0.0)


def _num_profiled_runs(default: int = 1) -> int:
    """Executions TorchScript profiles before it compiles an optimized graph.

    Read rather than set: this only decides how long to warm up, and leaves the
    executor doing what it does for everyone else. The accessor is private, so
    a torch version without it falls back to the long-standing default.
    """
    import torch

    try:
        return max(int(torch._C._jit_get_num_profiled_runs()), 1)  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError):
        return default


def _boot_ms(launched_at: float | None) -> float | None:
    """Milliseconds from the parent's launch timestamp to this module loading.

    Returns ``None`` if no launch timestamp was supplied, or if the difference is
    negative, which would mean the two clocks are not comparable (the assumption
    holds on Linux, where ``perf_counter`` is ``CLOCK_MONOTONIC``, but is not
    guaranteed elsewhere).
    """
    if launched_at is None:
        return None
    boot_ms = 1e3 * (_MODULE_READY_AT - launched_at)
    return boot_ms if boot_ms >= 0.0 else None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Path to a RunConfig JSON file."
    )
    parser.add_argument("--result", required=True, help="Path for the result JSON.")
    parser.add_argument(
        "--launched-at",
        type=float,
        default=None,
        help="Parent's perf_counter reading just before starting this process.",
    )
    args = parser.parse_args(argv)
    config = RunConfig.from_json(args.config)
    try:
        _write_outcome(_run_scf(config, args.launched_at), Path(args.result))
    finally:
        _release_gpu4pyscf_global_cache(config.device)


def _write_outcome(outcome: RunOutcome, result_path: Path) -> None:
    """Atomically persist a worker outcome."""
    temporary = result_path.with_suffix(f"{result_path.suffix}.tmp")
    temporary.write_text(json.dumps(dataclasses.asdict(outcome)), encoding="utf-8")
    temporary.replace(result_path)


def count_atomic_orbitals(molecule: Molecule, basis: str) -> int:
    """Return the number of atomic orbitals without constructing a mean field."""
    return int(_build_mol(molecule, basis).nao_nr())


def _num_aux_basis_functions(mf: SCF) -> int | None:
    """Number of density-fitting auxiliary basis functions (CPU or GPU backend)."""
    df = getattr(mf, "with_df", None)
    if df is None:
        return None
    if hasattr(df, "get_naoaux"):  # pyscf CPU DF
        return int(df.get_naoaux())
    auxmol = getattr(df, "auxmol", None)  # gpu4pyscf DF exposes the aux Mole
    return int(auxmol.nao_nr()) if auxmol is not None else None


def _build_mol(molecule: Molecule, basis: str) -> gto.Mole:
    from pyscf import gto

    atoms = [
        [int(number), (float(x), float(y), float(z))]
        for number, (x, y, z) in zip(
            molecule.atomic_numbers, molecule.geometry_bohr, strict=True
        )
    ]
    return gto.M(
        atom=atoms,
        basis=basis,
        charge=molecule.charge,
        spin=molecule.multiplicity - 1,
        unit="Bohr",
        verbose=0,
    )


def _build_mf(
    mol: gto.Mole,
    config: RunConfig,
    device: Device,
    *,
    functional: ExcFunctionalBase | str | None = None,
) -> SCF:
    func = (
        functional
        if functional is not None
        else _resolve_functional(config.functional, device)
    )
    mf = _make_ks(mol, func, ansatz=config.ansatz, device=device)

    if config.density_fit:
        mf = mf.density_fit(auxbasis=config.auxbasis)

    mf.grids.level = config.grid_level
    mf.grids.reset()  # force a rebuild at the requested level
    mf.conv_tol = config.conv_tol
    if config.conv_tol_grad is not None:
        mf.conv_tol_grad = config.conv_tol_grad
    mf.verbose = 0
    return mf


def _resolve_functional(
    spec: FunctionalSpec, device: Device
) -> ExcFunctionalBase | str:
    """Resolve the explicitly selected functional implementation.

    ``skala.functional`` is imported here rather than at module scope because it
    pulls in e3nn and torch, which take about 5.7 s. A classical functional never
    touches the network, and two thirds of the default grid is classical.
    """
    if spec.kind is FunctionalKind.NATIVE:
        return spec.name
    from skala.functional import load_functional

    return load_functional(spec.name, device=_torch_device(device))


def _make_ks(
    mol: gto.Mole, func: ExcFunctionalBase | str, *, ansatz: str, device: Device
) -> SCF:
    """Construct the (Skala or PySCF-native) KS object on the chosen device."""
    unrestricted = ansatz == "UKS"
    if isinstance(func, str):
        # PySCF-native functional (e.g. a baseline like "pbe").
        if device is Device.GPU:
            from gpu4pyscf import dft
        else:
            from pyscf import dft
        return dft.UKS(mol, xc=func) if unrestricted else dft.RKS(mol, xc=func)
    if device is Device.GPU:
        from skala.gpu4pyscf.dft import SkalaRKS, SkalaUKS
    else:
        from skala.pyscf.dft import SkalaRKS, SkalaUKS
    return SkalaUKS(mol, xc=func) if unrestricted else SkalaRKS(mol, xc=func)


def _release_gpu4pyscf_global_cache(device: Device) -> None:
    """Destroy cached CuPy arrays before the worker process shuts down."""
    if device is not Device.GPU:
        return
    mole_module = sys.modules.get("gpu4pyscf.gto.mole")
    if mole_module is None:
        return
    c2s_cache = getattr(mole_module, "_c2s", None)
    if isinstance(c2s_cache, dict):
        c2s_cache.clear()
        gc.collect()


def _release_warmup_memory(device: Device) -> None:
    """Release calculation-sized allocations before building the measured SCF."""
    gc.collect()
    if device is Device.GPU:
        import torch

        torch.cuda.empty_cache()


def validate_device(device: Device) -> None:
    """Fail clearly when the requested compute target is unavailable."""
    if device is Device.CPU:
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GPU benchmark requested, but torch reports no CUDA device")
    try:
        import cupy  # noqa: F401
        import gpu4pyscf  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "GPU benchmark requested, but cupy or gpu4pyscf is unavailable"
        ) from exc


def _timeline_device(device: Device) -> str:
    """Return the timeline selector for ``device``, without importing torch.

    :func:`skala_benchmark.timing.make_timeline` only needs to know whether the
    work runs on CUDA. Passing a string keeps torch out of the measured region:
    a classical functional on CPU never imports it otherwise, and importing it
    between the kernel timestamps would put seconds of module loading inside the
    measured calculation.
    """
    return "cuda" if device is Device.GPU else "cpu"


def _torch_device(device: Device) -> torch.device:
    import torch

    return torch.device("cuda" if device is Device.GPU else "cpu")


def _cuda_sync() -> None:
    """Block until pending GPU work finishes, so timings are accurate."""
    import cupy

    cupy.cuda.Stream.null.synchronize()


if __name__ == "__main__":
    main()
