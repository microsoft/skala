# DFT benchmark

This benchmark has two purposes:

1. validate another Skala implementation and compare its speed with the official
   implementation;
2. show Skala users precisely what performance to expect as molecular and basis
   size grow on documented compute environments.

The repository contains a fixed official result in `benchmarks/reference`.
Measurements from another machine or implementation are local comparison inputs;
they are not intended to be contributed back to the repository.

The benchmark has three stages with matching commands:

1. `run` executes one deterministic shard and records per-iteration timings.
2. `collect` merges those into report-ready Parquet tables.
3. `report` compares one or more collected environments in an offline HTML report.

The bundled protocol evaluates the molecule dataset with `def2-svp`, `def2-tzvp`,
and `def2-qzvp`, comparing Skala 1.1, r2SCAN, B3LYP, and M06-2X. Each measured
calculation is preceded by one unmeasured single-cycle SCF so process-wide
first-use costs fall outside the measurement.

## What is measured

Nothing is sampled or attributed. Four layers are timed directly, and they nest:

```text
xc_eval  in  numint  in  veff  in  cycle  in  kernel
```

| layer | how |
|---|---|
| `kernel` | wall clock around `mf.kernel()` |
| `cycle` | `mf.callback`, which pyscf and gpu4pyscf both invoke per iteration |
| `veff` | the mean field's `get_veff`: the J/K build plus the XC quadrature |
| `numint` | the mean field's `nr_uks` / `nr_rks` entry point |
| `xc_eval` | the functional itself, energy and derivative: `eval_xc_eff` for libxc; `get_exc` plus `torch.autograd.grad` for Skala |

Every patch is applied to the objects the benchmark itself constructs, so nothing
leaks into the pyscf, gpu4pyscf, or skala modules.

On CUDA the durations come from `torch.cuda.Event` rather than a host clock.
`get_exc` returns once its kernels are *queued* and the backward pass is queued
immediately behind it with no synchronization in between, so a host timer would
measure launch overhead instead of work -- an error that grows with chunk size.
Against a synchronized reference on an A100, a host clock under-reports Skala's
backward pass by 47% on water and 78% on a 13-atom molecule, while CUDA events
stay within 3%. The host-clock figures barely move between the two molecules,
which is the giveaway: they measure queueing, not computation. Events are
ordered within the stream that executes the work, so they are unaffected. Marks
are still created in host program order, so charging an interval to an iteration
remains a host-side decision.

The `numint` call does happen to be bracketed by synchronizations: `make_chunks`
calls `.item()` on the atomic grid sizes as the chunk loop starts, and `nr_uks`
ends with `E_xc.item()`. A host clock therefore gets `numint` right to within
1%. It gets everything *inside* it wrong, because nothing synchronizes between
`get_exc` and `torch.autograd.grad`. The work drains at the closing `.item()`,
outside both regions, so on a host clock the queued time is simply lost rather
than misattributed.

The phases partition the whole worker subprocess, at three levels:

```text
process_wall_ms == boot_ms + worker_ms + teardown_ms
startup_ms      == boot_ms + teardown_ms                       (definitional)
worker_ms       == load_ms + warmup_ms + build_ms + kernel_time_ms   (definitional)
kernel_time_ms  == setup_ms + sum(cycles.wall_ms) + finalize_ms      (cross-check)
```

A worker cannot time its own start-up, because it does not exist yet. It is
measured instead by comparing clocks: `time.perf_counter` is `CLOCK_MONOTONIC`
on Linux, which counts from system boot and is therefore comparable between
processes, so the parent passes its pre-`Popen` reading with `--launched-at` and
the worker subtracts it from the moment its module finished loading. That gives
`boot_ms` -- process creation, interpreter start-up, and imports -- as a
measurement rather than a residual. The check is one-directional: a negative
result would mean the clocks are not comparable, and the split is skipped.

`teardown_ms` is what remains: writing the result, interpreter shutdown, and the
parent reaping the process. It is the only genuinely unattributable part, and it
is small for a classical functional (~60 ms) but ten times larger for Skala
(~550 ms), which is torch tearing down a loaded model.

The `startup_ms` and `worker_ms` identities hold by construction and cannot
fail: the phases are consecutive readings of one clock, and the totals are
defined as the leftovers. They are useful for reading a row, not for validating
one.

The third compares two independent measurement paths -- the instrumentation
timeline against the runner's own stopwatch, and on GPU CUDA events against host
wall time -- so it can and does fail. It is how a lazy `import torch` evaluated
inside the measured region was found: it inflated `kernel_time_ms` by 2.1 s
while the phases inside the kernel knew nothing about it. It normally sits at
0.1 ms.

`startup_ms` is the part the worker cannot time itself: interpreter boot, module
imports, writing the result, and teardown. It is reported rather than hidden,
because a fresh interpreter per calculation is not free.

`load_ms` is loading the functional and `warmup_ms` is the single-cycle SCF
described below. `finalize_ms` is pyscf's post-loop convergence check, which
performs one further effective-potential build; it therefore costs about one
extra SCF iteration, and drops to ~0 if `conv_check` is disabled. `skala.functional` is imported lazily, inside the functional
lookup, because it pulls in e3nn and torch for about 5.7 s. A classical
functional never needs them, so its startup drops from roughly 6 s to 0.35 s and
its `load_ms` stays zero; for Skala the same cost simply appears in `load_ms`,
where it belongs.

`setup_ms` covers grid construction, one-electron integrals, the initial guess,
and the initial-guess Fock build.

Both pyscf and gpu4pyscf evaluate the effective potential once on the initial
guess *before* entering the SCF loop. That build is charged to setup rather than
to the first iteration: it happens before the callback that closes cycle 0, and
it carries the one-time density-fitting integral build, which is large (57 ms
against 0.9 ms of recurring J/K for water/def2-svp, and seconds for bigger
systems). Leaving it in would make every first iteration look about twice as
expensive as the rest and would show up as warmup that is not there. It is
detected structurally, by cycle 0 recording more potential builds than an
iteration needs, so a calculation without a pre-loop build is unaffected.

## Warmup

One-time costs are kept out of the per-iteration numbers in three ways, and the
per-iteration records exist so that this is verifiable rather than assumed:

1. The worker runs a single-cycle SCF before the measured one, so kernel
   compilation is already paid. This matters more than it sounds: Skala is a
   TorchScript module whose profiling executor compiles an optimized graph on the
   *second* execution, and CuPy compiles CUDA kernels on first launch. The warmup
   performs two exchange-correlation evaluations, which covers both. Measured on
   an A100, a first iteration costs 1441 ms without it and 47 ms with it; merely
   importing the modules and loading the weights only gets to 899 ms. The warmup
   is timed and reported as `warmup_ms`.
2. The pre-loop Fock build -- which carries the density-fitting integral build --
   is charged to `setup_ms`, not to the first iteration.
3. Scaling metrics use the **median of the steady-state iterations** (cycle 1
   onward), so a first iteration that still differs cannot distort a fit.

The `warmup` metric reports what remains, as cycle 0 divided by the steady-state
median. With the above in place it is close to 1 on both CPU and GPU; a value
meaningfully above 1 means some cost recurs per calculation and is worth
investigating.

## Install

Install the benchmark dependencies from a source checkout:

```bash
pip install -e '.[benchmark]'
```

Set threading variables before starting a run because worker subprocesses inherit
them:

```bash
export OMP_NUM_THREADS=16
```

Choose the physical core count appropriate for the machine. GPU runs additionally
require a compatible CUDA, CuPy, and `gpu4pyscf` installation. The requested
`--device` is strict: a GPU run fails if the GPU stack is unavailable instead of
silently producing CPU measurements.

## Run a local benchmark

Give every hardware/software configuration a stable id and a readable label:

```bash
python -m skala.benchmark run benchmark-output \
  --env-id cpu-local \
  --env-label 'Local 16-core CPU' \
  --device cpu \
  --max-orbitals 250 \
  --time-limit 4h
```

`--time-limit` applies to each DFT calculation, not the whole sweep. Use
`--max-atoms` or repeated `--name` options for a small smoke run, and repeated
`--basis` options to restrict the sweep to a subset of the protocol's basis
sets:

```bash
python -m skala.benchmark run benchmark-output \
  --env-id cpu-local --device cpu \
  --basis def2-svp --basis def2-tzvp
```

Restricting the basis set changes the sweep fingerprint, so a restricted run
needs its own output directory rather than resuming a full one.

The raw output has two parts:

```text
benchmark-output/
  environments/<env-id>.json
  measurements/shard_index=<i>/*.parquet
```

Each measurement row carries its per-iteration timings in a nested `cycles`
column. Each completed task is immediately written as its own atomic Parquet checkpoint,
so partial shard results remain readable while the next molecule is running.
Rerunning the same shard skips complete task identities. The stored sweep
fingerprint prevents resuming with a different device, protocol, molecule
ordering, or shard count. An unreadable checkpoint is reported as an error
rather than deleted.

### Sharding

The task grid is sorted by increasing molecule electron count. Shard `i` of `n`
receives `tasks[i::n]`, so allocation is round-robin, each shard preserves size
order, and the shards are disjoint and together cover the complete grid:

```bash
python -m skala.benchmark run benchmark-output \
  --env-id cpu-local \
  --env-label 'Local 16-core CPU' \
  --device cpu \
  --shard-index 0 \
  --num-shards 4
```

Run the other shard indices with the same output directory and environment id.

## Collect the raw results

Collection validates environment links and precomputes the scaling fits:

```bash
python -m skala.benchmark collect benchmark-output
```

By default this creates:

```text
benchmark-output/collected/
  environments.json
  measurements.json
  fits.json
```

The collected JSON is the portable, report-ready form of a benchmark run.
Per-iteration records remain nested in `measurements.json`.

## Generate a report

Pass the collected directory, not individual JSON files:

```bash
python -m skala.benchmark report report-out benchmark-output/collected
```

The command requires all three collected files and produces an offline bundle:

```text
report-out/
  index.html
  data.json
  fits.json
  metadata.yaml
  report-base.css
  report.css
  report.js
  d3.min.js
  katex.min.js
```

Open `report-out/index.html` locally. The report contains no network dependencies.

### Compare with the official result

Generate a private report that overlays your collected run on the official
repository baseline:

```bash
python -m skala.benchmark report local-report \
  benchmarks/reference \
  benchmark-output/collected
```

Environment ids must be unique across the inputs. Plot controls switch between
environments and basis sets while keeping global axes fixed.

The hosted documentation publishes the same official report and makes its
`environments.json`, `measurements.json`, and `fits.json` inputs available for
download.

## Time composition

Because the three measured layers nest, the report's stacked bands are
differences of measurements rather than attributed shares:

| band | definition |
|---|---|
| XC functional evaluation | `xc_eval` |
| Rest of XC quadrature | `numint - xc_eval` (AO evaluation, density assembly, Vxc assembly) |
| J/K build | `veff - numint` |
| Diagonalization, DIIS, bookkeeping | `cycle - veff` |

Both functional kinds are timed to first order, so the bands are comparable.
They are not perfectly equivalent: libxc returns the derivative with respect to
the density on the grid and pyscf contracts it into the AO basis afterwards,
outside the timed call, whereas Skala's backward differentiates all the way to
the density matrix and so performs that contraction inside. `xc_eval` therefore
slightly over-counts Skala relative to libxc, never the reverse.
