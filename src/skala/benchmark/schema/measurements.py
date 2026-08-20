# SPDX-License-Identifier: MIT

"""Partitioned parquet schema for benchmark *results*.

Each row is one measurement of a single DFT computation.
Rows link to an :class:`~skala.benchmark.schema.environment.Environment`
via ``env_id``. Shards are written Hive-partitioned by ``shard_index`` so a full
run merges by scanning the directory as a dataset.
"""

import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

#: One SCF iteration, timed directly (see :mod:`skala.benchmark.timing`). The
#: layers nest: ``xc_eval_ms <= numint_ms <= veff_ms <= wall_ms``, and for a
#: neural functional ``xc_eval_ms == forward_ms + backward_ms``.
CYCLE_STRUCT = pa.struct(
    [
        ("cycle", pa.int32()),
        ("wall_ms", pa.float64()),  # whole iteration
        ("veff_ms", pa.float64()),  # effective-potential build: J/K + XC
        ("numint_ms", pa.float64()),  # XC quadrature within the veff build
        ("xc_eval_ms", pa.float64()),  # the functional itself, energy + derivative
        ("forward_ms", pa.float64()),  # neural network forward (0 for classical)
        ("backward_ms", pa.float64()),  # neural network derivative (0 for classical)
        ("veff_calls", pa.int32()),
        ("numint_calls", pa.int32()),
        ("xc_eval_calls", pa.int32()),
    ]
)

MEASUREMENT_FIELDS: list[tuple[str, pa.DataType]] = [
    # Identity / linkage
    ("id", pa.string()),  # uuid
    ("env_id", pa.string()),  # -> Environment.env_id (a plain name)
    ("shard_index", pa.int32()),
    ("num_shards", pa.int32()),
    ("timestamp", pa.timestamp("us", tz="UTC")),
    ("skala_commit_hash", pa.string()),
    ("sweep_fingerprint", pa.string()),
    # Configuration
    ("basis", pa.string()),
    ("functional", pa.string()),
    ("functional_kind", pa.string()),  # "skala" | "native"
    ("mol_name", pa.string()),
    ("mol_hash", pa.string()),
    ("charge", pa.int32()),
    ("multiplicity", pa.int32()),
    ("ansatz", pa.string()),  # "RKS" | "UKS"
    ("density_fit", pa.bool_()),
    ("auxbasis", pa.string()),
    ("grid_level", pa.int32()),
    ("conv_tol", pa.float64()),
    ("conv_tol_grad", pa.float64()),
    # System size
    ("num_atoms", pa.int32()),
    ("num_electrons", pa.int32()),
    ("num_atomic_orbitals", pa.int32()),
    ("num_aux_basis_functions", pa.int32()),  # null when density fitting is off
    ("grid_size", pa.int64()),
    ("device", pa.string()),  # "cpu" | "gpu" -- what the run actually used
    # SCF outcome
    ("is_converged", pa.bool_()),
    ("num_scf_iterations", pa.int32()),
    ("total_energy", pa.float64()),
    # Timing. The worker phases partition the subprocess exactly:
    #   process_wall_ms == boot_ms + worker_ms + teardown_ms
    #   startup_ms      == boot_ms + teardown_ms
    #   worker_ms == load_ms + warmup_ms + build_ms + kernel_time_ms
    #   warmup_ms == process_warmup_ms + target_warmup_ms + settle_ms
    #   kernel_time_ms == setup_ms + sum(cycles.wall_ms) + finalize_ms
    ("process_wall_ms", pa.float64()),  # subprocess, as the orchestrator sees it
    ("startup_ms", pa.float64()),  # boot + teardown
    ("boot_ms", pa.float64()),  # process creation, interpreter start, imports
    ("teardown_ms", pa.float64()),  # result write, shutdown, reaping
    ("worker_ms", pa.float64()),  # everything the worker itself timed
    ("load_ms", pa.float64()),  # loading the functional onto the device
    # Warmup, in three stages. warmup_ms is their sum.
    ("warmup_ms", pa.float64()),
    ("process_warmup_ms", pa.float64()),  # converged SCF on H2/def2-svp
    ("target_warmup_ms", pa.float64()),  # single cycle on the real system
    ("settle_ms", pa.float64()),  # XC evaluations until the cost settles
    ("jit_compile_ms", pa.float64()),  # their excess over their steady state
    ("settle_evaluations_ms", pa.list_(pa.float64())),  # one per evaluation
    ("build_ms", pa.float64()),  # measured molecule + mean-field construction
    ("wall_time_ms", pa.float64()),  # build_ms + kernel_time_ms
    ("kernel_time_ms", pa.float64()),  # mf.kernel() only
    # Kernel entry to the start of the SCF loop: grid construction,
    # one-electron integrals, the initial guess, and the initial-guess Fock
    # build (which carries the one-time density-fitting integral build and,
    # being a full effective-potential evaluation, one XC quadrature).
    ("setup_ms", pa.float64()),
    # Work after the last cycle, chiefly pyscf's post-loop convergence check.
    # setup_ms + sum(cycles.wall_ms) + finalize_ms is the whole kernel.
    ("finalize_ms", pa.float64()),
    # Per-iteration timings. Cycle 0 carries the one-time warmup cost, so
    # steady-state metrics are derived from later cycles.
    ("cycles", pa.list_(CYCLE_STRUCT)),
    # Status
    ("status", pa.string()),  # "ok" | "error" | "oom" | "timeout"
    ("error", pa.string()),
]

MEASUREMENTS_SCHEMA = pa.schema(MEASUREMENT_FIELDS)

#: Column the dataset is Hive-partitioned by.
PARTITION_COLUMN = "shard_index"

#: A single measurement row, keyed by :data:`MEASUREMENTS_SCHEMA` field names.
Row = Mapping[str, Any]


def make_row(**fields: Any) -> Row:
    names = set(MEASUREMENTS_SCHEMA.names)
    unknown = set(fields) - names
    if unknown:
        raise ValueError(f"unknown measurement field(s): {sorted(unknown)}")
    row = {name: fields.get(name) for name in MEASUREMENTS_SCHEMA.names}
    if row.get("id") is None:
        row["id"] = str(uuid.uuid4())
    return row


def to_table(rows: list[Row]) -> pa.Table:
    """Build a parquet-ready table from measurement rows."""
    columns = {
        name: [row.get(name) for row in rows] for name in MEASUREMENTS_SCHEMA.names
    }
    return pa.table(columns, schema=MEASUREMENTS_SCHEMA)


def write_shard(
    rows: list[Row],
    directory: str | Path,
    shard_index: int,
    *,
    checkpoint_name: str | None = None,
) -> Path:
    """Write one shard as a Hive-partitioned parquet file.

    The file lands at ``<directory>/shard_index=<shard_index>/<uuid>.parquet``,
    so multiple shards coexist and merge via :func:`read_dataset`.
    """
    directory = Path(directory)
    part_dir = directory / f"{PARTITION_COLUMN}={shard_index}"
    part_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_name is not None and Path(checkpoint_name).name != checkpoint_name:
        raise ValueError("checkpoint_name must be a file name, not a path")
    path = part_dir / (checkpoint_name or f"{uuid.uuid4()}.parquet")
    table = to_table(rows)
    # The partition column is encoded in the path; drop it from the file body.
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    pq.write_table(table.drop_columns([PARTITION_COLUMN]), temporary)
    temporary.replace(path)
    return path


def read_dataset(directory: str | Path) -> pa.Table:
    """Read and merge all shards under ``directory`` into one table."""
    dataset = ds.dataset(
        directory,
        schema=MEASUREMENTS_SCHEMA,
        format="parquet",
        partitioning=ds.partitioning(
            pa.schema([(PARTITION_COLUMN, pa.int32())]), flavor="hive"
        ),
    )
    # Hive partitioning appends the partition column last; restore canonical order.
    return dataset.to_table().select(MEASUREMENTS_SCHEMA.names)
