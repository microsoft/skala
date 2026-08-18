# SPDX-License-Identifier: MIT

"""Tests for the benchmark measurements schema helpers."""

import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from skala.benchmark.schema.measurements import (
    MEASUREMENTS_SCHEMA,
    Row,
    make_row,
    read_dataset,
    to_table,
    write_shard,
)


def _row(shard_index: int = 0, mol_name: str = "H2O", **overrides: object) -> Row:
    fields: dict[str, object] = dict(
        env_id="cpu-4thread",
        shard_index=shard_index,
        num_shards=2,
        timestamp=datetime.datetime.now(datetime.UTC),
        skala_commit_hash="abc123",
        basis="def2-svp",
        functional="skala-1.1",
        mol_name=mol_name,
        mol_hash="h",
        charge=0,
        multiplicity=1,
        ansatz="RKS",
        density_fit=True,
        grid_level=3,
        conv_tol=5e-6,
        conv_tol_grad=5e-5,
        num_atoms=3,
        num_electrons=10,
        status="ok",
    )
    fields.update(overrides)
    return make_row(**fields)


def test_make_row_has_all_schema_columns_in_order() -> None:
    row = _row()
    assert list(row) == MEASUREMENTS_SCHEMA.names


def test_make_row_auto_fills_id() -> None:
    assert _row()["id"]
    assert _row()["id"] != _row()["id"]


def test_make_row_preserves_explicit_id() -> None:
    assert _row(id="fixed-id")["id"] == "fixed-id"


def test_make_row_defaults_missing_fields_to_none() -> None:
    assert _row()["error"] is None


def test_make_row_accepts_per_cycle_timings() -> None:
    assert "cycles" in MEASUREMENTS_SCHEMA.names
    assert _row()["cycles"] is None
    cycles = [
        {
            "cycle": 0,
            "wall_ms": 10.0,
            "numint_ms": 6.0,
            "forward_ms": 2.0,
            "numint_calls": 1,
            "forward_calls": 1,
        }
    ]
    assert _row(cycles=cycles)["cycles"] == cycles


def test_make_row_accepts_the_phase_columns() -> None:
    # setup + cycles + finalize partition the kernel.
    assert "setup_ms" in MEASUREMENTS_SCHEMA.names
    assert "finalize_ms" in MEASUREMENTS_SCHEMA.names
    assert _row(setup_ms=1.0, finalize_ms=2.0)["setup_ms"] == 1.0


def test_make_row_rejects_unknown_field() -> None:
    with pytest.raises(ValueError, match="unknown measurement field"):
        make_row(enviroment_id="x")


def test_to_table_conforms_to_schema() -> None:
    table = to_table([_row(), _row(mol_name="NH3")])
    assert table.num_rows == 2
    assert table.schema.equals(MEASUREMENTS_SCHEMA, check_metadata=False)


def test_to_table_rejects_wrong_type() -> None:
    with pytest.raises((pa.ArrowInvalid, pa.ArrowTypeError)):
        to_table([_row(charge="not-an-int")])


def test_write_and_read_merges_shards(tmp_path: Path) -> None:
    write_shard([_row(shard_index=0, mol_name="H2O")], tmp_path, 0)
    write_shard([_row(shard_index=1, mol_name="NH3")], tmp_path, 1)
    table = read_dataset(tmp_path)
    assert table.num_rows == 2
    assert table.schema.equals(MEASUREMENTS_SCHEMA, check_metadata=False)
    assert sorted(table.column("shard_index").to_pylist()) == [0, 1]
    assert set(table.column("mol_name").to_pylist()) == {"H2O", "NH3"}


def test_write_shard_partitions_by_shard_index(tmp_path: Path) -> None:
    path = write_shard([_row(shard_index=2)], tmp_path, 2)
    assert path.parent == tmp_path / "shard_index=2"
    assert path.suffix == ".parquet"


def test_write_shard_can_replace_named_checkpoint(tmp_path: Path) -> None:
    original = write_shard([_row(mol_name="old")], tmp_path, 0)

    replacement = write_shard(
        [_row(mol_name="new")],
        tmp_path,
        0,
        checkpoint_name=original.name,
    )

    assert replacement == original
    assert read_dataset(tmp_path).column("mol_name").to_pylist() == ["new"]


def test_read_dataset_adds_columns_missing_from_older_outputs(tmp_path: Path) -> None:
    partition = tmp_path / "shard_index=0"
    partition.mkdir()
    legacy = to_table([_row()]).drop_columns(
        ["shard_index", "functional_kind", "sweep_fingerprint"]
    )
    pq.write_table(legacy, partition / "legacy.parquet")

    table = read_dataset(tmp_path)

    assert table.column("functional_kind").to_pylist() == [None]
    assert table.column("sweep_fingerprint").to_pylist() == [None]


def test_read_dataset_preserves_new_columns_in_mixed_schema(tmp_path: Path) -> None:
    partition = tmp_path / "shard_index=0"
    partition.mkdir()
    legacy = to_table([_row(mol_name="legacy")]).drop_columns(
        ["shard_index", "functional_kind", "sweep_fingerprint"]
    )
    current = to_table(
        [
            _row(
                mol_name="current",
                functional_kind="native",
                sweep_fingerprint="fingerprint",
            )
        ]
    ).drop_columns(["shard_index"])
    pq.write_table(legacy, partition / "a-legacy.parquet")
    pq.write_table(current, partition / "b-current.parquet")

    table = read_dataset(tmp_path)
    rows = {row["mol_name"]: row for row in table.to_pylist()}

    assert rows["legacy"]["functional_kind"] is None
    assert rows["current"]["functional_kind"] == "native"
    assert rows["current"]["sweep_fingerprint"] == "fingerprint"
