# SPDX-License-Identifier: MIT

"""Command-line interface for running, collecting, and reporting benchmarks."""

from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Sequence
from pathlib import Path

from skala.benchmark.orchestrator import SweepRequest, parse_duration, run_sweep
from skala.benchmark.protocol import DEFAULT_PROTOCOL, Device


def main(argv: Sequence[str] | None = None) -> None:
    """Run one stage of the benchmark workflow."""
    parser = argparse.ArgumentParser(
        prog="python -m skala.benchmark",
        description="Run, collect, and report the Skala DFT benchmark.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one benchmark shard.")
    run_parser.add_argument("output_dir", help="Raw benchmark dataset directory.")
    run_parser.add_argument("--env-id", required=True, help="Stable environment id.")
    run_parser.add_argument(
        "--env-label",
        default=None,
        help="Human-readable environment label (default: --env-id).",
    )
    run_parser.add_argument(
        "--device",
        required=True,
        choices=[device.value for device in Device],
        help="Required compute target; unavailable targets fail instead of falling back.",
    )
    run_parser.add_argument("--shard-index", type=int, default=0)
    run_parser.add_argument("--num-shards", type=int, default=1)
    run_parser.add_argument("--max-atoms", type=int, default=None)
    run_parser.add_argument("--max-orbitals", type=int, default=None)
    run_parser.add_argument("--name", action="append", default=None)
    run_parser.add_argument(
        "--basis",
        action="append",
        default=None,
        help="Restrict the sweep to these basis sets (repeatable).",
    )
    run_parser.add_argument(
        "--time-limit",
        type=parse_duration,
        default=None,
        metavar="DURATION",
        help="Maximum wall time per computation, for example 240m or 4h.",
    )

    run_parser.add_argument(
        "--dataset-dir",
        default=None,
        help=(
            "Directory holding the fetched molecule set "
            "(default: the fetch-dataset location)."
        ),
    )

    fetch_parser = subparsers.add_parser(
        "fetch-dataset",
        help="Download the benchmark molecules from the datasets they come from.",
    )
    fetch_parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Directory to write the molecule set to.",
    )

    collect_parser = subparsers.add_parser(
        "collect", help="Collect a raw dataset into report-ready JSON files."
    )
    collect_parser.add_argument("input_dir", help="Raw benchmark dataset directory.")
    collect_parser.add_argument(
        "--output-dir",
        default=None,
        help="Collected directory (default: INPUT_DIR/collected).",
    )

    report_parser = subparsers.add_parser(
        "report", help="Generate an offline report from collected directories."
    )
    report_parser.add_argument("output_dir", help="Destination report directory.")
    report_parser.add_argument(
        "collected_dirs",
        nargs="+",
        help=(
            "One or more directories produced by collect, each containing "
            "environments.json, measurements.json, and fits.json."
        ),
    )
    report_parser.add_argument(
        "--prose",
        default=None,
        help="Optional prose YAML for an interpreted report.",
    )

    args = parser.parse_args(argv)
    if args.command == "run":
        _validate_shard_args(run_parser, args)
        protocol = DEFAULT_PROTOCOL
        if args.basis:
            unknown = sorted(set(args.basis) - set(protocol.bases))
            if unknown:
                run_parser.error(
                    f"unknown basis set(s): {', '.join(unknown)}; "
                    f"choose from {', '.join(protocol.bases)}"
                )
            protocol = dataclasses.replace(
                protocol,
                bases=tuple(b for b in protocol.bases if b in set(args.basis)),
            )
        run_sweep(
            SweepRequest(
                output_dir=Path(args.output_dir),
                env_id=args.env_id,
                env_label=args.env_label or args.env_id,
                device=Device(args.device),
                shard_index=args.shard_index,
                num_shards=args.num_shards,
                max_atoms=args.max_atoms,
                max_orbitals=args.max_orbitals,
                molecule_names=tuple(args.name) if args.name else None,
                time_limit_seconds=args.time_limit,
                protocol=protocol,
                dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None,
            )
        )
    elif args.command == "fetch-dataset":
        from skala.benchmark.fetch import main as fetch_main

        raise SystemExit(fetch_main(args.dataset_dir))
    elif args.command == "collect":
        from skala.benchmark.collect_results import collect_results

        output_dir = args.output_dir or str(Path(args.input_dir) / "collected")
        collect_results(args.input_dir, output_dir)
    else:
        from skala.benchmark.report.generate import generate

        generate(args.output_dir, args.collected_dirs, prose_path=args.prose)


def _validate_shard_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.num_shards < 1:
        parser.error("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if args.max_atoms is not None and args.max_atoms < 1:
        parser.error("--max-atoms must be >= 1")
    if args.max_orbitals is not None and args.max_orbitals < 1:
        parser.error("--max-orbitals must be >= 1")


if __name__ == "__main__":
    main()
