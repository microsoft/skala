# SPDX-License-Identifier: MIT

"""Raw measurement and collected JSON definitions for the benchmark suite.

- :mod:`skala.benchmark.schema.measurements` defines the partitioned parquet
  schema for benchmark *results* (one row per measured DFT computation,
  carrying its per-iteration timings in a nested ``cycles`` column).
- :mod:`skala.benchmark.schema.environment` defines the per-node *environment*
  spec, serialized as a JSON file named after ``env_id``.
- :mod:`skala.benchmark.schema.fits` serializes the precomputed scaling fits.

Results link to an environment via the ``env_id`` string.
"""
