#!/usr/bin/env bash
set -euo pipefail

export SkalaXC_DIR="${PREFIX}/lib/cmake/SkalaXC"
export SKALAXC_PYTHON_LAYOUT=CONDA

"${PYTHON}" -m pip install "${SRC_DIR}/SkalaXC/python" \
  --no-build-isolation \
  --no-deps \
  --prefix "${PREFIX}" \
  --verbose
