#!/usr/bin/env bash
set -euo pipefail

SKALAXC_SOURCE="${SRC_DIR}/SkalaXC"
NATIVE_BUILD="${SRC_DIR}/build-python-package"
NATIVE_PREFIX="${SRC_DIR}/build-python-prefix"
mapfile -t TORCH_CMAKE_ARGS < <(
  "${PYTHON}" "${SKALAXC_SOURCE}/python/tools/torch_config.py" --format cmake
)

cmake -S "${SKALAXC_SOURCE}" -B "${NATIVE_BUILD}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${NATIVE_PREFIX}" \
  -DSKALAXC_BUILD_EXAMPLES=OFF \
  -DSKALAXC_BUILD_FORTRAN=OFF \
  -DSKALAXC_BUILD_TESTS=OFF \
  -DSKALAXC_DOWNLOAD_MODELS=OFF \
  -DSKALAXC_ENABLE_CUDA=OFF \
  -DSKALAXC_ENABLE_MPI=OFF \
  '-DSKALAXC_INSTALL_RPATH=$ORIGIN/../../..' \
  -DSKALAXC_MODEL_PATH="${SRC_DIR}/data/skala_models" \
  -DBLAS_LIBRARIES="${PREFIX}/lib/libblas.so" \
  "${TORCH_CMAKE_ARGS[@]}"
cmake --build "${NATIVE_BUILD}" --parallel "${CPU_COUNT}"
cmake --install "${NATIVE_BUILD}"

SkalaXC_DIR="${NATIVE_PREFIX}/lib/cmake/SkalaXC" \
  "${PYTHON}" -m pip install "${SKALAXC_SOURCE}/python" \
  --no-build-isolation --no-deps -vv