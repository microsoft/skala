#!/usr/bin/env bash
set -euo pipefail

source_dir="${SRC_DIR}/SkalaXC"
build_dir="${BUILD_DIR}/skalaxc-native"

case "${SKALAXC_MPI_VARIANT}" in
  nompi) mpi_enabled=OFF ;;
  openmpi) mpi_enabled=ON ;;
  *) echo "Unsupported MPI variant: ${SKALAXC_MPI_VARIANT}" >&2; exit 2 ;;
esac

case "${SKALAXC_CUDA_VARIANT}" in
  cpu) cuda_enabled=OFF ;;
  cuda12|cuda13) cuda_enabled=ON ;;
  *) echo "Unsupported CUDA variant: ${SKALAXC_CUDA_VARIANT}" >&2; exit 2 ;;
esac

if [[ "$(uname -s)" == "Darwin" ]]; then
  install_rpath='@loader_path'
else
  install_rpath='$ORIGIN'
fi

torch_config="$(${PYTHON} "${source_dir}/python/tools/torch_config.py" --format json)"
torch_value() {
  printf '%s' "${torch_config}" | "${PYTHON}" -c \
    'import json, sys; print(json.load(sys.stdin)[sys.argv[1]])' "$1"
}
torch_dir="$(torch_value torch_dir)"
torch_cxx11_abi="$(torch_value cxx11_abi)"
torch_cuda_version="$(torch_value cuda)"
if [[ "${torch_cuda_version}" == "None" ]]; then
  torch_cuda_version=none
fi

export CONDA_PREFIX="${PREFIX}"
cmake -S "${source_dir}" -B "${build_dir}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  -DTorch_DIR="${torch_dir}" \
  -DSKALAXC_TORCH_CXX11_ABI="${torch_cxx11_abi}" \
  -DSKALAXC_TORCH_CUDA_VERSION="${torch_cuda_version}" \
  -DSKALAXC_BUILD_EXAMPLES=OFF \
  -DSKALAXC_BUILD_FORTRAN=ON \
  -DSKALAXC_BUILD_TESTS=OFF \
  -DSKALAXC_DOWNLOAD_MODELS=OFF \
  -DSKALAXC_ENABLE_CUDA="${cuda_enabled}" \
  -DSKALAXC_ENABLE_HDF5=ON \
  -DSKALAXC_ENABLE_MPI="${mpi_enabled}" \
  -DSKALAXC_ENABLE_OPENMP=ON \
  -DSKALAXC_ENABLE_SANITIZERS=OFF \
  -DSKALAXC_INSTALL_RPATH="${install_rpath}" \
  -DSKALAXC_MODEL_PATH="${SRC_DIR}/data/skala_models" \
  ${CMAKE_ARGS:-}

cmake --build "${build_dir}" --parallel "${CPU_COUNT}"
cmake --install "${build_dir}"
