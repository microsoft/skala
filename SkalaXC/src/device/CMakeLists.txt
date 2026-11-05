# SkalaXC CUDA device kernels.
#
# Built ONLY when SKALAXC_ENABLE_CUDA is ON and a CUDA compiler was found (the
# top-level CMakeLists enables the CUDA language behind that guard). These are
# faithful onedft->skala renames of the GauXC skala-branch ML device kernels.
# They compile against GauXC-master's private device headers, which the in-tree
# `gauxc` target exposes (and which must themselves be a CUDA-enabled GauXC
# build -- SKALAXC_ENABLE_CUDA propagates GAUXC_ENABLE_CUDA).
#
# NOTE: these kernels are the ML-specific VXC/gradient building blocks only. The
# full device orchestration driver (the SkalaXC-owned device-data adapter that
# feeds these kernels; see the port plan in the migration notes) is completed on
# a CUDA-capable machine, where it can be compiled and validated end-to-end.

add_library(skalaxc_cuda STATIC
  cuda/kernels/model_grid.cu
  cuda/kernels/skala.cu
  cuda/kernels/skala_exc_grad.cu
)
add_library(SkalaXC::cuda ALIAS skalaxc_cuda)

# `${CMAKE_CURRENT_SOURCE_DIR}` resolves device-local headers and the source
# root exposes shared private SkalaXC headers such as exceptions.hpp. GauXC's
# private device headers and CUDA infrastructure come from the `gauxc` target.
target_include_directories(skalaxc_cuda PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(skalaxc_cuda PRIVATE gauxc)

set_target_properties(skalaxc_cuda PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
