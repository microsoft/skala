#
# GauXC Copyright (c) 2020-2024, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy).
#
# (c) 2024-2025, Microsoft Corporation
#
# All rights reserved.
#
# See LICENSE.txt for details
#
cmake_minimum_required( VERSION 3.20 FATAL_ERROR )
project( gpu_snlink LANGUAGES CXX CUDA )

add_library( snlink_gpu 
${CMAKE_CURRENT_LIST_DIR}/../src/chebyshev_boys_computation.cu
${CMAKE_CURRENT_LIST_DIR}/../src/integral_0.cu
${CMAKE_CURRENT_LIST_DIR}/../src/integral_0_0.cu
${CMAKE_CURRENT_LIST_DIR}/../src/integral_1.cu
${CMAKE_CURRENT_LIST_DIR}/../src/integral_1_0.cu
${CMAKE_CURRENT_LIST_DIR}/../src/integral_1_1.cu
${CMAKE_CURRENT_LIST_DIR}/../src/integral_2.cu
${CMAKE_CURRENT_LIST_DIR}/../src/integral_2_0.cu
${CMAKE_CURRENT_LIST_DIR}/../src/integral_2_1.cu
${CMAKE_CURRENT_LIST_DIR}/../src/integral_2_2.cu
${CMAKE_CURRENT_LIST_DIR}/../src/obara_saika_integrals.cu
)
target_include_directories( snlink_gpu PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/../include
  ${CMAKE_CURRENT_LIST_DIR}/../src
  ${CMAKE_CURRENT_LIST_DIR}/../../../../../../../include
)
target_compile_options( snlink_gpu PRIVATE --expt-relaxed-constexpr ) 
target_compile_features( snlink_gpu PUBLIC cxx_std_17 )
set_property(TARGET snlink_gpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)

find_package( CUDAToolkit REQUIRED )
find_package( Libint2 REQUIRED )

add_executable( snlink_test test.cpp )
target_link_libraries( snlink_test PUBLIC snlink_gpu )
target_link_libraries( snlink_test PUBLIC Libint2::cxx )
if(NOT GAUXC_LINK_CUDA_STATIC)
  target_link_libraries( snlink_test PUBLIC CUDA::cudart )
else()
  target_link_libraries( snlink_test PUBLIC CUDA::cudart_static )
endif()
