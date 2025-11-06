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
set( GAUXC_OBARA_SAIKA_CUDA_SRC
     src/integral_0.cu
     src/integral_1.cu
     src/integral_2.cu
     src/integral_0_0.cu
     src/integral_1_0.cu
     src/integral_1_1.cu
     src/integral_2_0.cu
     src/integral_2_1.cu
     src/integral_2_2.cu
     src/obara_saika_integrals.cu
     src/chebyshev_boys_computation.cu
)
target_sources( gauxc PRIVATE ${GAUXC_OBARA_SAIKA_CUDA_SRC} )
target_include_directories( gauxc PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
)

#add_executable( cuda_obara_saika_test test/test_new.cpp )
#target_link_libraries( cuda_obara_saika_test PUBLIC gauxc )
