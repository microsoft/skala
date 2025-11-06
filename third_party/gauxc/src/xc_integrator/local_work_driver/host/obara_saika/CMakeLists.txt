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
set( GAUXC_OBARA_SAIKA_HOST_SRC
     src/integral_0.cxx
     src/integral_1.cxx
     src/integral_2.cxx
     src/integral_3.cxx
     src/integral_4.cxx
     src/integral_0_0.cxx
     src/integral_1_0.cxx
     src/integral_1_1.cxx
     src/integral_2_0.cxx
     src/integral_2_1.cxx
     src/integral_2_2.cxx
     src/integral_3_0.cxx
     src/integral_3_1.cxx
     src/integral_3_2.cxx
     src/integral_3_3.cxx
     src/integral_4_0.cxx
     src/integral_4_1.cxx
     src/integral_4_2.cxx
     src/integral_4_3.cxx
     src/integral_4_4.cxx
     src/obara_saika_integrals.cxx
     src/chebyshev_boys_computation.cxx
)
target_sources( gauxc PRIVATE ${GAUXC_OBARA_SAIKA_HOST_SRC} )
target_include_directories( gauxc PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
)
