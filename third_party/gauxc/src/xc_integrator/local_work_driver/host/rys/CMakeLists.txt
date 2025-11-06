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
set( GAUXC_RYS_HOST_SRC 
     src/boys_table.c
     src/jacobi_table.c
     src/rys_1rw.c
     src/rys_2rw.c
     src/rys_3rw.c
     src/rys_4rw.c
     src/rys_5rw.c
     src/rys_integral.c
     src/rys_rw.c
     src/rys_xrw.c )

target_sources( gauxc PRIVATE ${GAUXC_RYS_HOST_SRC} )
target_include_directories( gauxc PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
)
