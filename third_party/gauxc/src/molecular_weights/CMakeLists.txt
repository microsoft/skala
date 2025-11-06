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
target_sources( gauxc PRIVATE molecular_weights.cxx )
add_subdirectory( host )
if(GAUXC_HAS_DEVICE)
  add_subdirectory( device )
endif()
