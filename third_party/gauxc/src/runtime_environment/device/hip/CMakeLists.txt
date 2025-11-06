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

find_package( hip REQUIRED )
find_package( hipblas REQUIRED )

target_sources( gauxc PRIVATE hip_backend.cxx )
target_link_libraries( gauxc PUBLIC hip::host roc::hipblas )
