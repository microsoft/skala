/**
 * GauXC Copyright (c) 2020-2024, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy).
 *
 * (c) 2024-2025, Microsoft Corporation
 *
 * All rights reserved.
 *
 * See LICENSE.txt for details
 */
#pragma once

namespace GauXC {

void compute_grid_to_center_dist( int32_t npts, int32_t natoms,
  const double* coords, const double* points_x,  const double* points_y, 
  const double* points_z, double* dist, int32_t lddist, hipStream_t stream );

}
