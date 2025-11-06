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

$for( L in range(L_max + 1))
#include "collocation/collocation_shell_to_task_kernels_cartesian_l$(L).hpp"\
$endfor

$for( L in range(L_max + 1))
#include "collocation/collocation_shell_to_task_kernels_cartesian_l$(L)_gradient.hpp"\
$endfor

$for( L in range(L_max + 1))
#include "collocation/collocation_shell_to_task_kernels_cartesian_l$(L)_hessian.hpp"\
$endfor

$for( L in range(L_max + 1))
#include "collocation/collocation_shell_to_task_kernels_cartesian_l$(L)_laplacian.hpp"\
$endfor

$for( L in range(L_max + 1))
#include "collocation/collocation_shell_to_task_kernels_cartesian_l$(L)_lapgrad.hpp"\
$endfor

$for( L in range(L_max + 1))
#include "collocation/collocation_shell_to_task_kernels_spherical_l$(L).hpp"\
$endfor

$for( L in range(L_max + 1))
#include "collocation/collocation_shell_to_task_kernels_spherical_l$(L)_gradient.hpp"\
$endfor

$for( L in range(L_max + 1))
#include "collocation/collocation_shell_to_task_kernels_spherical_l$(L)_hessian.hpp"\
$endfor

$for( L in range(L_max + 1))
#include "collocation/collocation_shell_to_task_kernels_spherical_l$(L)_laplacian.hpp"\
$endfor

$for( L in range(L_max + 1))
#include "collocation/collocation_shell_to_task_kernels_spherical_l$(L)_lapgrad.hpp"\
$endfor
