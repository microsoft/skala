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
  double *hrrx = hrr_array + $(0*(lA+1) * (lB+1) * nroot);
  double *hrry = hrr_array + $(1*(lA+1) * (lB+1) * nroot);
  double *hrrz = hrr_array + $(2*(lA+1) * (lB+1) * nroot);

$for( i in range(len(rys_targets)) )\
  double _rys_target_$(i);
$endfor\
  double hrrx_tmp, hrry_tmp, hrrz_tmp;

$for( k in range(nroot) )\
$for( i in range(len(rys_targets)) )\
  _rys_target_$(i) = 1.;
$endfor

$for( tgt_idx,u_target in enumerate(uniq_targets,start=0) )\
$py(i,j = u_target)\
$py(idx = (lB+1)*nroot*i + nroot*j + k)\
  hrrx_tmp = hrrx[$(idx)];
  hrry_tmp = hrry[$(idx)];
  hrrz_tmp = hrrz[$(idx)];
$py(_targets = [_i for _i,target in enumerate(rys_targets,start=0) if u_target in target])\
$for(_i in _targets)\
$py(need_x = bool(rys_targets[_i][0] == u_target) )\
$py(need_y = bool(rys_targets[_i][1] == u_target) )\
$py(need_z = bool(rys_targets[_i][2] == u_target) )\
$py(_exp_str = [])\
$py(if need_x: _exp_str.append('hrrx_tmp'))\
$py(if need_y: _exp_str.append('hrry_tmp'))\
$py(if need_z: _exp_str.append('hrrz_tmp'))\
$py(_exp_str = ' * '.join(_exp_str))\
$if(tgt_idx==0)\
  _rys_target_$(_i) = $(_exp_str);\
$else\
  _rys_target_$(_i) *= $(_exp_str);\
$endif
$endfor
$endfor
$for( i in range(len(rys_targets)) )\
$if(k==0)\
  result[$(i)] = beta * result[$(i)] + weights[$(k)] * _rys_target_$(i);
$else\
  result[$(i)] += weights[$(k)] * _rys_target_$(i);
$endif\
$endfor
$endfor
