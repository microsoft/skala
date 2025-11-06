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
#include "host/blas.hpp"
#include <vector>
#include <tuple>
#include <cstdint>

namespace GauXC  {
namespace detail {

template <typename _F1, typename _F2>
void submat_set(int32_t M, int32_t N, int32_t MSub, 
  int32_t NSub, _F1 *ABig, int32_t LDAB, _F2 *ASmall, 
  int32_t LDAS, 
  const std::vector<std::array<int32_t,3>> &submat_map_rows,
  const std::vector<std::array<int32_t,3>> &submat_map_cols) {

  (void)(M);
  (void)(N);
  (void)(MSub);
  (void)(NSub);

  int32_t i(0);
  for( auto& iCut : submat_map_rows ) {
    int32_t deltaI = iCut[1];
    int32_t j(0);
  for( auto& jCut : submat_map_cols ) {
    int32_t deltaJ = jCut[1];
  
    auto* ABig_use   = ABig   + iCut[0] + jCut[0] * LDAB;
    auto* ASmall_use = ASmall + i       + j       * LDAS;


    GauXC::blas::lacpy( 'A', deltaI, deltaJ, ABig_use, LDAB, 
                         ASmall_use, LDAS );

  
    j += deltaJ;
  }
    i += deltaI;
  }
  
}

template <typename _F1, typename _F2>
void submat_set(int32_t M, int32_t N, int32_t MSub, 
  int32_t NSub, _F1 *ABig, int32_t LDAB, _F2 *ASmall, 
  int32_t LDAS, 
  const std::vector<std::array<int32_t,3>> &submat_map ) {

  submat_set(M, N, MSub, NSub, ABig, LDAB, ASmall, LDAS,
    submat_map, submat_map );

}

#if 0
template <typename _F1, typename _F2>
void submat_set_row_pack(int32_t M, int32_t N, int32_t MSub, 
  int32_t NSub, _F1 *ABig, int32_t LDAB, _F2 *ASmall, 
  int32_t LDAS, 
  const std::vector<std::array<int32_t,3>> &submat_map ) {

  decltype(submat_map) col_map = { { 0, N, 0 } };

  submat_set(M, N, MSub, NSub, ABig, LDAB, ASmall, LDAS,
    submat_map, col_map );

}

template <typename _F1, typename _F2>
void submat_set_col_pack(int32_t M, int32_t N, int32_t MSub, 
  int32_t NSub, _F1 *ABig, int32_t LDAB, _F2 *ASmall, 
  int32_t LDAS, 
  const std::vector<std::array<int32_t,3>> &submat_map ) {

  decltype(submat_map) row_map = { { 0, M, 0 } };
  submat_set(M, N, MSub, NSub, ABig, LDAB, ASmall, LDAS,
    row_map, submat_map );

}
#endif

template <typename _F1, typename _F2>
void inc_by_submat(int32_t M, int32_t N, int32_t MSub, 
  int32_t NSub, _F1 *ABig, int32_t LDAB, _F2 *ASmall, 
  int32_t LDAS, 
  const std::vector<std::array<int32_t,3>> &submat_map_row,
  const std::vector<std::array<int32_t,3>> &submat_map_col) {

  (void)(M);
  (void)(N);
  (void)(MSub);
  (void)(NSub);

  int32_t i(0);
  for( auto& iCut : submat_map_row ) {
    int32_t deltaI = iCut[1];
    int32_t j(0);
  for( auto& jCut : submat_map_col ) {
    int32_t deltaJ = jCut[1];
  
    auto* ABig_use   = ABig   + iCut[0] + jCut[0] * LDAB;
    auto* ASmall_use = ASmall + i       + j       * LDAS;


    for( int32_t jj = 0; jj < deltaJ; ++jj )
    for( int32_t ii = 0; ii < deltaI; ++ii ) {
      ABig_use[ ii + jj * LDAB ] += ASmall_use[ ii + jj * LDAS ];
    }

  
    j += deltaJ;
  }
    i += deltaI;
  }
  

}

template <typename _F1, typename _F2>
void inc_by_submat_atomic(int32_t M, int32_t N, int32_t MSub, 
  int32_t NSub, _F1 *ABig, int32_t LDAB, _F2 *ASmall, 
  int32_t LDAS, 
  const std::vector<std::array<int32_t,3>> &submat_map_row,
  const std::vector<std::array<int32_t,3>> &submat_map_col) {

  (void)(M);
  (void)(N);
  (void)(MSub);
  (void)(NSub);

  int32_t i(0);
  for( auto& iCut : submat_map_row ) {
    int32_t deltaI = iCut[1];
    int32_t j(0);
  for( auto& jCut : submat_map_col ) {
    int32_t deltaJ = jCut[1];
  
    auto* ABig_use   = ABig   + iCut[0] + jCut[0] * LDAB;
    auto* ASmall_use = ASmall + i       + j       * LDAS;


    for( int32_t jj = 0; jj < deltaJ; ++jj )
    for( int32_t ii = 0; ii < deltaI; ++ii ) {
      #ifdef _OPENMP
      #pragma omp atomic
      #endif
      ABig_use[ ii + jj * LDAB ] += ASmall_use[ ii + jj * LDAS ];
    }

  
    j += deltaJ;
  }
    i += deltaI;
  }
  

}


template <typename _F1, typename _F2>
void inc_by_submat(int32_t M, int32_t N, int32_t MSub, 
  int32_t NSub, _F1 *ABig, int32_t LDAB, _F2 *ASmall, 
  int32_t LDAS, 
  const std::vector<std::array<int32_t,3>> &submat_map ) {

  inc_by_submat(M,N,MSub,NSub, ABig, LDAB, ASmall, LDAS,
    submat_map, submat_map );

}

template <typename _F1, typename _F2>
void inc_by_submat_atomic(int32_t M, int32_t N, int32_t MSub, 
  int32_t NSub, _F1 *ABig, int32_t LDAB, _F2 *ASmall, 
  int32_t LDAS, 
  const std::vector<std::array<int32_t,3>> &submat_map) {

  inc_by_submat_atomic(M,N,MSub,NSub, ABig, LDAB, ASmall, LDAS,
    submat_map, submat_map );

}

}
}
