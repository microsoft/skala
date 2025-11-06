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

#include <cereal/cereal.hpp>

//#ifdef __PGI
//  #define _GAUXC_COMP_IS_PGI
//  #undef __PGI
//#endif
#define EIGEN_DONT_VECTORIZE
#define EIGEN_NO_CUDA
#include <Eigen/Core>
//#ifdef _GAUXC_COMP_IS_PGI
//  #define __PGI
//#endif

namespace cereal {

template <typename Archive, typename T, int _Rows, int _Cols, int _Opts,
          int _MaxRows, int _MaxCols>
inline std::enable_if_t<
  traits::is_output_serializable< BinaryData<T>, Archive>::value and
  std::is_arithmetic<T>::value and not std::is_same<T, bool>::value
> CEREAL_SAVE_FUNCTION_NAME( 
    Archive &ar, 
    const Eigen::Matrix<T,_Rows,_Cols,_Opts,_MaxRows,_MaxCols>& mat
) {

  //ar( _Rows, _Cols, _Opts, _MaxRows, _MaxCols );
  int32_t rows = mat.rows();
  int32_t cols = mat.cols();
  ar( rows, cols );
  ar( binary_data( mat.data(), static_cast<std::size_t>(rows * cols * sizeof(T)) ));

}



template <typename Archive, typename T, int _Rows, int _Cols, int _Opts,
          int _MaxRows, int _MaxCols>
inline std::enable_if_t<
  traits::is_input_serializable< BinaryData<T>, Archive>::value and
  std::is_arithmetic<T>::value and not std::is_same<T, bool>::value
> CEREAL_LOAD_FUNCTION_NAME( 
    Archive &ar, 
    Eigen::Matrix<T,_Rows,_Cols,_Opts,_MaxRows,_MaxCols>& mat
) {

  //ar( Rows, Cols, Opts, MaxRows, MaxCols );

  int32_t rows;
  int32_t cols;
  ar( rows, cols );

  mat.resize( rows, cols );

  ar( binary_data( mat.data(), static_cast<std::size_t>(rows * cols * sizeof(T)) ));

}

}
