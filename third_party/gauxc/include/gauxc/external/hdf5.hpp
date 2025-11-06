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
#include <gauxc/gauxc_config.hpp>
#ifdef GAUXC_HAS_HDF5
#include <gauxc/shell.hpp>
#include <gauxc/atom.hpp>

namespace GauXC {
void write_hdf5_record( const std::vector<Shell<double>>& shell, std::string fname, std::string dset );
void write_hdf5_record( const std::vector<Atom>& mol, std::string fname, std::string dset );
void read_hdf5_record( std::vector<Shell<double>>& shell, std::string fname, std::string dset );
void read_hdf5_record( std::vector<Atom>& mol, std::string fname, std::string dset );

#if 0
void write_hdf5_record( int32_t M, int32_t N, const double* A, int32_t LDA, std::string fname, std::string dset );
void read_hdf5_record( int32_t M, int32_t N, double* A, int32_t LDA, std::string fname, std::string dset );

#if __has_include(<Eigen/Core>)
template <typename Derived>
inline void write_hdf5_record( const Eigen::MatrixBase<Derived>& A, std::string fname, std::string dset ) {
  write_hdf5_record( A.rows(), A.cols(), A.data(), A.rows(), fname, dset );
}
template <typename Derived>
inline void read_hdf5_record( Eigen::MatrixBase<Derived>& A, std::string fname, std::string dset ) {
  read_hdf5_record( A.rows(), A.cols(), A.data(), A.rows(), fname, dset );
}
#endif
#endif

}
#endif
