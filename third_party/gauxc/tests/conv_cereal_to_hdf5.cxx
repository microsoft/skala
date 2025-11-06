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
#include <gauxc/external/cereal.hpp>
#include <gauxc/external/hdf5.hpp>
#include <cereal/archives/binary.hpp>
#include "eigen3_matrix_serialization.hpp"
#include <highfive/H5File.hpp>
#include "standards.hpp"
#include "basis/parse_basis.hpp"
#include <fstream>

using namespace GauXC;
int main( int argc, char** argv ) {

  std::vector< std::string > opts( argc );
  for( int i = 0; i < argc; ++i ) opts[i] = argv[i];

  std::string test_case   = opts.at(1);
  std::string basis_set   = opts.at(2);
  std::string cereal_file = opts.at(3);
  std::string hdf5_file   = opts.at(4);

  // Construct Molecule
  Molecule mol;
  if( test_case.find("benzene") != std::string::npos )
    mol = make_benzene();
  else if( test_case.find("water") != std::string::npos )
    mol = make_water();
  else if( test_case.find("taxol") != std::string::npos )
    mol = make_taxol();
  else if( test_case.find("ubiquitin") != std::string::npos )
    mol = make_ubiquitin();
  else
    throw std::runtime_error("Unknown Test Case");

  // Construct BasisSet
  BasisSet<double> basis; 
  if( basis_set.find("6-31gd") != std::string::npos ) 
    basis = std::move(make_631Gd( mol, SphericalType(false) ));
  else if( basis_set.find("cc-pvdz") != std::string::npos ) 
    basis = std::move(make_ccpvdz( mol, SphericalType(true) ));
  else
    throw std::runtime_error("Unknown Basis Set");

  // Read in cereal file
    using matrix_type = Eigen::MatrixXd;
  matrix_type P,VXC_ref;
  double EXC_ref;
  {
    std::ifstream infile( cereal_file, std::ios::binary );

    if( !infile.good() ) throw std::runtime_error(cereal_file + " not found");
    cereal::BinaryInputArchive ar(infile);
    ar( EXC_ref, P, VXC_ref );
  }

  // Write HDF5 file
  write_hdf5_record( mol,   hdf5_file, "/MOLECULE" );
  write_hdf5_record( basis, hdf5_file, "/BASIS"    );
  {
    using namespace HighFive;
    File file( hdf5_file, File::ReadWrite );
    DataSpace space( P.rows(), P.cols() );
    DataSet den = file.createDataSet<double>( "/DENSITY", space );
    den.write_raw( P.data() );
    DataSet vxc = file.createDataSet<double>( "/VXC", space );
    vxc.write_raw( VXC_ref.data() );

    DataSpace singleton(1);
    DataSet exc = file.createDataSet<double>("/EXC", singleton );
    exc.write_raw( &EXC_ref );
  }
}
