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
#include "hdf5_util.hpp"
#include <gauxc/exceptions.hpp>

namespace GauXC {

using namespace HighFive;


void write_hdf5_record( const std::vector<Shell<double>>& basis, std::string fname, 
  std::string dset ) {


  File file( fname, File::OpenOrCreate );
  
  auto shell_type = create_shell_type();

  DataSpace space(basis.size());
  auto d_id = H5Dcreate( file.getId(), dset.c_str(), shell_type, space.getId(),
    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );

  if( d_id < 0 ) GAUXC_GENERIC_EXCEPTION("Dataset Creation Failed");

  std::vector<shell_t> shells;
  for( auto& shell : basis ) {
    shells.push_back(
      shell_t{
        shell.nprim(), shell.l(), shell.pure(),
        shell.alpha(), shell.coeff(), shell.O()
      });
  }

  H5Dwrite( d_id, shell_type, space.getId(), space.getId(), H5P_DEFAULT, 
    shells.data() );

  H5Tclose( shell_type );
  H5Dclose( d_id );


}




void write_hdf5_record( const std::vector<Atom>& mol, std::string fname, std::string dset ) {

  File file( fname, File::OpenOrCreate );
  
  auto atom_type = create_atom_type();

  DataSpace space(mol.size());
  auto d_id = H5Dcreate( file.getId(), dset.c_str(), atom_type, space.getId(),
    H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );

  if( d_id < 0 ) GAUXC_GENERIC_EXCEPTION("Dataset Creation Failed");

  H5Dwrite( d_id, atom_type, space.getId(), space.getId(), H5P_DEFAULT, 
    mol.data() );

  H5Tclose( atom_type );
  H5Dclose( d_id );

}


}
