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
#include <gauxc/external/hdf5.hpp>
#include <highfive/H5File.hpp>
#include <gauxc/shell.hpp>
#include <gauxc/atom.hpp>

namespace GauXC {

struct shell_t {
  int32_t nprim, l, pure;
  Shell<double>::prim_array alpha, coeff;
  Shell<double>::cart_array O;
};

inline hid_t create_shell_type() {

  hsize_t prim_dims[1] = {16};
  hsize_t cart_dims[1] = {3};
  auto prim_array_type = H5Tarray_create( H5T_NATIVE_DOUBLE, 1, prim_dims );
  auto cart_array_type = H5Tarray_create( H5T_NATIVE_DOUBLE, 1, cart_dims );

  auto shell_type = H5Tcreate( H5T_COMPOUND, sizeof(shell_t) );
  H5Tinsert( shell_type, "NPRIM",  HOFFSET( shell_t, nprim ), H5T_NATIVE_INT );
  H5Tinsert( shell_type, "L",      HOFFSET( shell_t, l ),     H5T_NATIVE_INT );
  H5Tinsert( shell_type, "PURE",   HOFFSET( shell_t, pure ),  H5T_NATIVE_INT );
  H5Tinsert( shell_type, "ALPHA",  HOFFSET( shell_t, alpha),  prim_array_type );
  H5Tinsert( shell_type, "COEFF",  HOFFSET( shell_t, coeff),  prim_array_type );
  H5Tinsert( shell_type, "ORIGIN", HOFFSET( shell_t, O),      cart_array_type );
  
  return shell_type;
}


inline hid_t create_atom_type() {

  auto atom_type = H5Tcreate( H5T_COMPOUND, sizeof(Atom) );
  H5Tinsert( atom_type, "Atomic Number", HOFFSET( Atom, Z), H5T_NATIVE_INT    );
  H5Tinsert( atom_type, "X Coordinate", HOFFSET( Atom, x ), H5T_NATIVE_DOUBLE );
  H5Tinsert( atom_type, "Y Coordinate", HOFFSET( Atom, y ), H5T_NATIVE_DOUBLE );
  H5Tinsert( atom_type, "Z Coordinate", HOFFSET( Atom, z ), H5T_NATIVE_DOUBLE );

  return atom_type;

}

}
