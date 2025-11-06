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

#include <gauxc/types.hpp>
#include <gauxc/gauxc_config.hpp>

namespace GauXC {

/// A named type pertaining to the atomic number (number of protons) of an Atom
using AtomicNumber = detail::NamedType< int64_t, struct AtomicNumberType >;

/**
 *  @brief A struct to represent the state of an atom (charge and spacial location)
 */
struct Atom {

  AtomicNumber Z; ///< Atomic number

  double x;       ///< X coordinate (bohr)
  double y;       ///< Y coordinate (bohr)
  double z;       ///< Z coordinate (bohr)

  /**
   *  @brief Construct an Atom object with default state
   */
  Atom() = default;

  /**
   *  @brief Construct an Atom object with a specified state
   *
   *  @param[in] _Z Atomic number
   *  @param[in] _x X coordinate (bohr)
   *  @param[in] _y Y coordinate (bohr)
   *  @param[in] _z Z coordinate (bohr)
   */
  Atom( AtomicNumber _Z, double _x, double _y, double _z ) :
    Z(_Z), x(_x), y(_y), z(_z) { }

  /**
   *  @brief (De)serialize an atom object to/from a particular cereal archive
   *
   *  @tparam Archive Cereal archive type
   *
   *  @param[in/out] ar Cereal archive
   */
  template <typename Archive>
  void serialize( Archive& ar ) {
    ar(  Z, x, y, z );
  }

}; // struct Atom

/**
 *  @brief Check equality of two Atom objects
 *
 *  @param[in] a1 First atom object
 *  @param[in] a2 Second atom object
 *  @returns   true if a1 and a2 represent identical atom objects,
 *             false otherwise
 */
inline bool operator==( const Atom& a1, const Atom& a2 ) {
  return a1.Z == a2.Z and a1.x == a2.x and a1.y == a2.y and a1.z == a2.z; 
}

/**
 *  @brief Check inequality of two Atom objects
 *
 *  @param[in] a1 First atom object
 *  @param[in] a2 Second atom object
 *  @returns   false if a1 and a2 represent identical atom objects,
 *             true otherwise
 */
inline bool operator!=( const Atom& a1, const Atom& a2 ) {
  return not (a1 == a2);
}

} // namespace GauXC
