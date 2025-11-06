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

#include <gauxc/molecule.hpp>

namespace GauXC {

class MolMeta {

  size_t              natoms_;
  std::vector<double> rab_;
  std::vector<double> dist_nearest_; 
  size_t              sum_atomic_charges_;

  void compute_rab(const Molecule&);
  void compute_dist_nearest();

public:

  MolMeta() = delete;
  MolMeta( const Molecule& );

  MolMeta( const MolMeta & );
  MolMeta( MolMeta&& ) noexcept;

  ~MolMeta() noexcept;

  size_t natoms() const { return natoms_; }

  const auto& rab()          const { return rab_; }
        auto& rab()                { return rab_; }

  const auto& dist_nearest() const { return dist_nearest_; }
        auto& dist_nearest()       { return dist_nearest_; }

  size_t sum_atomic_charges() const { return sum_atomic_charges_; }

  template <typename Archive>
  void serialize( Archive& ar ) {
    ar( natoms_, rab_, dist_nearest_ );
  }

};

}
