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
#include <gauxc/grid.hpp>
#include <integratorxx/composite_quadratures/spherical_quadrature.hpp>
#include <integratorxx/composite_quadratures/pruned_spherical_quadrature.hpp>

#include <variant>

namespace GauXC {

/// Generic specification of an unpruned atomic quadrature
struct UnprunedAtomicGridSpecification {
  RadialQuad  radial_quad;  ///< Radial quadrature specification
  RadialSize  radial_size;  ///< Number of radial quadrature points
  RadialScale radial_scale; ///< Radial scaling factor

  AngularSize angular_size; /// Number of angular quadrature points
};

/// Speficiation of a pruned region of an atomic quadrature
struct PruningRegion {
  size_t idx_st;             ///< Starting radial index for pruned region
  size_t idx_en;             ///< Ending radial index (exclusive) for the pruned region
  AngularSize angular_size;  ///< Number of angular quadrature points in the pruned region

  /// Check equality of `PruningRegion` instances
  bool operator==(const PruningRegion& other) const {
    return other.idx_st == idx_st and 
           other.idx_en == idx_en and
           other.angular_size == angular_size;
  }
};

/// Generic specification of a pruned atomic quadrature
struct PrunedAtomicGridSpecification {
  RadialQuad  radial_quad;  ///< Radial quadrature specification
  RadialSize  radial_size;  ///< Number of radial quadrature points
  RadialScale radial_scale; ///< Radial scaling factor

  std::vector<PruningRegion> pruning_regions; ///< List of pruning regions over the radial quadrature
};


/// Generate a "Robust"-Psi4 Pruning specification from an unpruned quadrature specification
PrunedAtomicGridSpecification robust_psi4_pruning_scheme(
  UnprunedAtomicGridSpecification
);

/// Generate a Pruning specification according to the Treutler-Ahlrichs scheme from an unpruned specification
PrunedAtomicGridSpecification treutler_pruning_scheme(
  UnprunedAtomicGridSpecification
);

/// High-level specification of pruning schemes for atomic quadratures
enum class PruningScheme {
  Unpruned, /// Unpruned atomic quadrature
  Robust,   /// The "Robust" scheme of Psi4
  Treutler  /// The Treutler-Ahlrichs scheme
};

/// Generate a pruning specification from a specificed pruning scheme and 
/// an unpruned grid specification
PrunedAtomicGridSpecification create_pruned_spec(
  PruningScheme, UnprunedAtomicGridSpecification
);

using atomic_grid_variant = 
  std::variant<UnprunedAtomicGridSpecification,
               PrunedAtomicGridSpecification>;


/// Factory for Atomic grids
struct AtomicGridFactory {

  template <typename RadialType, typename AngularType>
  using unpruned_sphere_type = typename
    IntegratorXX::SphericalQuadrature< 
      std::decay_t<RadialType>, std::decay_t<AngularType>
    >;
  template <typename RadialType, typename AngularType>
  using pruned_sphere_type = typename
    IntegratorXX::PrunedSphericalQuadrature< 
      std::decay_t<RadialType>, std::decay_t<AngularType>
    >;

  /**
   *  @brief Generate an unpruned atomic grid given a suppled radial and
   *         angular quadrature.
   *
   *  All arguments are passed with perfect forwarding
   *
   *  @tparam RadialType Type of the radial quadrature
   *  @tparam AngularType Type of the angular quadrature
   * 
   *  @oaram     rq Radial quadrature from which to construct the atomic quadrature. 
   *  @oaram     aq Angular quadrature from which to construct the atomic quadrature. 
   *  @param[in] bsz Batch size for the grid generation.
   */
  template <typename RadialType, typename AngularType>
  static Grid generate_unpruned_grid( RadialType&& rq, AngularType&& aq, BatchSize bsz ) {
    using sphere_type = unpruned_sphere_type<RadialType,AngularType>;
    return Grid( std::make_shared<sphere_type>( 
      std::forward<RadialType>(rq), std::forward<AngularType>(aq) 
      ), bsz
    );
  }

  template <typename RadialType, typename RadialPartitionType>
  static Grid generate_pruned_grid( RadialType&& rq, 
    RadialPartitionType&& rgp, BatchSize bsz ) {
    using angular_type = typename std::decay_t<RadialPartitionType>::angular_type; 
    using sphere_type = pruned_sphere_type<RadialType,angular_type>;
    return Grid( std::make_shared<sphere_type>( 
      std::forward<RadialType>(rq), std::forward<RadialPartitionType>(rgp)
      ), bsz
    );
  }


  static Grid generate_unpruned_grid( RadialQuad, RadialSize, AngularSize, 
                                      RadialScale, BatchSize bsz );
  static Grid generate_pruned_grid( RadialQuad, RadialSize, 
    const std::vector<PruningRegion>&, RadialScale, BatchSize bsz );


  static Grid generate_grid( UnprunedAtomicGridSpecification gs, BatchSize bsz ); 
  static Grid generate_grid( PrunedAtomicGridSpecification gs, BatchSize bsz ); 

  static Grid generate_grid( atomic_grid_variant gs, BatchSize bsz );

};

}
