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

#include <gauxc/basisset.hpp>
#include <gauxc/molecule.hpp>

namespace GauXC {

/// A class to manage index maps pertaining to a BasisSet object
class BasisSetMap {

  using ao_range    = std::pair<int32_t,int32_t>;
  using shell_range = std::pair<int32_t, int32_t>;

  int32_t nshells_; ///< Number of basis shells

  std::vector<int32_t>      shell_sizes_;       ///< Shell sizes
  std::vector<int32_t>      shell_ls_;          ///< Shell angular momenta
  std::vector<bool>         shell_pure_;        ///< Shell purity (cart/sph)
  std::vector<int32_t>      shell_to_first_ao_; ///< Map from shell index to first basis function of that shell
  std::vector<ao_range>     shell_to_ao_range_; ///< Map from shell index to range of basis functions for that shell
  std::vector<int32_t>      shell_to_center_;   ///< Map from shell index to basis center (atom in a moledule)
  std::vector<shell_range>  center_to_shell_range_; ///< Map from unique basis center to range of shells
 
public:

  /**
   *  @brief Construct a BasisSetMap object from a BasisSet
   *
   *  Generate the maps from shell indices to basis function and center indices
   *
   *  @param[in] basis BasisSet for which to generate maps
   *  @param[in] mol   Molecule instatance upon which `basis` is defined.
   */
  template <typename F>
  BasisSetMap( const BasisSet<F>& basis, const Molecule& mol ) :
    nshells_( basis.nshells() ) {	

    shell_sizes_.resize( nshells_ );
    shell_ls_.resize( nshells_ );
    shell_pure_.resize( nshells_ );
    for( int32_t i = 0; i < nshells_; ++i ) {
      shell_sizes_[i] = basis.at(i).size();
      shell_ls_[i]    = basis.at(i).l(); 
      shell_pure_[i]  = basis.at(i).pure();
    }

    shell_to_first_ao_.reserve( nshells_ );
    shell_to_ao_range_.reserve( nshells_ );

    size_t st_idx = 0;
    for( const auto& shell : basis ) {
      size_t range_end = st_idx + shell.size();
      shell_to_first_ao_.emplace_back( st_idx );
      shell_to_ao_range_.push_back({ st_idx, range_end });
      st_idx = range_end;
    }

    shell_to_center_.resize( nshells_ );
    size_t sh_idx = 0;
    for( const auto& shell : basis ) {
      auto at_pos = std::find_if( mol.begin(), mol.end(), [&](const Atom& at) { 
        return at.x == shell.O()[0] and at.y == shell.O()[1] and at.z == shell.O()[2];
      });
      if( at_pos != mol.end() ) shell_to_center_[sh_idx] = std::distance( mol.begin(), at_pos );
      else shell_to_center_[sh_idx] = -1;
      ++sh_idx;
    }

  }



  /// Return the map from shell indicies to starting basis functions (const)
  const auto& shell_to_first_ao() const { return shell_to_first_ao_; }
  
  /// Return the map from shell indicies to starting basis functions (non-const)
  auto& shell_to_first_ao()       { return shell_to_first_ao_; }

  /// Return the map from shell indicies to basis function ranges (const)
  const auto& shell_to_ao_range() const { return shell_to_ao_range_; }
  
  /// Return the map from shell indicies to basis function ranges (non-const)
  auto& shell_to_ao_range() { return shell_to_ao_range_; }

  /// Return container that stores the shell sizes (const)
  const auto& shell_sizes() const { return shell_sizes_; }

  /// Return container that stores the shell sizes (non-const)
  auto& shell_sizes() { return shell_sizes_; }

  /// Return map from shell indices to unique basis centers (const)
  const auto& shell_to_center() const { return shell_to_center_; }

  /// Return map from shell indices to unique basis centers (non-const)
  auto& shell_to_center() { return shell_to_center_; }


  /**
   *  @brief Get first basis function index for a specified shell
   *
   *  @param[in] i Shell index
   *  @returns   Basis function index for shell "i"
   */
  auto shell_to_first_ao(int32_t i) const { return shell_to_first_ao_.at(i); }

  /**
   *  @brief Get first basis function range for a specified shell
   *
   *  @param[in] i Shell index
   *  @returns   Basis function range for shell "i"
   */
  auto shell_to_ao_range(int32_t i) const { return shell_to_ao_range_.at(i); }

  /**
   *  @brief Get the size of shell "i"
   *
   *  @param[in] i Shell index
   *  @returns   Size of shell "i"
   */
  auto shell_size(int32_t i) const { return shell_sizes_.at(i); }

  /**
   *  @brief Get unique basis center index for shell "i"
   *
   *  @param[in] i Shell index
   *  @returns   Basis center index for shell "i"
   */
  const auto& shell_to_center(int32_t i) const { return shell_to_center_[i]; }

  /**
   *  @brief Get unique basis center index for shell "i"
   *
   *  @param[in] i Shell index
   *  @returns   Basis center index for shell "i"
   */
  auto& shell_to_center(int32_t i) { return shell_to_center_[i]; }


  /**
   *  @brief Get angular momentum for shell "i"
   *
   *  @param[in] i Shell index
   *  @returns   Angular momentum for shell "i"
   */
  auto shell_l(size_t i) const { return shell_ls_.at(i); }

  /**
   *  @brief Get purity (cart/sph) of shell "i"
   *
   *  @param[in] i Shell index
   *  @returns   Purity of shell "i"
   */
  auto shell_pure(size_t i) const { return shell_pure_.at(i); }

  /// Get max angular momentum for the basis set
  inline uint32_t max_l() const {
    return *std::max_element(shell_ls_.begin(), shell_ls_.end());
  }

  /// Count the number of shells with angular momentum `l`
  inline size_t nshells_with_l(uint32_t l) const {
    return std::count( shell_ls_.begin(), shell_ls_.end(), l );
  }

  /// Check whether shells of angular momentum `l` are spherical (pure)
  inline bool l_purity(uint32_t l) const {
    // Find first shell with L
    auto first_shell_w_l = std::find( shell_ls_.begin(), shell_ls_.end(), l );
    return shell_pure( std::distance( shell_ls_.begin(), first_shell_w_l ) );
  }

  /// Obtain list of shell offsets for a shell subset if compressed in memory 
  template <typename IntegralType, typename IntegralIterator>
  std::vector<IntegralType> shell_offs( IntegralIterator begin, 
                                        IntegralIterator end ) const {

    const size_t nshells_list = std::distance(begin,end);
    std::vector<IntegralType> shell_offs(nshells_list);
    shell_offs.at(0) = 0;
    for(auto i = 1ul; i < nshells_list; ++i) {
      shell_offs.at(i) = shell_offs.at(i-1) + shell_size(*(begin+i-1));
    }
    return shell_offs;  

  }
}; // class BasisSetMap

} // namespace GauXC
