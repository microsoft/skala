#pragma once

#include "index_types.hpp"

#include <cstdint>
#include <vector>

namespace SkalaXC::detail {

/** @brief Deterministic atom ownership and accumulated cost for each rank. */
struct AtomicDomainAssignment {
  std::vector<types::CommunicatorRank> owner_by_atom;  ///< Runtime rank per
                                                       ///< atom.
  std::vector<types::WorkEstimate> rank_costs;  ///< Saturating cost per rank.
};

/** @brief Assign descending-cost atoms to the least-loaded runtime rank. */
AtomicDomainAssignment assign_atomic_domains(
    const std::vector<types::WorkEstimate>& atom_costs,
    types::CommunicatorSize rank_count);

}  // namespace SkalaXC::detail