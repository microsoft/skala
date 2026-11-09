#include "atomic_domain_assignment.hpp"
#include "saturating_math.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace SkalaXC::detail {

AtomicDomainAssignment assign_atomic_domains(
    const std::vector<types::WorkEstimate>& atom_costs,
    types::CommunicatorSize rank_count) {
  if (rank_count.raw() <= 0)
    throw std::invalid_argument("rank_count must be positive");

  AtomicDomainAssignment result;
  result.owner_by_atom.resize(atom_costs.size());
  result.rank_costs.resize(static_cast<std::size_t>(rank_count.raw()));

  std::vector<std::size_t> atom_order(atom_costs.size());
  std::iota(atom_order.begin(), atom_order.end(), std::size_t{0});
  std::stable_sort(atom_order.begin(), atom_order.end(),
                   [&atom_costs](std::size_t lhs, std::size_t rhs) {
                     return atom_costs[lhs].raw() > atom_costs[rhs].raw();
                   });

  for (const auto atom : atom_order) {
    const auto rank = types::CommunicatorRank{static_cast<int>(std::distance(
        result.rank_costs.begin(),
        std::min_element(
            result.rank_costs.begin(), result.rank_costs.end(),
            [](types::WorkEstimate left, types::WorkEstimate right) {
              return left.raw() < right.raw();
            })))};
    result.owner_by_atom[atom] = rank;

    const auto cost = atom_costs[atom];
    auto& rank_cost = result.rank_costs[static_cast<std::size_t>(rank.raw())];
    rank_cost =
        types::WorkEstimate{saturating_add(rank_cost.raw(), cost.raw())};
  }

  return result;
}

}  // namespace SkalaXC::detail