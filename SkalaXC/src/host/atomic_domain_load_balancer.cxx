#include "atomic_domain_load_balancer.hpp"

#include "atomic_domain_assignment.hpp"
#include "saturating_math.hpp"

#include "load_balancer/load_balancer_impl.hpp"

#include <gauxc/util/geometry.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace SkalaXC::detail {
namespace {

/** @brief Basis-shell screening policy for atomic-domain task generation. */
enum class ScreeningMode { Petite, FillIn };

ScreeningMode parse_screening_mode(std::string kernel_name) {
  std::transform(kernel_name.begin(), kernel_name.end(), kernel_name.begin(),
                 [](unsigned char value) {
                   return static_cast<char>(std::toupper(value));
                 });
  if (kernel_name.empty() || kernel_name == "DEFAULT" ||
      kernel_name == "REPLICATED" || kernel_name == "REPLICATED-PETITE")
    return ScreeningMode::Petite;
  if (kernel_name == "REPLICATED-FILLIN") return ScreeningMode::FillIn;
  throw std::invalid_argument("Load Balancer Kernel Not Recognized: " +
                              kernel_name);
}

/** @brief Materialize complete atomic grids on their assigned MPI ranks. */
class AtomicDomainLoadBalancer final : public GauXC::detail::LoadBalancerImpl {
 public:
  AtomicDomainLoadBalancer(const GauXC::RuntimeEnvironment& runtime,
                           const GauXC::Molecule& molecule,
                           const GauXC::MolGrid& grid,
                           const GauXC::BasisSet<double>& basis,
                           ScreeningMode screening_mode)
      : LoadBalancerImpl(runtime, molecule, grid, basis),
        screening_mode_(screening_mode) {}

  std::unique_ptr<GauXC::detail::LoadBalancerImpl> clone() const override {
    return std::make_unique<AtomicDomainLoadBalancer>(*this);
  }

 private:
  std::pair<std::vector<std::int32_t>, std::size_t> screen(
      const std::array<double, 3>& box_lower,
      const std::array<double, 3>& box_upper) const {
    std::vector<std::int32_t> intersecting_shells;
    intersecting_shells.reserve(basis_->nshells());
    for (std::size_t shell = 0; shell < basis_->size(); ++shell) {
      const auto& basis_shell = (*basis_)[shell];
      if (GauXC::geometry::cube_sphere_intersect(box_lower, box_upper,
                                                 basis_shell.O(),
                                                 basis_shell.cutoff_radius()))
        intersecting_shells.push_back(static_cast<std::int32_t>(shell));
    }

    if (screening_mode_ == ScreeningMode::FillIn &&
        !intersecting_shells.empty()) {
      const auto first = intersecting_shells.front();
      const auto last = intersecting_shells.back();
      intersecting_shells.resize(static_cast<std::size_t>(last) -
                                 static_cast<std::size_t>(first) + 1);
      std::iota(intersecting_shells.begin(), intersecting_shells.end(), first);
    }

    const auto basis_functions = std::accumulate(
        intersecting_shells.begin(), intersecting_shells.end(), std::size_t{0},
        [this](std::size_t count, std::int32_t shell) {
          return count + (*basis_)[static_cast<std::size_t>(shell)].size();
        });
    return {std::move(intersecting_shells), basis_functions};
  }

  types::WorkEstimate atom_cost(types::AtomIndex atom_index) const {
    const auto& atom = (*mol_)[atom_index.raw()];
    const std::array<double, 3> center{atom.x, atom.y, atom.z};
    auto& batcher = mg_->get_grid(atom.Z).batcher();
    batcher.quadrature().recenter(center);
    const auto batch_count = batcher.nbatches();
    std::vector<types::WorkEstimate> batch_costs(batch_count);

#pragma omp parallel for
    for (std::size_t batch = 0; batch < batch_count; ++batch) {
      auto [lower, upper, points, weights] = batcher.at(batch);
      (void)weights;
      if (points.empty()) continue;
      const auto [shells, basis_functions] = screen(lower, upper);
      if (shells.empty()) continue;
      batch_costs[batch] = types::WorkEstimate{
          saturating_multiply(points.size(), basis_functions)};
    }

    types::WorkEstimate total{};
    for (const auto cost : batch_costs)
      total = types::WorkEstimate{saturating_add(total.raw(), cost.raw())};
    return total;
  }

  std::vector<GauXC::XCTask> materialize_atom(
      types::AtomIndex atom_index) const {
    const auto& atom = (*mol_)[atom_index.raw()];
    const std::array<double, 3> center{atom.x, atom.y, atom.z};
    auto& batcher = mg_->get_grid(atom.Z).batcher();
    batcher.quadrature().recenter(center);
    const auto batch_count = batcher.nbatches();
    std::vector<std::optional<GauXC::XCTask>> batch_tasks(batch_count);

#pragma omp parallel for
    for (std::size_t batch = 0; batch < batch_count; ++batch) {
      auto [lower, upper, points, weights] = batcher.at(batch);
      if (points.empty()) continue;
      auto [shells, basis_functions] = screen(lower, upper);
      if (shells.empty()) continue;

      GauXC::XCTask task;
      task.iParent = static_cast<std::int32_t>(atom_index.raw());
      task.npts = static_cast<std::int32_t>(points.size());
      task.points = std::move(points);
      task.weights = std::move(weights);
      task.bfn_screening.shell_list = std::move(shells);
      task.bfn_screening.nbe = static_cast<std::int32_t>(basis_functions);
      task.dist_nearest = molmeta_->dist_nearest()[atom_index.raw()];
      batch_tasks[batch] = std::move(task);
    }

    std::vector<GauXC::XCTask> tasks;
    tasks.reserve(batch_count);
    for (auto& task : batch_tasks)
      if (task) tasks.push_back(std::move(*task));
    return tasks;
  }

  std::vector<GauXC::XCTask> create_local_tasks_() const override {
    const auto atom_count = mol_->natoms();
    std::vector<types::WorkEstimate> atom_costs(atom_count);
    for (std::size_t atom = 0; atom < atom_count; ++atom)
      atom_costs[atom] = atom_cost(types::AtomIndex{atom});

    const auto assignment = assign_atomic_domains(
        atom_costs, types::CommunicatorSize{runtime_.comm_size()});

    const types::CommunicatorRank local_rank{runtime_.comm_rank()};
    std::vector<GauXC::XCTask> local_tasks;
    for (std::size_t atom = 0; atom < atom_count; ++atom) {
      if (assignment.owner_by_atom[atom] != local_rank) continue;
      auto atom_tasks = materialize_atom(types::AtomIndex{atom});
      local_tasks.insert(local_tasks.end(),
                         std::make_move_iterator(atom_tasks.begin()),
                         std::make_move_iterator(atom_tasks.end()));
    }

    std::stable_sort(local_tasks.begin(), local_tasks.end(),
                     [](const GauXC::XCTask& lhs, const GauXC::XCTask& rhs) {
                       if (lhs.iParent != rhs.iParent)
                         return lhs.iParent < rhs.iParent;
                       return lhs.bfn_screening.shell_list <
                              rhs.bfn_screening.shell_list;
                     });

    std::vector<GauXC::XCTask> merged_tasks;
    merged_tasks.reserve(local_tasks.size());
    for (auto& task : local_tasks) {
      if (merged_tasks.empty() || !merged_tasks.back().equiv_with(task))
        merged_tasks.push_back(std::move(task));
      else
        merged_tasks.back().merge_with(task);
    }
    return merged_tasks;
  }

  ScreeningMode screening_mode_;
};

}  // namespace

GauXC::LoadBalancer make_atomic_domain_load_balancer(
    const GauXC::RuntimeEnvironment& runtime, const GauXC::Molecule& molecule,
    const GauXC::MolGrid& grid, const GauXC::BasisSet<double>& basis,
    std::string kernel_name) {
  auto implementation = std::make_unique<AtomicDomainLoadBalancer>(
      runtime, molecule, grid, basis,
      parse_screening_mode(std::move(kernel_name)));
  return GauXC::LoadBalancer(std::move(implementation));
}

}  // namespace SkalaXC::detail