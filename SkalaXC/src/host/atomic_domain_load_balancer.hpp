#pragma once

#include <gauxc/load_balancer.hpp>

#include <string>

namespace GauXC {
class BasisSetMap;
class MolGrid;
class Molecule;
class RuntimeEnvironment;
template <typename>
class BasisSet;
}  // namespace GauXC

namespace SkalaXC::detail {

/**
 * @brief Build host-side tasks that keep each atomic grid on one MPI rank.
 *
 * The resulting GauXC task list is consumed by both host and CUDA local-work
 * drivers; only grid generation, screening, and assignment execute on the host.
 */
GauXC::LoadBalancer make_atomic_domain_load_balancer(
    const GauXC::RuntimeEnvironment& runtime, const GauXC::Molecule& molecule,
    const GauXC::MolGrid& grid, const GauXC::BasisSet<double>& basis,
    std::string kernel_name);

}  // namespace SkalaXC::detail