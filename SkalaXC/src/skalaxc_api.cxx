/**
 * SkalaXC public C++ API implementation.
 *
 * This translation unit is the ONLY place (together with the C wrapper) where
 * the public SkalaXC API meets GauXC / LibTorch. The public pipeline mirrors
 * GauXC (RuntimeEnvironment -> MolGrid -> LoadBalancer -> MolecularWeights ->
 * XCIntegratorFactory -> XCIntegrator) but every stage owns its GauXC state
 * behind a PIMPL, and GauXC exceptions are translated to SkalaXC::Exception so
 * no GauXC type escapes the boundary.
 */
#include <skalaxc/skalaxc.hpp>
#include <skalaxc/skalaxc_config.hpp>

#include <gauxc/atom.hpp>
#include <gauxc/basisset.hpp>
#include <gauxc/enums.hpp>
#ifdef SKALAXC_HAS_HDF5
#include <gauxc/external/hdf5.hpp>
#endif
#include <gauxc/load_balancer.hpp>
#include <gauxc/molecular_weights.hpp>
#include <gauxc/molecule.hpp>
#include <gauxc/molgrid.hpp>
#include <gauxc/molgrid/defaults.hpp>
#include <gauxc/runtime_environment.hpp>
#include <gauxc/shell.hpp>
#include <gauxc/util/mpi.hpp>
#include <gauxc/xc_integrator/local_work_driver.hpp>
#include <gauxc/xc_task.hpp>

// GauXC reusable internal (host local work driver: partition_weights).
// Reachable via the in-tree `gauxc` target's PUBLIC BUILD_INTERFACE include of
// ${GauXC}/src.
#include "xc_integrator/local_work_driver/host/local_host_work_driver.hpp"

#include "host/atomic_domain_load_balancer.hpp"
#include "host/eigen_types.hpp"
#include "host/skala_host_driver.hpp"
#include "skala_driver.hpp"

#ifdef SKALAXC_HAS_CUDA
#include "device/skala_device_driver.hpp"
#endif

#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#ifdef SKALAXC_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

namespace SkalaXC {

std::string_view version() noexcept { return SKALAXC_VERSION_STRING; }

// ===========================================================================
// PIMPL definitions: each pipeline stage owns its GauXC state.
// ===========================================================================
struct RuntimeEnvironment::Impl {
  GauXC::RuntimeEnvironment rt;
  ExecutionSpace ex = ExecutionSpace::Host;
  types::DeviceId device_id{-1};
  double device_memory_fraction = 0.0;
  explicit Impl(SKALAXC_MPI_CODE(MPI_Comm comm)) : rt{GAUXC_MPI_CODE(comm)} {}
#ifdef SKALAXC_HAS_CUDA
  Impl(SKALAXC_MPI_CODE(MPI_Comm comm, ) DeviceRuntimeSettings settings)
      : rt{GauXC::DeviceRuntimeEnvironment(SKALAXC_MPI_CODE(comm, )
                                               settings.memory_fraction)},
        ex(ExecutionSpace::Device),
        device_id(types::DeviceId{settings.device_id}),
        device_memory_fraction(settings.memory_fraction) {}
#endif
};

struct MolGrid::Impl {
  GauXC::MolGrid mg;
  explicit Impl(GauXC::MolGrid grid) : mg(std::move(grid)) {}
};

struct LoadBalancer::Impl {
  GauXC::LoadBalancer lb;
  ExecutionSpace ex = ExecutionSpace::Host;
  types::DeviceId device_id{-1};
  double device_memory_fraction = 0.0;
  /// Pre-partition ("raw") quadrature weights, captured by modify_weights.
  std::vector<std::vector<double>> raw_weights;
  Impl(GauXC::LoadBalancer balancer, ExecutionSpace execution_space,
       types::DeviceId selected_device, double selected_memory_fraction)
      : lb(std::move(balancer)),
        ex(execution_space),
        device_id(selected_device),
        device_memory_fraction(selected_memory_fraction) {}
};

struct MolecularWeights::Impl {
  ExecutionSpace ex = ExecutionSpace::Host;
  std::string kernel;
  MolecularWeightsSettings settings;
};

namespace detail {

struct IntegratorCore::Impl {
  std::unique_ptr<SkalaDriver> driver;
  std::int64_t nbf = 0;
  std::int64_t natoms = 0;
};

/// Grants this implementation TU access to the opaque pipeline-stage internals.
struct Access {
  static RuntimeEnvironment::Impl* impl(RuntimeEnvironment& r) {
    return r.pimpl_.get();
  }
  static const RuntimeEnvironment::Impl* impl(const RuntimeEnvironment& r) {
    return r.pimpl_.get();
  }
  static MolGrid::Impl* impl(MolGrid& m) { return m.pimpl_.get(); }
  static const MolGrid::Impl* impl(const MolGrid& m) { return m.pimpl_.get(); }
  static LoadBalancer::Impl* impl(LoadBalancer& l) { return l.pimpl_.get(); }
  static const LoadBalancer::Impl* impl(const LoadBalancer& l) {
    return l.pimpl_.get();
  }
  static MolecularWeights::Impl* impl(MolecularWeights& w) {
    return w.pimpl_.get();
  }
  static const MolecularWeights::Impl* impl(const MolecularWeights& w) {
    return w.pimpl_.get();
  }

  static MolGrid make_molgrid(std::unique_ptr<MolGrid::Impl> p) {
    return MolGrid(std::move(p));
  }
  static LoadBalancer make_load_balancer(
      std::unique_ptr<LoadBalancer::Impl> p) {
    return LoadBalancer(std::move(p));
  }
  static MolecularWeights make_molecular_weights(
      std::unique_ptr<MolecularWeights::Impl> p) {
    return MolecularWeights(std::move(p));
  }
};

}  // namespace detail

namespace {

#ifdef SKALAXC_HAS_CUDA
void activate_cuda_device(types::DeviceId device_id) {
  int device_count = 0;
  auto status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess)
    throw Exception(std::string("Failed to query CUDA devices: ") +
                    cudaGetErrorString(status));
  if (device_id.raw() < 0 || device_id.raw() >= device_count)
    throw Exception("CUDA device_id is outside the available device range");
  status = cudaSetDevice(device_id.raw());
  if (status != cudaSuccess)
    throw Exception(std::string("Failed to select CUDA device: ") +
                    cudaGetErrorString(status));
}
#endif

// ---- SkalaXC value type -> GauXC conversions -----------------------------
GauXC::Molecule to_gauxc(const Molecule& mol) {
  GauXC::Molecule out;
  out.reserve(mol.size());
  for (const auto& a : mol)
    out.push_back(GauXC::Atom{GauXC::AtomicNumber(a.Z.raw()), a.x, a.y, a.z});
  return out;
}

GauXC::BasisSet<double> to_gauxc(const BasisSet<double>& basis) {
  GauXC::BasisSet<double> out;
  out.reserve(basis.size());
  for (const auto& s : basis) {
    const std::int32_t nprim = s.nprim();
    if (nprim < 1 || nprim > 32)
      throw Exception("Shell primitive count must be in [1,32]");
    GauXC::Shell<double>::prim_array alpha;
    alpha.fill(0.0);
    GauXC::Shell<double>::prim_array coeff;
    coeff.fill(0.0);
    for (std::int32_t i = 0; i < nprim; ++i) {
      alpha[i] = s.alpha_data()[i];
      coeff[i] = s.coeff_data()[i];
    }
    GauXC::Shell<double>::cart_array O{s.O_data()[0], s.O_data()[1],
                                       s.O_data()[2]};
    out.push_back(GauXC::Shell<double>(
        GauXC::PrimSize(nprim), GauXC::AngularMomentum(s.l()),
        GauXC::SphericalType(s.pure()), alpha, coeff, O, s.normalized()));
  }
  return out;
}

GauXC::PruningScheme to_gauxc(PruningScheme s) {
  switch (s) {
    case PruningScheme::Unpruned:
      return GauXC::PruningScheme::Unpruned;
    case PruningScheme::Robust:
      return GauXC::PruningScheme::Robust;
    case PruningScheme::Treutler:
      return GauXC::PruningScheme::Treutler;
  }
  throw Exception("Unknown SkalaXC PruningScheme");
}

GauXC::RadialQuad to_gauxc(RadialQuad q) {
  switch (q) {
    case RadialQuad::Becke:
      return GauXC::RadialQuad::Becke;
    case RadialQuad::MuraKnowles:
      return GauXC::RadialQuad::MuraKnowles;
    case RadialQuad::MurrayHandyLaming:
      return GauXC::RadialQuad::MurrayHandyLaming;
    case RadialQuad::TreutlerAhlrichs:
      return GauXC::RadialQuad::TreutlerAhlrichs;
  }
  throw Exception("Unknown SkalaXC RadialQuad");
}

GauXC::AtomicGridSizeDefault to_gauxc(AtomicGridSizeDefault g) {
  switch (g) {
    case AtomicGridSizeDefault::FineGrid:
      return GauXC::AtomicGridSizeDefault::FineGrid;
    case AtomicGridSizeDefault::UltraFineGrid:
      return GauXC::AtomicGridSizeDefault::UltraFineGrid;
    case AtomicGridSizeDefault::SuperFineGrid:
      return GauXC::AtomicGridSizeDefault::SuperFineGrid;
    case AtomicGridSizeDefault::GM3:
      return GauXC::AtomicGridSizeDefault::GM3;
    case AtomicGridSizeDefault::GM5:
      return GauXC::AtomicGridSizeDefault::GM5;
  }
  throw Exception("Unknown SkalaXC AtomicGridSize");
}

GauXC::XCWeightAlg to_gauxc(XCWeightAlg a) {
  switch (a) {
    case XCWeightAlg::NOTPARTITIONED:
      return GauXC::XCWeightAlg::NOTPARTITIONED;
    case XCWeightAlg::Becke:
      return GauXC::XCWeightAlg::Becke;
    case XCWeightAlg::SSF:
      return GauXC::XCWeightAlg::SSF;
    case XCWeightAlg::LKO:
      return GauXC::XCWeightAlg::LKO;
  }
  throw Exception("Unknown SkalaXC XCWeightAlg");
}

}  // namespace

#ifdef SKALAXC_HAS_HDF5
// ===========================================================================
// HDF5 record readers (mirror GauXC::read_hdf5_record).
// ===========================================================================
void read_hdf5_record(std::vector<Atom>& mol, std::string fname,
                      std::string dset) {
  try {
    GauXC::Molecule gmol;
    GauXC::read_hdf5_record(gmol, std::move(fname), std::move(dset));
    mol.clear();
    mol.reserve(gmol.size());
    for (const auto& a : gmol)
      mol.push_back(Atom{AtomicNumber(a.Z.get()), a.x, a.y, a.z});
  } catch (const Exception&) {
    throw;
  } catch (const std::exception& e) {
    throw Exception(e.what());
  }
}

void read_hdf5_record(std::vector<Shell<double>>& basis, std::string fname,
                      std::string dset) {
  try {
    GauXC::BasisSet<double> gbasis;
    GauXC::read_hdf5_record(gbasis, std::move(fname), std::move(dset));
    basis.clear();
    basis.reserve(gbasis.size());
    for (const auto& gs : gbasis) {
      Shell<double>::prim_array alpha{};
      Shell<double>::prim_array coeff{};
      const std::int32_t nprim = gs.nprim();
      for (std::int32_t i = 0; i < nprim && i < 32; ++i) {
        alpha[i] = gs.alpha_data()[i];
        coeff[i] = gs.coeff_data()[i];
      }
      Shell<double>::cart_array O{gs.O_data()[0], gs.O_data()[1],
                                  gs.O_data()[2]};
      // GauXC has already normalized the coefficients on read; wrap them
      // verbatim (normalize=false) so the pipeline reuses them unchanged.
      basis.push_back(Shell<double>(PrimSize(nprim), AngularMomentum(gs.l()),
                                    SphericalType(gs.pure()), alpha, coeff, O,
                                    /*normalize=*/false));
    }
  } catch (const Exception&) {
    throw;
  } catch (const std::exception& e) {
    throw Exception(e.what());
  }
}
#endif  // SKALAXC_HAS_HDF5

// ===========================================================================
// RuntimeEnvironment
// ===========================================================================
RuntimeEnvironment::RuntimeEnvironment(SKALAXC_MPI_CODE(MPI_Comm comm))
    : pimpl_(std::make_unique<Impl>(SKALAXC_MPI_CODE(comm))) {}
RuntimeEnvironment::RuntimeEnvironment(SKALAXC_MPI_CODE(MPI_Comm comm, )
                                           DeviceRuntimeSettings settings) {
#ifdef SKALAXC_HAS_CUDA
  if (!(settings.memory_fraction > 0.0 && settings.memory_fraction <= 1.0))
    throw Exception("CUDA memory_fraction must be in (0, 1]");
  activate_cuda_device(types::DeviceId{settings.device_id});
  pimpl_ = std::make_unique<Impl>(SKALAXC_MPI_CODE(comm, ) settings);
#else
  (void)settings;
  SKALAXC_MPI_CODE((void)comm;)
  throw Exception("SkalaXC was built without CUDA support");
#endif
}
RuntimeEnvironment::~RuntimeEnvironment() = default;
RuntimeEnvironment::RuntimeEnvironment(RuntimeEnvironment&&) noexcept = default;
RuntimeEnvironment& RuntimeEnvironment::operator=(
    RuntimeEnvironment&&) noexcept = default;

int RuntimeEnvironment::comm_rank() const { return pimpl_->rt.comm_rank(); }
int RuntimeEnvironment::comm_size() const { return pimpl_->rt.comm_size(); }

// ===========================================================================
// MolGrid / MolGridFactory
// ===========================================================================
MolGrid::MolGrid(std::unique_ptr<Impl> impl) : pimpl_(std::move(impl)) {}
MolGrid::~MolGrid() = default;
MolGrid::MolGrid(MolGrid&&) noexcept = default;
MolGrid& MolGrid::operator=(MolGrid&&) noexcept = default;

MolGrid MolGridFactory::create_default_molgrid(
    const Molecule& mol, PruningScheme pruning_scheme, BatchSize batch_size,
    RadialQuad radial_quad, AtomicGridSizeDefault grid_size) {
  try {
    GauXC::Molecule gmol = to_gauxc(mol);
    GauXC::MolGrid gmg = GauXC::MolGridFactory::create_default_molgrid(
        gmol, to_gauxc(pruning_scheme), GauXC::BatchSize(batch_size.raw()),
        to_gauxc(radial_quad), to_gauxc(grid_size));
    return detail::Access::make_molgrid(
        std::make_unique<MolGrid::Impl>(std::move(gmg)));
  } catch (const Exception&) {
    throw;
  } catch (const std::exception& e) {
    throw Exception(e.what());
  }
}

// ===========================================================================
// LoadBalancer / LoadBalancerFactory
// ===========================================================================
LoadBalancer::LoadBalancer(std::unique_ptr<Impl> impl)
    : pimpl_(std::move(impl)) {}
LoadBalancer::~LoadBalancer() = default;
LoadBalancer::LoadBalancer(LoadBalancer&&) noexcept = default;
LoadBalancer& LoadBalancer::operator=(LoadBalancer&&) noexcept = default;

LoadBalancerFactory::LoadBalancerFactory(ExecutionSpace ex,
                                         std::string kernel_name)
    : ex_(ex), kernel_name_(std::move(kernel_name)) {}

LoadBalancer LoadBalancerFactory::get_instance(const RuntimeEnvironment& rt,
                                               const Molecule& mol,
                                               const MolGrid& mg,
                                               const BasisSet<double>& basis) {
  try {
    const auto* rtimpl = detail::Access::impl(rt);
    const auto* mgimpl = detail::Access::impl(mg);
    if (ex_ == ExecutionSpace::Device) {
#ifdef SKALAXC_HAS_CUDA
      if (rtimpl->ex != ExecutionSpace::Device)
        throw Exception("Device load balancing requires a device runtime");
      activate_cuda_device(rtimpl->device_id);
#else
      throw Exception("SkalaXC was built without CUDA support");
#endif
    }
    GauXC::Molecule gmol = to_gauxc(mol);
    GauXC::BasisSet<double> gbasis = to_gauxc(basis);
    GauXC::LoadBalancer glb = detail::make_atomic_domain_load_balancer(
        rtimpl->rt, gmol, mgimpl->mg, gbasis, kernel_name_);
    return detail::Access::make_load_balancer(
        std::make_unique<LoadBalancer::Impl>(std::move(glb), ex_,
                                             rtimpl->device_id,
                                             rtimpl->device_memory_fraction));
  } catch (const Exception&) {
    throw;
  } catch (const std::exception& e) {
    throw Exception(e.what());
  }
}

// ===========================================================================
// MolecularWeights / MolecularWeightsFactory
// ===========================================================================
MolecularWeights::MolecularWeights(std::unique_ptr<Impl> impl)
    : pimpl_(std::move(impl)) {}
MolecularWeights::~MolecularWeights() = default;
MolecularWeights::MolecularWeights(MolecularWeights&&) noexcept = default;
MolecularWeights& MolecularWeights::operator=(MolecularWeights&&) noexcept =
    default;

void MolecularWeights::modify_weights(LoadBalancer& lb) const {
  try {
    auto* lbimpl = detail::Access::impl(lb);
    auto& glb = lbimpl->lb;
    if (pimpl_->ex != lbimpl->ex)
      throw Exception(
          "Molecular weights and load balancer execution spaces must match");
    if (glb.state().modified_weights_are_stored)
      throw Exception("Attempting to overwrite modified weights");

    // Replicate GauXC::MolecularWeights::modify_weights (sort tasks ->
    // partition) but snapshot the pre-partition ("raw") quadrature weights in
    // between. SkalaXC owns those raw weights so GauXC master stays untouched;
    // ML models that request the atomic_grid_weights feature consume them.
    auto lwd = GauXC::LocalWorkDriverFactory::make_local_work_driver(
        GauXC::ExecutionSpace::Host, "Default");
    auto* host_lwd = dynamic_cast<GauXC::LocalHostWorkDriver*>(lwd.get());
    if (!host_lwd) throw Exception("Expected a LocalHostWorkDriver");

    auto& tasks = glb.get_tasks();
    std::stable_sort(tasks.begin(), tasks.end(),
                     [](const GauXC::XCTask& a, const GauXC::XCTask& b) {
                       return (a.points.size() * a.bfn_screening.nbe) >
                              (b.points.size() * b.bfn_screening.nbe);
                     });

    lbimpl->raw_weights.resize(tasks.size());
    for (std::size_t i = 0; i < tasks.size(); ++i)
      lbimpl->raw_weights[i] = tasks[i].weights;

    const GauXC::XCWeightAlg weight_alg = to_gauxc(pimpl_->settings.weight_alg);
    if (pimpl_->ex == ExecutionSpace::Host) {
      host_lwd->partition_weights(weight_alg, glb.molecule(), glb.molmeta(),
                                  tasks.begin(), tasks.end());
      glb.state().modified_weights_are_stored = true;
      glb.state().weight_alg = weight_alg;
    } else {
#ifdef SKALAXC_HAS_CUDA
      activate_cuda_device(lbimpl->device_id);
      GauXC::MolecularWeightsSettings settings;
      settings.weight_alg = weight_alg;
      GauXC::MolecularWeightsFactory factory(GauXC::ExecutionSpace::Device,
                                             pimpl_->kernel, settings);
      factory.get_shared_instance()->modify_weights(glb);
#else
      throw Exception("SkalaXC was built without CUDA support");
#endif
    }
  } catch (const Exception&) {
    throw;
  } catch (const std::exception& e) {
    throw Exception(e.what());
  }
}

MolecularWeightsFactory::MolecularWeightsFactory(
    ExecutionSpace ex, std::string kernel_name,
    MolecularWeightsSettings settings)
    : ex_(ex), kernel_name_(std::move(kernel_name)), settings_(settings) {}

MolecularWeights MolecularWeightsFactory::get_instance() {
#ifndef SKALAXC_HAS_CUDA
  if (ex_ == ExecutionSpace::Device)
    throw Exception("SkalaXC was built without CUDA support");
#endif
  auto impl = std::make_unique<MolecularWeights::Impl>();
  impl->ex = ex_;
  impl->kernel = kernel_name_;
  impl->settings = settings_;
  return detail::Access::make_molecular_weights(std::move(impl));
}

// ===========================================================================
// detail::IntegratorCore + make_integrator_core
// ===========================================================================
namespace detail {

IntegratorCore::IntegratorCore(std::unique_ptr<Impl> impl)
    : pimpl_(std::move(impl)) {}
IntegratorCore::~IntegratorCore() = default;
IntegratorCore::IntegratorCore(IntegratorCore&&) noexcept = default;
IntegratorCore& IntegratorCore::operator=(IntegratorCore&&) noexcept = default;

std::int64_t IntegratorCore::nbf() const { return pimpl_->nbf; }
std::int64_t IntegratorCore::natoms() const { return pimpl_->natoms; }

double IntegratorCore::eval_exc_vxc(const double* Ps, const double* Pz,
                                    double* VXCs, double* VXCz) {
  if (!Ps || !Pz || !VXCs || !VXCz)
    throw Exception("SkalaXC matrix data must not be null");
  const std::int64_t n = pimpl_->nbf;
  try {
    const Eigen::Map<const ColMajorMatrix> scalar_density(Ps, n, n);
    const Eigen::Map<const ColMajorMatrix> spin_density(Pz, n, n);
    Eigen::Map<ColMajorMatrix> scalar_potential(VXCs, n, n);
    Eigen::Map<ColMajorMatrix> spin_potential(VXCz, n, n);
    return pimpl_->driver->eval_exc_vxc_uks(scalar_density, spin_density,
                                            scalar_potential, spin_potential);
  } catch (const Exception&) {
    throw;
  } catch (const std::exception& e) {
    throw Exception(e.what());
  }
}

void IntegratorCore::eval_exc_grad(const double* Ps, const double* Pz,
                                   double* gradient) {
  if (!Ps || !Pz || !gradient)
    throw Exception("SkalaXC gradient data must not be null");
  const std::int64_t n = pimpl_->nbf;
  const std::int64_t na = pimpl_->natoms;
  try {
    const Eigen::Map<const ColMajorMatrix> scalar_density(Ps, n, n);
    const Eigen::Map<const ColMajorMatrix> spin_density(Pz, n, n);
    Eigen::Map<RowMajorMatrix> gradient_matrix(gradient, na, 3);
    pimpl_->driver->eval_exc_grad_uks(scalar_density, spin_density,
                                      gradient_matrix);
  } catch (const Exception&) {
    throw;
  } catch (const std::exception& e) {
    throw Exception(e.what());
  }
}

DiagnosticsSnapshot IntegratorCore::diagnostics() const {
  return pimpl_->driver->diagnostics();
}

void IntegratorCore::reset_diagnostics() {
  pimpl_->driver->reset_diagnostics();
}

std::unique_ptr<IntegratorCore> make_integrator_core(
    ExecutionSpace ex, const functional_type& func, const LoadBalancer& lb,
    TimingSettings timing_settings, DomainBatchMode domain_batch_mode) {
  try {
    if (func.empty())
      throw Exception("SkalaXC functional model must not be empty");
    const auto* lbimpl = Access::impl(lb);
    if (ex != lbimpl->ex)
      throw Exception(
          "XC integrator and load balancer execution spaces must match");
    // GauXC exposes only a non-const state() accessor; reading the
    // modified-weights flag is a logically const observation.
    if (!const_cast<GauXC::LoadBalancer&>(lbimpl->lb)
             .state()
             .modified_weights_are_stored)
      throw Exception(
          "SkalaXC XCIntegratorFactory requires modified weights; call "
          "MolecularWeights::modify_weights first");
    auto impl = std::make_unique<IntegratorCore::Impl>();
    if (ex == ExecutionSpace::Host) {
      impl->driver = std::make_unique<SkalaHostDriver>(
          lbimpl->lb, lbimpl->raw_weights, func.model(), timing_settings,
          domain_batch_mode);
    } else {
#ifdef SKALAXC_HAS_CUDA
      impl->driver = std::make_unique<SkalaDeviceDriver>(
          lbimpl->lb, lbimpl->raw_weights, func.model(), lbimpl->device_id,
          lbimpl->device_memory_fraction, timing_settings, domain_batch_mode);
#else
      throw Exception("SkalaXC was built without CUDA support");
#endif
    }
    impl->driver->log_model_load_timing();
    impl->nbf = lbimpl->lb.basis().nbf();
    impl->natoms = static_cast<std::int64_t>(lbimpl->lb.molecule().size());
    return std::make_unique<IntegratorCore>(std::move(impl));
  } catch (const Exception&) {
    throw;
  } catch (const std::exception& e) {
    throw Exception(e.what());
  }
}

}  // namespace detail

}  // namespace SkalaXC
