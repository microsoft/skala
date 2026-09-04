/**
 * SkalaXC public C API implementation.
 *
 * Thin extern "C" wrapper over the SkalaXC C++ pipeline. Each stage is an
 * opaque handle owning the corresponding move-only SkalaXC value type; status
 * codes and a thread-local fixed error buffer are used so no GauXC / Torch /
 * C++ exception crosses this boundary. The XC-integrator handle holds the
 * non-template evaluation core (SkalaXC::detail::IntegratorCore) directly,
 * which exposes nbf/natoms and raw-pointer UKS evaluation.
 */
#include <skalaxc/skalaxc.h>
#include <skalaxc/skalaxc.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

constexpr std::size_t error_message_capacity = 2048;
thread_local std::array<char, error_message_capacity> g_last_error{};

void set_error(const char* message) noexcept {
  std::size_t length = 0;
  if (message)
    while (length + 1 < g_last_error.size() && message[length] != '\0') {
      g_last_error[length] = message[length];
      ++length;
    }
  g_last_error[length] = '\0';
}

template <bool InvalidArgumentStatus = false, typename Function>
skalaxc_status_t translate_exceptions(Function&& function) noexcept {
  try {
    return function();
  } catch (const std::invalid_argument& error) {
    set_error(error.what());
    return InvalidArgumentStatus ? SKALAXC_INVALID_ARGUMENT : SKALAXC_ERROR;
  } catch (const std::exception& error) {
    set_error(error.what());
    return SKALAXC_ERROR;
  } catch (...) {
    set_error("unknown error");
    return SKALAXC_ERROR;
  }
}

template <typename Function, typename Result>
Result translate_query_exceptions(Function&& function,
                                  Result failure) noexcept {
  try {
    return function();
  } catch (const std::exception& error) {
    set_error(error.what());
  } catch (...) {
    set_error("unknown error");
  }
  return failure;
}

template <typename Enum>
int enum_value(const Enum& value) noexcept {
  // C callers may supply values outside the C++ enum's range. Read the C ABI
  // representation without an enum-to-integer conversion so those values can
  // be validated by the switch statements below without triggering undefined
  // behavior or sanitizer checks first.
  static_assert(sizeof(Enum) == sizeof(int));
  int result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

SkalaXC::ExecutionSpace to_exec(const enum SkalaXC_ExecutionSpace& ex) {
  switch (enum_value(ex)) {
    case SkalaXC_ExecutionSpace_Host:
      return SkalaXC::ExecutionSpace::Host;
    case SkalaXC_ExecutionSpace_Device:
      return SkalaXC::ExecutionSpace::Device;
    default:
      throw std::invalid_argument("invalid execution space");
  }
}

SkalaXC::PruningScheme to_pruning(const enum SkalaXC_PruningScheme& pruning) {
  switch (enum_value(pruning)) {
    case SkalaXC_PruningScheme_Unpruned:
      return SkalaXC::PruningScheme::Unpruned;
    case SkalaXC_PruningScheme_Robust:
      return SkalaXC::PruningScheme::Robust;
    case SkalaXC_PruningScheme_Treutler:
      return SkalaXC::PruningScheme::Treutler;
    default:
      throw std::invalid_argument("invalid pruning scheme");
  }
}

SkalaXC::RadialQuad to_radial_quad(const enum SkalaXC_RadialQuad& radial_quad) {
  switch (enum_value(radial_quad)) {
    case SkalaXC_RadialQuad_Becke:
      return SkalaXC::RadialQuad::Becke;
    case SkalaXC_RadialQuad_MuraKnowles:
      return SkalaXC::RadialQuad::MuraKnowles;
    case SkalaXC_RadialQuad_MurrayHandyLaming:
      return SkalaXC::RadialQuad::MurrayHandyLaming;
    case SkalaXC_RadialQuad_TreutlerAhlrichs:
      return SkalaXC::RadialQuad::TreutlerAhlrichs;
    default:
      throw std::invalid_argument("invalid radial quadrature");
  }
}

SkalaXC::AtomicGridSizeDefault to_atomic_grid(
    const enum SkalaXC_AtomicGridSizeDefault& atomic_grid) {
  switch (enum_value(atomic_grid)) {
    case SkalaXC_AtomicGridSizeDefault_FineGrid:
      return SkalaXC::AtomicGridSizeDefault::FineGrid;
    case SkalaXC_AtomicGridSizeDefault_UltraFineGrid:
      return SkalaXC::AtomicGridSizeDefault::UltraFineGrid;
    case SkalaXC_AtomicGridSizeDefault_SuperFineGrid:
      return SkalaXC::AtomicGridSizeDefault::SuperFineGrid;
    case SkalaXC_AtomicGridSizeDefault_GM3:
      return SkalaXC::AtomicGridSizeDefault::GM3;
    case SkalaXC_AtomicGridSizeDefault_GM5:
      return SkalaXC::AtomicGridSizeDefault::GM5;
    default:
      throw std::invalid_argument("invalid atomic grid size");
  }
}

SkalaXC::TimingSettings to_timing_settings(
    const skalaxc_timing_settings_t* settings) {
  SkalaXC::TimingSettings result;
  if (settings) {
    result.verbose = settings->verbose != 0;
    result.debug_logging = settings->debug_logging != 0;
  }
  return result;
}

SkalaXC::DomainBatchMode to_domain_batch_mode(
    const enum SkalaXC_DomainBatchMode& mode) {
  switch (enum_value(mode)) {
    case SkalaXC_DomainBatchMode_Conservative:
      return SkalaXC::DomainBatchMode::Conservative;
    case SkalaXC_DomainBatchMode_Aggressive:
      return SkalaXC::DomainBatchMode::Aggressive;
    default:
      throw std::invalid_argument("invalid domain batch mode");
  }
}

std::int64_t clamp_nanoseconds(std::uint64_t value) {
  constexpr auto maximum =
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  return value > maximum ? std::numeric_limits<std::int64_t>::max()
                         : static_cast<std::int64_t>(value);
}

void copy_diagnostics(const SkalaXC::DiagnosticsSnapshot& source,
                      skalaxc_diagnostics_snapshot_t& destination) {
  static_assert(SKALAXC_TIMING_METRIC_COUNT == SkalaXC::timing_metric_count);
  destination.backend = static_cast<int32_t>(source.backend);
  destination.rank = source.rank;
  destination.communicator_size = source.communicator_size;
  destination.device_id = source.device_id;
  destination.openmp_threads = source.openmp_threads;
  destination.device_memory_fraction = source.device_memory_fraction;
  destination.domain_batch_mode =
      static_cast<int32_t>(source.domain_batch_mode);
  for (std::size_t index = 0; index < SkalaXC::timing_metric_count; ++index) {
    const auto& source_timing = source.timings[index];
    auto& destination_timing = destination.timings[index];
    destination_timing.last_nanoseconds =
        clamp_nanoseconds(source_timing.last_nanoseconds);
    destination_timing.total_nanoseconds =
        clamp_nanoseconds(source_timing.total_nanoseconds);
    destination_timing.call_count =
        static_cast<int64_t>(source_timing.call_count);
    destination_timing.status = static_cast<int32_t>(source_timing.status);
  }
  destination.exc_vxc_calls = static_cast<int64_t>(source.exc_vxc_calls);
  destination.exc_gradient_calls =
      static_cast<int64_t>(source.exc_gradient_calls);
  destination.model_batches = static_cast<int64_t>(source.model_batches);
  destination.domains = static_cast<int64_t>(source.domains);
  destination.tasks = static_cast<int64_t>(source.tasks);
  destination.points = static_cast<int64_t>(source.points);
  destination.local_atoms = static_cast<int64_t>(source.local_atoms);
  destination.configured_model_batches =
      static_cast<int64_t>(source.configured_model_batches);
  destination.task_points_min = static_cast<int64_t>(source.task_points_min);
  destination.task_points_max = static_cast<int64_t>(source.task_points_max);
  destination.task_basis_min = static_cast<int64_t>(source.task_basis_min);
  destination.task_basis_max = static_cast<int64_t>(source.task_basis_max);
  destination.model_batch_points_min =
      static_cast<int64_t>(source.model_batch_points_min);
  destination.model_batch_points_max =
      static_cast<int64_t>(source.model_batch_points_max);
  destination.max_domains_per_model_batch =
      static_cast<int64_t>(source.max_domains_per_model_batch);
}

SkalaXC::XCWeightAlg to_weight(const enum SkalaXC_XCWeightAlg& a) {
  switch (enum_value(a)) {
    case SkalaXC_XCWeightAlg_NOTPARTITIONED:
      return SkalaXC::XCWeightAlg::NOTPARTITIONED;
    case SkalaXC_XCWeightAlg_Becke:
      return SkalaXC::XCWeightAlg::Becke;
    case SkalaXC_XCWeightAlg_LKO:
      return SkalaXC::XCWeightAlg::LKO;
    case SkalaXC_XCWeightAlg_SSF:
      return SkalaXC::XCWeightAlg::SSF;
    default:
      throw std::invalid_argument("invalid XC weight algorithm");
  }
}

}  // namespace

// ===========================================================================
// Opaque handle definitions (one per pipeline stage).
// ===========================================================================

struct skalaxc_runtime_environment {
  SkalaXC::RuntimeEnvironment rt;
  explicit skalaxc_runtime_environment(SkalaXC::RuntimeEnvironment r)
      : rt(std::move(r)) {}
};

struct skalaxc_molecule {
  SkalaXC::Molecule mol;
};

struct skalaxc_basisset {
  SkalaXC::BasisSet<double> basis;
};

struct skalaxc_molgrid {
  SkalaXC::MolGrid mg;
  explicit skalaxc_molgrid(SkalaXC::MolGrid m) : mg(std::move(m)) {}
};

struct skalaxc_load_balancer {
  SkalaXC::LoadBalancer lb;
  explicit skalaxc_load_balancer(SkalaXC::LoadBalancer l) : lb(std::move(l)) {}
};

struct skalaxc_molecular_weights {
  SkalaXC::MolecularWeights mw;
  explicit skalaxc_molecular_weights(SkalaXC::MolecularWeights m)
      : mw(std::move(m)) {}
};

struct skalaxc_functional {
  SkalaXC::functional_type func;
};

struct skalaxc_xc_integrator {
  std::unique_ptr<SkalaXC::detail::IntegratorCore> core;
};

extern "C" {

const char* skalaxc_version(void) { return SkalaXC::version().data(); }

void skalaxc_timing_settings_default(skalaxc_timing_settings_t* settings) {
  if (!settings) return;
  settings->verbose = 0;
  settings->debug_logging = 0;
}

// ---------------------------------------------------------------------------
// Runtime environment
// ---------------------------------------------------------------------------

skalaxc_status_t skalaxc_runtime_environment_create(
    SKALAXC_MPI_CODE(MPI_Comm comm, ) skalaxc_runtime_environment_t* out) {
  if (out) *out = nullptr;
  if (!out) {
    set_error("null argument");
    return SKALAXC_INVALID_ARGUMENT;
  }
  return translate_exceptions([&] {
    auto handle = std::make_unique<skalaxc_runtime_environment>(
        SkalaXC::RuntimeEnvironment(SKALAXC_MPI_CODE(comm)));
    *out = handle.release();
    return SKALAXC_SUCCESS;
  });
}

void skalaxc_device_runtime_settings_default(
    skalaxc_device_runtime_settings_t* settings) {
  if (!settings) return;
  settings->device_id = 0;
  settings->memory_fraction = 0.75;
}

skalaxc_status_t skalaxc_device_runtime_environment_create(
    SKALAXC_MPI_CODE(MPI_Comm comm, )
        const skalaxc_device_runtime_settings_t* settings,
    skalaxc_runtime_environment_t* out) {
  if (out) *out = nullptr;
  if (!out) {
    set_error("null argument");
    return SKALAXC_INVALID_ARGUMENT;
  }
  return translate_exceptions([&] {
    SkalaXC::DeviceRuntimeSettings cpp_settings;
    if (settings) {
      cpp_settings.device_id = settings->device_id;
      cpp_settings.memory_fraction = settings->memory_fraction;
    }
    auto handle = std::make_unique<skalaxc_runtime_environment>(
        SkalaXC::RuntimeEnvironment(SKALAXC_MPI_CODE(comm, ) cpp_settings));
    *out = handle.release();
    return SKALAXC_SUCCESS;
  });
}

#ifdef SKALAXC_HAS_MPI
skalaxc_status_t skalaxc_runtime_environment_create_f(
    MPI_Fint comm, skalaxc_runtime_environment_t* out) {
  return skalaxc_runtime_environment_create(MPI_Comm_f2c(comm), out);
}

skalaxc_status_t skalaxc_device_runtime_environment_create_f(
    MPI_Fint comm, const skalaxc_device_runtime_settings_t* settings,
    skalaxc_runtime_environment_t* out) {
  return skalaxc_device_runtime_environment_create(MPI_Comm_f2c(comm), settings,
                                                   out);
}
#endif

int skalaxc_runtime_environment_comm_rank(skalaxc_runtime_environment_t rt) {
  if (!rt) {
    set_error("null argument");
    return -1;
  }
  return translate_query_exceptions([&] { return rt->rt.comm_rank(); }, -1);
}

int skalaxc_runtime_environment_comm_size(skalaxc_runtime_environment_t rt) {
  if (!rt) {
    set_error("null argument");
    return -1;
  }
  return translate_query_exceptions([&] { return rt->rt.comm_size(); }, -1);
}

void skalaxc_runtime_environment_destroy(skalaxc_runtime_environment_t rt) {
  delete rt;
}

// ---------------------------------------------------------------------------
// Molecule
// ---------------------------------------------------------------------------

skalaxc_status_t skalaxc_molecule_create(skalaxc_molecule_t* out) {
  if (out) *out = nullptr;
  if (!out) {
    set_error("null argument");
    return SKALAXC_INVALID_ARGUMENT;
  }
  return translate_exceptions([&] {
    auto handle = std::make_unique<skalaxc_molecule>();
    *out = handle.release();
    return SKALAXC_SUCCESS;
  });
}

skalaxc_status_t skalaxc_molecule_add_atom(skalaxc_molecule_t mol,
                                           int64_t atomic_number, double x,
                                           double y, double z) {
  if (!mol) {
    set_error("null argument");
    return SKALAXC_INVALID_ARGUMENT;
  }
  return translate_exceptions([&] {
    mol->mol.emplace_back(SkalaXC::AtomicNumber(atomic_number), x, y, z);
    return SKALAXC_SUCCESS;
  });
}

skalaxc_status_t skalaxc_molecule_from_arrays(int64_t natoms, const int64_t* Z,
                                              const double* atom_xyz,
                                              skalaxc_molecule_t* out) {
  if (out) *out = nullptr;
  if (!Z || !atom_xyz || !out || natoms < 0) {
    set_error("null argument");
    return SKALAXC_INVALID_ARGUMENT;
  }
  return translate_exceptions([&] {
    auto handle = std::make_unique<skalaxc_molecule>();
    handle->mol.reserve(static_cast<std::size_t>(natoms));
    for (int64_t i = 0; i < natoms; ++i) {
      handle->mol.emplace_back(SkalaXC::AtomicNumber(Z[i]), atom_xyz[3 * i],
                               atom_xyz[3 * i + 1], atom_xyz[3 * i + 2]);
    }
    *out = handle.release();
    return SKALAXC_SUCCESS;
  });
}

#ifdef SKALAXC_HAS_HDF5
skalaxc_status_t skalaxc_molecule_from_hdf5(const char* path, const char* dset,
                                            skalaxc_molecule_t* out) {
  if (out) *out = nullptr;
  if (!path || !dset || !out) {
    set_error("null argument");
    return SKALAXC_INVALID_ARGUMENT;
  }
  return translate_exceptions([&] {
    auto handle = std::make_unique<skalaxc_molecule>();
    SkalaXC::read_hdf5_record(handle->mol, path, dset);
    *out = handle.release();
    return SKALAXC_SUCCESS;
  });
}
#endif

int64_t skalaxc_molecule_natoms(skalaxc_molecule_t mol) {
  if (!mol) {
    set_error("null argument");
    return -1;
  }
  return translate_query_exceptions(
      [&] { return static_cast<int64_t>(mol->mol.natoms()); }, int64_t{-1});
}

void skalaxc_molecule_destroy(skalaxc_molecule_t mol) { delete mol; }

// ---------------------------------------------------------------------------
// Basis set
// ---------------------------------------------------------------------------

skalaxc_status_t skalaxc_basisset_create(skalaxc_basisset_t* out) {
  if (out) *out = nullptr;
  if (!out) {
    set_error("null argument");
    return SKALAXC_INVALID_ARGUMENT;
  }
  return translate_exceptions([&] {
    auto handle = std::make_unique<skalaxc_basisset>();
    *out = handle.release();
    return SKALAXC_SUCCESS;
  });
}

skalaxc_status_t skalaxc_basisset_add_shell(
    skalaxc_basisset_t basis, int32_t l, int32_t pure, const double* center_xyz,
    int32_t nprim, const double* exponents, const double* coefficients,
    int32_t normalize) {
  if (!basis || !center_xyz || !exponents || !coefficients) {
    set_error("null argument");
    return SKALAXC_INVALID_ARGUMENT;
  }
  if (nprim < 1 || nprim > 32) {
    set_error("nprim must be in [1, 32]");
    return SKALAXC_INVALID_ARGUMENT;
  }
  return translate_exceptions([&] {
    SkalaXC::Shell<double>::prim_array alpha{};
    SkalaXC::Shell<double>::prim_array coeff{};
    for (int32_t i = 0; i < nprim; ++i) {
      alpha[static_cast<std::size_t>(i)] = exponents[i];
      coeff[static_cast<std::size_t>(i)] = coefficients[i];
    }
    const SkalaXC::Shell<double>::cart_array center{
        center_xyz[0], center_xyz[1], center_xyz[2]};
    basis->basis.emplace_back(
        SkalaXC::PrimSize(nprim), SkalaXC::AngularMomentum(l),
        SkalaXC::SphericalType(pure), alpha, coeff, center, normalize != 0);
    return SKALAXC_SUCCESS;
  });
}

skalaxc_status_t skalaxc_basisset_from_arrays(
    int64_t nshells, const int32_t* shell_l, const int32_t* shell_pure,
    const double* shell_xyz, const int32_t* shell_nprim, const double* prim_exp,
    const double* prim_coeff, skalaxc_basisset_t* out) {
  if (out) *out = nullptr;
  if (!shell_l || !shell_pure || !shell_xyz || !shell_nprim || !prim_exp ||
      !prim_coeff || !out || nshells < 0) {
    set_error("null argument");
    return SKALAXC_INVALID_ARGUMENT;
  }
  return translate_exceptions([&] {
    auto handle = std::make_unique<skalaxc_basisset>();
    handle->basis.reserve(static_cast<std::size_t>(nshells));
    int64_t offset = 0;
    for (int64_t s = 0; s < nshells; ++s) {
      const int32_t nprim = shell_nprim[s];
      if (nprim < 1 || nprim > 32) {
        set_error("nprim must be in [1, 32]");
        return SKALAXC_INVALID_ARGUMENT;
      }
      SkalaXC::Shell<double>::prim_array alpha{};
      SkalaXC::Shell<double>::prim_array coeff{};
      for (int32_t i = 0; i < nprim; ++i) {
        alpha[static_cast<std::size_t>(i)] = prim_exp[offset + i];
        coeff[static_cast<std::size_t>(i)] = prim_coeff[offset + i];
      }
      const SkalaXC::Shell<double>::cart_array center{
          shell_xyz[3 * s], shell_xyz[3 * s + 1], shell_xyz[3 * s + 2]};
      handle->basis.emplace_back(
          SkalaXC::PrimSize(nprim), SkalaXC::AngularMomentum(shell_l[s]),
          SkalaXC::SphericalType(shell_pure[s]), alpha, coeff, center, true);
      offset += nprim;
    }
    *out = handle.release();
    return SKALAXC_SUCCESS;
  });
}

#ifdef SKALAXC_HAS_HDF5
skalaxc_status_t skalaxc_basisset_from_hdf5(const char* path, const char* dset,
                                            skalaxc_basisset_t* out) {
  if (out) *out = nullptr;
  if (!path || !dset || !out) {
    set_error("null argument");
    return SKALAXC_INVALID_ARGUMENT;
  }
  return translate_exceptions([&] {
    auto handle = std::make_unique<skalaxc_basisset>();
    SkalaXC::read_hdf5_record(handle->basis, path, dset);
    *out = handle.release();
    return SKALAXC_SUCCESS;
  });
}
#endif

int64_t skalaxc_basisset_nbf(skalaxc_basisset_t basis) {
  if (!basis) {
    set_error("null argument");
    return -1;
  }
  return translate_query_exceptions(
      [&] { return static_cast<int64_t>(basis->basis.nbf()); }, int64_t{-1});
}

void skalaxc_basisset_destroy(skalaxc_basisset_t basis) { delete basis; }

// ---------------------------------------------------------------------------
// Molecular grid
// ---------------------------------------------------------------------------

skalaxc_status_t skalaxc_molgrid_create_default(
    skalaxc_molecule_t mol, const skalaxc_grid_settings_t* grid,
    skalaxc_molgrid_t* out) {
  if (out) *out = nullptr;
  if (!mol || !out) {
    set_error("null argument");
    return SKALAXC_INVALID_ARGUMENT;
  }
  return translate_exceptions<true>([&] {
    SkalaXC::PruningScheme pruning = SkalaXC::PruningScheme::Unpruned;
    int64_t batch_size = 512;
    SkalaXC::RadialQuad radial = SkalaXC::RadialQuad::MuraKnowles;
    SkalaXC::AtomicGridSizeDefault atomic =
        SkalaXC::AtomicGridSizeDefault::UltraFineGrid;
    if (grid) {
      pruning = to_pruning(grid->pruning);
      batch_size = grid->batch_size;
      radial = to_radial_quad(grid->radial_quad);
      atomic = to_atomic_grid(grid->atomic_grid);
    }
    auto mg = SkalaXC::MolGridFactory::create_default_molgrid(
        mol->mol, pruning, SkalaXC::BatchSize(batch_size), radial, atomic);
    auto handle = std::make_unique<skalaxc_molgrid>(std::move(mg));
    *out = handle.release();
    return SKALAXC_SUCCESS;
  });
}

void skalaxc_molgrid_destroy(skalaxc_molgrid_t mg) { delete mg; }

// ---------------------------------------------------------------------------
// Load balancer
// ---------------------------------------------------------------------------

skalaxc_status_t skalaxc_load_balancer_create(enum SkalaXC_ExecutionSpace ex,
                                              skalaxc_runtime_environment_t rt,
                                              skalaxc_molecule_t mol,
                                              skalaxc_molgrid_t mg,
                                              skalaxc_basisset_t basis,
                                              skalaxc_load_balancer_t* out) {
  if (out) *out = nullptr;
  if (!rt || !mol || !mg || !basis || !out) {
    set_error("null argument");
    return SKALAXC_INVALID_ARGUMENT;
  }
  return translate_exceptions<true>([&] {
    SkalaXC::LoadBalancerFactory factory(to_exec(ex));
    auto lb = factory.get_instance(rt->rt, mol->mol, mg->mg, basis->basis);
    auto handle = std::make_unique<skalaxc_load_balancer>(std::move(lb));
    *out = handle.release();
    return SKALAXC_SUCCESS;
  });
}

void skalaxc_load_balancer_destroy(skalaxc_load_balancer_t lb) { delete lb; }

// ---------------------------------------------------------------------------
// Molecular weights
// ---------------------------------------------------------------------------

skalaxc_status_t skalaxc_molecular_weights_create(
    enum SkalaXC_ExecutionSpace ex, enum SkalaXC_XCWeightAlg weight_alg,
    skalaxc_molecular_weights_t* out) {
  if (out) *out = nullptr;
  if (!out) {
    set_error("null argument");
    return SKALAXC_INVALID_ARGUMENT;
  }
  return translate_exceptions<true>([&] {
    SkalaXC::MolecularWeightsSettings settings;
    settings.weight_alg = to_weight(weight_alg);
    SkalaXC::MolecularWeightsFactory factory(to_exec(ex), "Default", settings);
    auto mw = factory.get_instance();
    auto handle = std::make_unique<skalaxc_molecular_weights>(std::move(mw));
    *out = handle.release();
    return SKALAXC_SUCCESS;
  });
}

skalaxc_status_t skalaxc_molecular_weights_modify_weights(
    skalaxc_molecular_weights_t mw, skalaxc_load_balancer_t lb) {
  if (!mw || !lb) {
    set_error("null argument");
    return SKALAXC_INVALID_ARGUMENT;
  }
  return translate_exceptions([&] {
    mw->mw.modify_weights(lb->lb);
    return SKALAXC_SUCCESS;
  });
}

void skalaxc_molecular_weights_destroy(skalaxc_molecular_weights_t mw) {
  delete mw;
}

// ---------------------------------------------------------------------------
// Functional
// ---------------------------------------------------------------------------

skalaxc_status_t skalaxc_functional_create(const char* model,
                                           skalaxc_functional_t* out) {
  if (out) *out = nullptr;
  if (!model || !out) {
    set_error("null argument");
    return SKALAXC_INVALID_ARGUMENT;
  }
  return translate_exceptions([&] {
    auto handle = std::make_unique<skalaxc_functional>(
        skalaxc_functional{SkalaXC::functional_type(model)});
    *out = handle.release();
    return SKALAXC_SUCCESS;
  });
}

void skalaxc_functional_destroy(skalaxc_functional_t func) { delete func; }

// ---------------------------------------------------------------------------
// XC integrator
// ---------------------------------------------------------------------------

void skalaxc_integrator_settings_default(
    skalaxc_integrator_settings_t* settings) {
  if (!settings) return;
  skalaxc_timing_settings_default(&settings->timing);
  settings->domain_batch_mode = SkalaXC_DomainBatchMode_Conservative;
}

skalaxc_status_t skalaxc_xc_integrator_create(enum SkalaXC_ExecutionSpace ex,
                                              skalaxc_functional_t func,
                                              skalaxc_load_balancer_t lb,
                                              skalaxc_xc_integrator_t* out) {
  return skalaxc_xc_integrator_create_with_timing(ex, func, lb, nullptr, out);
}

skalaxc_status_t skalaxc_xc_integrator_create_with_timing(
    enum SkalaXC_ExecutionSpace ex, skalaxc_functional_t func,
    skalaxc_load_balancer_t lb, const skalaxc_timing_settings_t* settings,
    skalaxc_xc_integrator_t* out) {
  skalaxc_integrator_settings_t integrator_settings;
  skalaxc_integrator_settings_default(&integrator_settings);
  if (settings) integrator_settings.timing = *settings;
  return skalaxc_xc_integrator_create_with_settings(ex, func, lb,
                                                    &integrator_settings, out);
}

skalaxc_status_t skalaxc_xc_integrator_create_with_settings(
    enum SkalaXC_ExecutionSpace ex, skalaxc_functional_t func,
    skalaxc_load_balancer_t lb, const skalaxc_integrator_settings_t* settings,
    skalaxc_xc_integrator_t* out) {
  if (out) *out = nullptr;
  if (!func || !lb || !out) {
    set_error("null argument");
    return SKALAXC_INVALID_ARGUMENT;
  }
  return translate_exceptions<true>([&] {
    skalaxc_integrator_settings_t resolved_settings;
    skalaxc_integrator_settings_default(&resolved_settings);
    if (settings) resolved_settings = *settings;
    auto core = SkalaXC::detail::make_integrator_core(
        to_exec(ex), func->func, lb->lb,
        to_timing_settings(&resolved_settings.timing),
        to_domain_batch_mode(resolved_settings.domain_batch_mode));
    auto handle = std::make_unique<skalaxc_xc_integrator>();
    handle->core = std::move(core);
    *out = handle.release();
    return SKALAXC_SUCCESS;
  });
}

int64_t skalaxc_xc_integrator_nbf(skalaxc_xc_integrator_t xc) {
  if (!xc || !xc->core) {
    set_error("null argument");
    return -1;
  }
  return translate_query_exceptions([&] { return xc->core->nbf(); },
                                    int64_t{-1});
}

int64_t skalaxc_xc_integrator_natoms(skalaxc_xc_integrator_t xc) {
  if (!xc || !xc->core) {
    set_error("null argument");
    return -1;
  }
  return translate_query_exceptions([&] { return xc->core->natoms(); },
                                    int64_t{-1});
}

skalaxc_status_t skalaxc_xc_integrator_eval_exc_vxc_uks(
    skalaxc_xc_integrator_t xc, const double* Ps, const double* Pz,
    double* VXCs, double* VXCz, double* exc_out) {
  if (!xc || !xc->core || !Ps || !Pz || !VXCs || !VXCz || !exc_out) {
    set_error("null argument");
    return SKALAXC_INVALID_ARGUMENT;
  }
  return translate_exceptions([&] {
    *exc_out = xc->core->eval_exc_vxc(Ps, Pz, VXCs, VXCz);
    return SKALAXC_SUCCESS;
  });
}

skalaxc_status_t skalaxc_xc_integrator_eval_exc_grad_uks(
    skalaxc_xc_integrator_t xc, const double* Ps, const double* Pz,
    double* gradient_out) {
  if (!xc || !xc->core || !Ps || !Pz || !gradient_out) {
    set_error("null argument");
    return SKALAXC_INVALID_ARGUMENT;
  }
  return translate_exceptions([&] {
    xc->core->eval_exc_grad(Ps, Pz, gradient_out);
    return SKALAXC_SUCCESS;
  });
}

skalaxc_status_t skalaxc_xc_integrator_get_diagnostics(
    skalaxc_xc_integrator_t xc, skalaxc_diagnostics_snapshot_t* out) {
  if (!xc || !xc->core || !out) {
    set_error("null argument");
    return SKALAXC_INVALID_ARGUMENT;
  }
  return translate_exceptions([&] {
    copy_diagnostics(xc->core->diagnostics(), *out);
    return SKALAXC_SUCCESS;
  });
}

skalaxc_status_t skalaxc_xc_integrator_reset_diagnostics(
    skalaxc_xc_integrator_t xc) {
  if (!xc || !xc->core) {
    set_error("null argument");
    return SKALAXC_INVALID_ARGUMENT;
  }
  return translate_exceptions([&] {
    xc->core->reset_diagnostics();
    return SKALAXC_SUCCESS;
  });
}

void skalaxc_xc_integrator_destroy(skalaxc_xc_integrator_t xc) { delete xc; }

// ---------------------------------------------------------------------------
// Grid settings + error reporting
// ---------------------------------------------------------------------------

void skalaxc_grid_settings_default(skalaxc_grid_settings_t* settings) {
  if (!settings) return;
  settings->pruning = SkalaXC_PruningScheme_Unpruned;
  settings->batch_size = 512;
  settings->radial_quad = SkalaXC_RadialQuad_MuraKnowles;
  settings->atomic_grid = SkalaXC_AtomicGridSizeDefault_UltraFineGrid;
}

const char* skalaxc_last_error_message(void) { return g_last_error.data(); }

}  // extern "C"
