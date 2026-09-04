/**
 * @file
 * @brief SkalaXC public C++ API.
 *
 * The public surface mirrors GauXC's public XC pipeline
 * (RuntimeEnvironment -> Molecule / BasisSet -> MolGrid -> LoadBalancer ->
 * MolecularWeights -> functional_type -> XCIntegratorFactory -> XCIntegrator)
 * so GauXC-style host code ports by changing only the namespace. SkalaXC owns
 * ABI-isolated replicas of the value types; only the host UKS methods SkalaXC
 * actually supports (eval_exc_vxc, eval_exc_grad) are exposed. The Skala ML
 * model selector lives on SkalaXC::functional_type -- the single semantic
 * deviation from GauXC (whose functional_type is an ExchCXX functional).
 *
 * ABI isolation contract: this header includes no GauXC or LibTorch headers
 * and exposes no GauXC/Torch/Eigen types. All such usage is confined to the
 * implementation translation units behind a PIMPL.
 *
 * The sole exception is a native MPI_Comm on the RuntimeEnvironment
 * constructor in MPI builds (SKALAXC_HAS_MPI). MPI_Comm is a standard MPI C
 * type, not a GauXC/Torch type, so the isolation guarantee is preserved.
 */
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <skalaxc/c/mpi.h>
#include <skalaxc/detail/integrator_core.hpp>
#include <skalaxc/skalaxc_export.h>

namespace SkalaXC {

/**
 * @brief Return the SkalaXC semantic version.
 *
 * The returned view references static storage owned by the library.
 * @return SkalaXC semantic version string.
 */
SKALAXC_EXPORT std::string_view version() noexcept;

/**
 * @brief Exception type thrown across the SkalaXC boundary.
 *
 * GauXC exceptions are translated internally and re-thrown as this
 * std::runtime_error-derived type so no GauXC exception type escapes.
 */
class SKALAXC_EXPORT Exception : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// ===========================================================================
// Strong scalar wrappers.
// ===========================================================================
namespace types {

/**
 * @brief Type-safe scalar wrapper with opt-in additive arithmetic.
 * @tparam T Stored scalar type.
 * @tparam ParameterType Tag distinguishing otherwise identical scalar types.
 * @tparam Additive Whether additive operators are available.
 */
template <typename T, typename ParameterType, bool Additive = false>
class NamedType {
 public:
  using value_type = T;  ///< Stored scalar type.

  static constexpr bool is_additive = Additive;  ///< Additive-operator policy.

  /** @brief Construct a value initialized scalar wrapper. */
  constexpr NamedType() : value_() {}

  /**
   * @brief Construct from a scalar value.
   * @param value Scalar value to copy.
   */
  constexpr explicit NamedType(const T& value) : value_(value) {}

  /**
   * @brief Construct by moving a scalar value.
   * @param value Scalar value to move.
   */
  constexpr explicit NamedType(T&& value) : value_(std::move(value)) {}

  /** @return Stored scalar value. */
  constexpr T raw() const noexcept { return value_; }

 private:
  T value_;
};

/**
 * @brief Compare two named values for equality.
 * @param a Left operand.
 * @param b Right operand.
 * @return Whether the stored values are equal.
 */
template <typename T, typename ParameterType, bool Additive>
constexpr bool operator==(const NamedType<T, ParameterType, Additive>& a,
                          const NamedType<T, ParameterType, Additive>& b) {
  return a.raw() == b.raw();
}

/**
 * @brief Compare two named values for inequality.
 * @param a Left operand.
 * @param b Right operand.
 * @return Whether the stored values differ.
 */
template <typename T, typename ParameterType, bool Additive>
constexpr bool operator!=(const NamedType<T, ParameterType, Additive>& a,
                          const NamedType<T, ParameterType, Additive>& b) {
  return !(a == b);
}

/** @brief Named scalar type that supports addition. */
template <typename T, typename ParameterType>
using AdditiveNamedType = NamedType<T, ParameterType, true>;

/**
 * @brief Add two values of the same additive named type.
 * @param left Left operand.
 * @param right Right operand.
 * @return Sum with the same named type.
 */
template <typename T, typename ParameterType>
constexpr AdditiveNamedType<T, ParameterType> operator+(
    AdditiveNamedType<T, ParameterType> left,
    AdditiveNamedType<T, ParameterType> right) {
  return AdditiveNamedType<T, ParameterType>{left.raw() + right.raw()};
}

/**
 * @brief Add a named value in place.
 * @param left Value to update.
 * @param right Value to add.
 * @return Reference to @p left.
 */
template <typename T, typename ParameterType>
constexpr AdditiveNamedType<T, ParameterType>& operator+=(
    AdditiveNamedType<T, ParameterType>& left,
    AdditiveNamedType<T, ParameterType> right) {
  left = left + right;
  return left;
}

}  // namespace types

namespace detail {

/// Grants the implementation translation unit access to stage internals.
struct Access;

}  // namespace detail

/** @brief Type-safe atomic number. */
using AtomicNumber = types::NamedType<std::int64_t, struct AtomicNumberTag>;
/** @brief Type-safe number of primitives in a basis shell. */
using PrimSize = types::NamedType<std::int32_t, struct PrimSizeTag>;
/** @brief Type-safe basis-shell angular momentum. */
using AngularMomentum =
    types::NamedType<std::int32_t, struct AngularMomentumTag>;
/** @brief Type-safe spherical-shell indicator. */
using SphericalType = types::NamedType<std::int32_t, struct SphericalTypeTag>;
/** @brief Type-safe radial quadrature size. */
using RadialSize = types::NamedType<std::int64_t, struct RadialSizeTag>;
/** @brief Type-safe angular quadrature size. */
using AngularSize = types::NamedType<std::int64_t, struct AngularSizeTag>;
/** @brief Type-safe quadrature batch size. */
using BatchSize = types::NamedType<std::int64_t, struct BatchSizeTag>;
/** @brief Type-safe radial quadrature scale. */
using RadialScale = types::NamedType<double, struct RadialScaleTag>;

// ===========================================================================
// Molecule / basis value types (mirror GauXC).
// ===========================================================================

/** @brief A single atom (nucleus). Coordinates are in bohr. */
struct Atom {
  AtomicNumber Z;  ///< atomic number
  double x;        ///< nuclear x-coordinate (bohr)
  double y;        ///< nuclear y-coordinate (bohr)
  double z;        ///< nuclear z-coordinate (bohr)

  Atom() = default;

  /**
   * @brief Construct an atom from its nuclear charge and coordinates.
   * @param Z_ Atomic number.
   * @param x_ Nuclear x-coordinate in bohr.
   * @param y_ Nuclear y-coordinate in bohr.
   * @param z_ Nuclear z-coordinate in bohr.
   */
  Atom(AtomicNumber Z_, double x_, double y_, double z_)
      : Z(Z_), x(x_), y(y_), z(z_) {}
};

/** @brief Molecular geometry (mirrors GauXC::Molecule : std::vector<Atom>). */
class Molecule : public std::vector<Atom> {
 public:
  using std::vector<Atom>::vector;

  /** @return Number of atoms in the molecule. */
  std::size_t natoms() const { return this->size(); }

  /** @return Largest atomic number in the molecule, or zero when empty. */
  AtomicNumber maxZ() const {
    std::int64_t z = 0;
    for (const auto& a : *this) z = std::max(z, a.Z.raw());
    return AtomicNumber(z);
  }
};

/**
 * @brief A single contracted Gaussian basis shell (mirrors GauXC::Shell<F>).
 *
 * Stores the primitive exponents/coefficients and center as supplied; the
 * @p normalize flag is honored when the shell is realized inside the library.
 * @tparam F Primitive exponent and coefficient scalar type.
 */
template <typename F>
class Shell {
 public:
  using prim_array = std::array<F, 32>;      ///< Primitive storage type.
  using cart_array = std::array<double, 3>;  ///< Cartesian center type.

  Shell() = default;

  /**
   * @brief Construct a contracted Gaussian shell.
   * @param nprim Number of active entries in @p alpha and @p coeff.
   * @param l Angular momentum.
   * @param pure Nonzero for a pure spherical shell; zero for Cartesian.
   * @param alpha Primitive exponents; only the first @p nprim are used.
   * @param coeff Contraction coefficients; only the first @p nprim are used.
   * @param O Shell center in bohr.
   * @param normalize Whether to normalize the realized shell.
   */
  Shell(PrimSize nprim, AngularMomentum l, SphericalType pure, prim_array alpha,
        prim_array coeff, cart_array O, bool normalize = true)
      : nprim_(nprim.raw()),
        l_(l.raw()),
        pure_(pure.raw()),
        alpha_(alpha),
        coeff_(coeff),
        O_(O),
        normalize_(normalize) {}

  /** @return Number of primitives. */
  std::int32_t nprim() const { return nprim_; }
  /** @return Angular momentum. */
  std::int32_t l() const { return l_; }
  /** @return Nonzero for a pure spherical shell. */
  std::int32_t pure() const { return pure_; }
  /** @return Whether normalization was requested. */
  bool normalized() const { return normalize_; }

  /** @return Pointer to the primitive exponent storage. */
  const F* alpha_data() const { return alpha_.data(); }
  /** @return Pointer to the contraction coefficient storage. */
  const F* coeff_data() const { return coeff_.data(); }
  /** @return Pointer to the three shell-center coordinates in bohr. */
  const double* O_data() const { return O_.data(); }

  /** @return Number of Cartesian basis functions in the shell. */
  std::int32_t cart_size() const { return (l_ + 1) * (l_ + 2) / 2; }
  /** @return Number of pure spherical basis functions in the shell. */
  std::int32_t pure_size() const { return 2 * l_ + 1; }
  /** @return Active basis-function count for the configured shell type. */
  std::int32_t size() const { return pure_ ? pure_size() : cart_size(); }

 private:
  std::int32_t nprim_ = 0;
  std::int32_t l_ = 0;
  std::int32_t pure_ = 1;
  prim_array alpha_{};
  prim_array coeff_{};
  cart_array O_{};
  bool normalize_ = true;
};

/**
 * @brief Basis set (mirrors GauXC::BasisSet<F> : std::vector<Shell<F>>).
 * @tparam F Primitive exponent and coefficient scalar type.
 */
template <typename F>
struct BasisSet : public std::vector<Shell<F>> {
  using std::vector<Shell<F>>::vector;

  /** @return Number of shells. */
  std::int32_t nshells() const {
    return static_cast<std::int32_t>(this->size());
  }

  /** @return Number of basis functions in the configured shell types. */
  std::int32_t nbf() const {
    std::int32_t n = 0;
    for (const auto& s : *this) n += s.size();
    return n;
  }

  /** @return Number of basis functions when all shells are Cartesian. */
  std::int32_t nbf_cart() const {
    std::int32_t n = 0;
    for (const auto& s : *this) n += s.cart_size();
    return n;
  }

  /** @return Maximum angular momentum, or zero for an empty basis. */
  std::int32_t max_l() const {
    std::int32_t l = 0;
    for (const auto& s : *this) l = std::max(l, s.l());
    return l;
  }
};

// ===========================================================================
// Enums (mirror GauXC, same enumerator names).
// ===========================================================================

/** @brief Radial quadrature scheme (mirrors GauXC::RadialQuad). */
enum class RadialQuad {
  Becke,
  MuraKnowles,
  MurrayHandyLaming,
  TreutlerAhlrichs
};

/** @brief Atomic grid size preset (mirrors GauXC::AtomicGridSizeDefault). */
enum class AtomicGridSizeDefault {
  FineGrid,
  UltraFineGrid,
  SuperFineGrid,
  GM3,
  GM5
};

/** @brief Pruning scheme for atomic quadratures (mirrors GauXC). */
enum class PruningScheme { Unpruned, Robust, Treutler };

/** @brief Execution space (mirrors GauXC::ExecutionSpace). */
enum class ExecutionSpace { Host, Device };

/** @brief XC weight partitioning scheme (mirrors GauXC::XCWeightAlg). */
enum class XCWeightAlg { NOTPARTITIONED, Becke, SSF, LKO };

// ===========================================================================
// Settings value types (mirror GauXC).
// ===========================================================================

/** @brief Molecular-weight partitioning settings (mirrors GauXC). */
struct MolecularWeightsSettings {
  XCWeightAlg weight_alg = XCWeightAlg::SSF;  ///< Partitioning algorithm.
};

/** @brief CUDA runtime allocation and device-selection settings. */
struct DeviceRuntimeSettings {
  int device_id = 0;              ///< CUDA device ordinal.
  double memory_fraction = 0.75;  ///< Fraction of available device memory.
};

/** @brief Policy for grouping complete local atomic domains into model calls.
 */
enum class DomainBatchMode {
  /** @brief Evaluate one complete atomic domain per model call. */
  Conservative,
  /** @brief Batch all local domains having the same exact grid size. */
  Aggressive
};

/** @brief Configuration for lightweight, rank-local diagnostics. */
struct TimingSettings {
  /** @brief Wait for complete CUDA event timings when diagnostics are read. */
  bool verbose = false;
  /** @brief Emit human-readable, rank-local diagnostics to stderr. */
  bool debug_logging = false;
};

/** @brief Stable identifiers for integrator timing phases. */
enum class TimingMetric : std::size_t {
  ModelLoad,
  FeatureConstruction,
  ModelBatchPacking,
  ModelForward,
  ModelBackward,
  PotentialMapping,
  AOAssembly,
  GradientAssembly,
  MPIReduction,
  TotalEXCVXC,
  TotalEXCGradient,
  Count
};

/** @brief Number of stable timing metrics. */
inline constexpr std::size_t timing_metric_count =
    static_cast<std::size_t>(TimingMetric::Count);

/** @brief Availability of one timing value in a diagnostics snapshot. */
enum class TimingStatus { Unavailable, Pending, Complete };

/** @brief Last and cumulative values for one timing phase. */
struct TimingValue {
  std::uint64_t last_nanoseconds = 0;   ///< Most recent completed duration.
  std::uint64_t total_nanoseconds = 0;  ///< Sum of completed durations.
  std::uint64_t call_count = 0;         ///< Number of phase invocations.
  TimingStatus status = TimingStatus::Unavailable;  ///< Value availability.
};

/**
 * @brief Rank-local, non-collective snapshot of integrator diagnostics.
 *
 * Setup topology and model-load timing persist across reset_diagnostics();
 * evaluation timings, calls, processed batches, and domains do not.
 */
struct DiagnosticsSnapshot {
  ExecutionSpace backend = ExecutionSpace::Host;  ///< Evaluation backend.
  int rank = 0;                         ///< Rank in the runtime communicator.
  int communicator_size = 1;            ///< Runtime communicator size.
  int device_id = -1;                   ///< CUDA ordinal, or -1 on the host.
  int openmp_threads = 1;               ///< OpenMP threads at construction.
  double device_memory_fraction = 0.0;  ///< GauXC CUDA arena fraction.
  DomainBatchMode domain_batch_mode = DomainBatchMode::Conservative;
  ///< Configured complete-domain batching policy.
  std::array<TimingValue, timing_metric_count> timings{};  ///< Phase timings.
  std::uint64_t exc_vxc_calls = 0;       ///< EXC/VXC evaluations since reset.
  std::uint64_t exc_gradient_calls = 0;  ///< Gradient evaluations since reset.
  std::uint64_t model_batches = 0;       ///< Model batches since reset.
  std::uint64_t domains = 0;             ///< Atomic domains since reset.
  std::uint64_t tasks = 0;               ///< Local quadrature tasks processed.
  std::uint64_t points = 0;              ///< Local quadrature points processed.
  std::uint64_t local_atoms = 0;         ///< Atomic domains owned by this rank.
  std::uint64_t configured_model_batches = 0;  ///< Planned model batches.
  std::uint64_t task_points_min = 0;           ///< Minimum points per task.
  std::uint64_t task_points_max = 0;           ///< Maximum points per task.
  std::uint64_t task_basis_min = 0;  ///< Minimum basis functions per task.
  std::uint64_t task_basis_max = 0;  ///< Maximum basis functions per task.
  std::uint64_t model_batch_points_min = 0;  ///< Minimum batch point count.
  std::uint64_t model_batch_points_max = 0;  ///< Maximum batch point count.
  std::uint64_t max_domains_per_model_batch =
      0;  ///< Maximum domains per batch.

  /**
   * @brief Access one timing value.
   * @param metric Timing phase to access.
   * @return Timing value for @p metric.
   */
  const TimingValue& timing(TimingMetric metric) const {
    return timings.at(static_cast<std::size_t>(metric));
  }
};

/** @brief Base XC integrator settings (mirrors GauXC::IntegratorSettingsXC). */
struct IntegratorSettingsXC {
  virtual ~IntegratorSettingsXC() = default;
};

/**
 * @brief XC-gradient settings (mirrors GauXC::IntegratorSettingsEXC_GRAD).
 *
 * SkalaXC supports only @c include_weight_derivatives == true (the GauXC
 * default); requesting @c false throws from XCIntegrator::eval_exc_grad.
 */
struct IntegratorSettingsEXC_GRAD : public IntegratorSettingsXC {
  bool include_weight_derivatives = true;  ///< Include molecular-weight terms.
};

/**
 * @brief Skala ML functional selector.
 *
 * SkalaXC deviation from GauXC: GauXC's functional_type is an ExchCXX
 * functional; here it carries the Skala model selector ("LDA"/"PBE"/"TPSS" or
 * a path to a .fun TorchScript model), consumed when the integrator is built.
 */
class functional_type {
 public:
  functional_type() = default;

  /**
   * @brief Construct a functional selector.
   * @param model Model name or path to a TorchScript @c .fun file.
   */
  functional_type(std::string model) : model_(std::move(model)) {}

  /**
   * @brief Construct a functional selector from a C string.
   * @param model Model name or path; a null pointer selects an empty model.
   */
  functional_type(const char* model) : model_(model ? model : "") {}

  /** @return Configured model name or path. */
  const std::string& model() const { return model_; }
  /** @return Whether no model selector is configured. */
  bool empty() const { return model_.empty(); }

 private:
  std::string model_;
};

#ifdef SKALAXC_HAS_HDF5
// ===========================================================================
// HDF5 record readers (mirror GauXC::read_hdf5_record overloads).
// ===========================================================================

/**
 * @brief Read a molecule record from an HDF5 file (mirrors GauXC).
 * @param mol Destination atom vector.
 * @param fname HDF5 file path.
 * @param dset Molecule dataset path.
 */
SKALAXC_EXPORT void read_hdf5_record(std::vector<Atom>& mol, std::string fname,
                                     std::string dset);

/**
 * @brief Read a basis-set record from an HDF5 file (mirrors GauXC).
 * @param basis Destination basis-shell vector.
 * @param fname HDF5 file path.
 * @param dset Basis-set dataset path.
 */
SKALAXC_EXPORT void read_hdf5_record(std::vector<Shell<double>>& basis,
                                     std::string fname, std::string dset);
#endif  // SKALAXC_HAS_HDF5

// ===========================================================================
// Pipeline stages (opaque; own GauXC state behind PIMPL).
// ===========================================================================

/** @brief Runtime environment (mirrors GauXC::RuntimeEnvironment). */
class SKALAXC_EXPORT RuntimeEnvironment {
 public:
  /**
   * @brief Construct a host runtime environment.
   * @param comm MPI communicator used by the runtime.
   */
  explicit RuntimeEnvironment(SKALAXC_MPI_CODE(MPI_Comm comm));

  /**
   * @brief Construct a device runtime environment.
   * @param comm MPI communicator used by the runtime.
   * @param settings CUDA device and memory-allocation settings.
   */
  explicit RuntimeEnvironment(SKALAXC_MPI_CODE(MPI_Comm comm, )
                                  DeviceRuntimeSettings settings);
  ~RuntimeEnvironment();

  /**
   * @brief Move-construct a runtime environment.
   * @param other Runtime environment to move from.
   */
  RuntimeEnvironment(RuntimeEnvironment&& other) noexcept;

  /**
   * @brief Move-assign a runtime environment.
   * @param other Runtime environment to move from.
   * @return Reference to this runtime environment.
   */
  RuntimeEnvironment& operator=(RuntimeEnvironment&& other) noexcept;
  RuntimeEnvironment(const RuntimeEnvironment&) = delete;
  RuntimeEnvironment& operator=(const RuntimeEnvironment&) = delete;

  /** @return Rank in the configured communicator. */
  int comm_rank() const;
  /** @return Number of ranks in the configured communicator. */
  int comm_size() const;

  struct Impl;

 private:
  friend struct detail::Access;
  std::unique_ptr<Impl> pimpl_;
};

/** @brief Molecular integration grid (mirrors GauXC::MolGrid). */
class SKALAXC_EXPORT MolGrid {
 public:
  ~MolGrid();

  /**
   * @brief Move-construct a molecular grid.
   * @param other Molecular grid to move from.
   */
  MolGrid(MolGrid&& other) noexcept;

  /**
   * @brief Move-assign a molecular grid.
   * @param other Molecular grid to move from.
   * @return Reference to this molecular grid.
   */
  MolGrid& operator=(MolGrid&& other) noexcept;
  MolGrid(const MolGrid&) = delete;
  MolGrid& operator=(const MolGrid&) = delete;

  struct Impl;

 private:
  friend struct detail::Access;
  explicit MolGrid(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> pimpl_;
};

/** @brief Factory for default molecular grids (mirrors GauXC::MolGridFactory).
 */
struct SKALAXC_EXPORT MolGridFactory {
  /**
   * @brief Create a molecular grid from a standard atomic-grid preset.
   * @param mol Molecular geometry in bohr.
   * @param pruning_scheme Atomic-grid pruning scheme.
   * @param batch_size Maximum quadrature batch size.
   * @param radial_quad Radial quadrature scheme.
   * @param grid_size Atomic-grid size preset.
   * @return Constructed molecular integration grid.
   */
  static MolGrid create_default_molgrid(const Molecule& mol,
                                        PruningScheme pruning_scheme,
                                        BatchSize batch_size,
                                        RadialQuad radial_quad,
                                        AtomicGridSizeDefault grid_size);
};

/** @brief Quadrature load balancer (mirrors GauXC::LoadBalancer). */
class SKALAXC_EXPORT LoadBalancer {
 public:
  ~LoadBalancer();

  /**
   * @brief Move-construct a load balancer.
   * @param other Load balancer to move from.
   */
  LoadBalancer(LoadBalancer&& other) noexcept;

  /**
   * @brief Move-assign a load balancer.
   * @param other Load balancer to move from.
   * @return Reference to this load balancer.
   */
  LoadBalancer& operator=(LoadBalancer&& other) noexcept;
  LoadBalancer(const LoadBalancer&) = delete;
  LoadBalancer& operator=(const LoadBalancer&) = delete;

  struct Impl;

 private:
  friend struct detail::Access;
  explicit LoadBalancer(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> pimpl_;
};

/** @brief Factory for load balancers (mirrors GauXC::LoadBalancerFactory). */
class SKALAXC_EXPORT LoadBalancerFactory {
 public:
  /**
   * @brief Construct a load-balancer factory.
   * @param ex Execution space for the generated load balancer.
   * @param kernel_name GauXC load-balancing kernel name.
   */
  explicit LoadBalancerFactory(ExecutionSpace ex,
                               std::string kernel_name = "Default");

  /**
   * @brief Build a load balancer for a molecule, grid, and basis set.
   * @param rt Runtime environment retained by the load balancer.
   * @param mol Molecular geometry in bohr.
   * @param mg Molecular integration grid.
   * @param basis Gaussian basis set.
   * @return Constructed quadrature load balancer.
   */
  LoadBalancer get_instance(const RuntimeEnvironment& rt, const Molecule& mol,
                            const MolGrid& mg, const BasisSet<double>& basis);

 private:
  ExecutionSpace ex_;
  std::string kernel_name_;
};

/** @brief Molecular partition weights (mirrors GauXC::MolecularWeights). */
class SKALAXC_EXPORT MolecularWeights {
 public:
  ~MolecularWeights();

  /**
   * @brief Move-construct a molecular-weights object.
   * @param other Molecular-weights object to move from.
   */
  MolecularWeights(MolecularWeights&& other) noexcept;

  /**
   * @brief Move-assign a molecular-weights object.
   * @param other Molecular-weights object to move from.
   * @return Reference to this molecular-weights object.
   */
  MolecularWeights& operator=(MolecularWeights&& other) noexcept;
  MolecularWeights(const MolecularWeights&) = delete;
  MolecularWeights& operator=(const MolecularWeights&) = delete;

  /**
   * @brief Partition the quadrature weights stored on a load balancer.
   * @param lb Load balancer whose quadrature weights are modified in place.
   */
  void modify_weights(LoadBalancer& lb) const;

  struct Impl;

 private:
  friend struct detail::Access;
  explicit MolecularWeights(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> pimpl_;
};

/**
 * @brief Factory for molecular weights (mirrors
 * GauXC::MolecularWeightsFactory).
 */
class SKALAXC_EXPORT MolecularWeightsFactory {
 public:
  /**
   * @brief Construct a molecular-weights factory.
   * @param ex Execution space for weight partitioning.
   * @param kernel_name GauXC molecular-weights kernel name.
   * @param settings Molecular-weight partitioning settings.
   */
  MolecularWeightsFactory(ExecutionSpace ex, std::string kernel_name,
                          MolecularWeightsSettings settings = {});

  /** @return Molecular-weights object configured by this factory. */
  MolecularWeights get_instance();

 private:
  ExecutionSpace ex_;
  std::string kernel_name_;
  MolecularWeightsSettings settings_;
};

// ===========================================================================
// XCIntegrator (mirrors GauXC::XCIntegrator<MatrixType>, host UKS subset).
// ===========================================================================

/**
 * @brief SkalaXC ML exchange-correlation integrator (mirrors GauXC).
 *
 * A thin templated facade that marshals the caller's column-major @p MatrixType
 * to/from the non-template evaluation core (detail::IntegratorCore). Only the
 * host UKS methods SkalaXC supports are exposed.
 *
 * @warning An integrator instance is not safe for concurrent calls, moves, or
 * destruction. Serialize access to a shared instance or use one integrator per
 * calling thread. Distinct instances may execute concurrently.
 */
template <typename MatrixType>
class XCIntegrator {
 public:
  using matrix_type = MatrixType;  ///< Caller-provided matrix type.
  using value_type = typename MatrixType::value_type;  ///< Matrix scalar type.
  using exc_vxc_type_uks =
      std::tuple<value_type, matrix_type, matrix_type>;  ///< EXC/VXC result.
  using exc_grad_type = std::vector<value_type>;  ///< Atom-major XC gradient.

  static_assert(std::is_same<value_type, double>::value,
                "SkalaXC XCIntegrator requires a double matrix type");

  XCIntegrator() = default;

  /**
   * @brief Construct an integrator from an evaluation core.
   * @param core Evaluation core owned by the integrator.
   */
  explicit XCIntegrator(std::unique_ptr<detail::IntegratorCore> core)
      : core_(std::move(core)) {}

  XCIntegrator(const XCIntegrator&) = delete;
  XCIntegrator& operator=(const XCIntegrator&) = delete;

  /**
   * @brief Move-construct an XC integrator.
   * @param other XC integrator to move from.
   */
  XCIntegrator(XCIntegrator&& other) noexcept = default;

  /**
   * @brief Move-assign an XC integrator.
   * @param other XC integrator to move from.
   * @return Reference to this XC integrator.
   */
  XCIntegrator& operator=(XCIntegrator&& other) noexcept = default;

  /**
   * @brief Evaluate the UKS ML XC energy and potential (scalar, z).
   * @param Ps Scalar density matrix in column-major storage.
   * @param Pz Spin-density matrix in column-major storage.
   * @param settings XC evaluation settings.
   * @return XC energy, scalar potential matrix, and spin potential matrix.
   */
  exc_vxc_type_uks eval_exc_vxc(
      const MatrixType& Ps, const MatrixType& Pz,
      const IntegratorSettingsXC& settings = IntegratorSettingsXC{}) {
    require_core();
    const std::int64_t n = core_->nbf();
    MatrixType VXCs(n, n);
    MatrixType VXCz(n, n);
    const value_type EXC = eval_exc_vxc(Ps, Pz, VXCs, VXCz, settings);
    return std::make_tuple(EXC, std::move(VXCs), std::move(VXCz));
  }

  /**
   * @brief Evaluate UKS ML XC energy and potential into caller-owned matrices.
   * @param Ps Scalar density matrix in column-major storage.
   * @param Pz Spin-density matrix in column-major storage.
   * @param VXCs Pre-sized `nbf` by `nbf` scalar potential output.
   * @param VXCz Pre-sized `nbf` by `nbf` spin potential output.
   * @param settings XC evaluation settings.
   * @return XC energy.
   * @throws Exception If the integrator is uninitialized or a matrix extent is
   * invalid. Extents are validated before either output is modified.
   */
  value_type eval_exc_vxc(
      const MatrixType& Ps, const MatrixType& Pz, MatrixType& VXCs,
      MatrixType& VXCz,
      const IntegratorSettingsXC& settings = IntegratorSettingsXC{}) {
    (void)settings;
    require_core();
    const std::int64_t n = core_->nbf();
    check_square(Ps, n, "density");
    check_square(Pz, n, "density");
    check_square(VXCs, n, "potential output");
    check_square(VXCz, n, "potential output");
    return core_->eval_exc_vxc(Ps.data(), Pz.data(), VXCs.data(), VXCz.data());
  }

  /**
   * @brief Evaluate the UKS ML XC nuclear gradient (atom-major xyz).
   * @param Ps Scalar density matrix in column-major storage.
   * @param Pz Spin-density matrix in column-major storage.
   * @param settings XC-gradient evaluation settings.
   * @return XC nuclear gradient with exactly three values per atom.
   */
  exc_grad_type eval_exc_grad(
      const MatrixType& Ps, const MatrixType& Pz,
      const IntegratorSettingsXC& settings = IntegratorSettingsXC{}) {
    require_core();
    exc_grad_type gradient(static_cast<std::size_t>(3 * core_->natoms()));
    eval_exc_grad(Ps, Pz, gradient, settings);
    return gradient;
  }

  /**
   * @brief Evaluate the UKS ML XC nuclear gradient into caller-owned storage.
   * @param Ps Scalar density matrix in column-major storage.
   * @param Pz Spin-density matrix in column-major storage.
   * @param gradient Pre-sized `3 * natoms` atom-major xyz output.
   * @param settings XC-gradient evaluation settings.
   * @throws Exception If the integrator is uninitialized, an input matrix
   * extent is invalid, the gradient extent is invalid, or unsupported settings
   * are requested. Inputs and output extent are validated before the output is
   * modified.
   */
  void eval_exc_grad(
      const MatrixType& Ps, const MatrixType& Pz, exc_grad_type& gradient,
      const IntegratorSettingsXC& settings = IntegratorSettingsXC{}) {
    require_core();
    if (const auto* g =
            dynamic_cast<const IntegratorSettingsEXC_GRAD*>(&settings)) {
      if (!g->include_weight_derivatives)
        throw Exception(
            "SkalaXC eval_exc_grad supports include_weight_derivatives=true "
            "only");
    }
    const std::int64_t n = core_->nbf();
    check_square(Ps, n, "density");
    check_square(Pz, n, "density");
    if (gradient.size() != static_cast<std::size_t>(3 * core_->natoms()))
      throw Exception("SkalaXC gradient output must contain 3 * natoms values");
    core_->eval_exc_grad(Ps.data(), Pz.data(), gradient.data());
  }

  /**
   * @brief Return rank-local diagnostics without performing MPI collectives.
   * @return Snapshot of rank-local timing and workload diagnostics.
   */
  DiagnosticsSnapshot diagnostics() const {
    require_core();
    return core_->diagnostics();
  }

  /** @brief Clear evaluation timings and counters for this integrator. */
  void reset_diagnostics() {
    require_core();
    core_->reset_diagnostics();
  }

 private:
  void require_core() const {
    if (!core_) throw Exception("SkalaXC XCIntegrator is not initialized");
  }

  template <typename M>
  static void check_square(const M& m, std::int64_t n, const char* kind) {
    static_assert(
        std::is_convertible<decltype(std::declval<const M&>().data()),
                            const value_type*>::value,
        "SkalaXC matrices must expose contiguous double data via .data()");
    if (m.rows() != n || m.cols() != n)
      throw Exception(std::string("SkalaXC ") + kind +
                      " matrices must be nbf x nbf");
  }

  std::unique_ptr<detail::IntegratorCore> core_;
};

/**
 * @brief Factory for XC integrators (mirrors GauXC::XCIntegratorFactory).
 */
template <typename MatrixType>
class XCIntegratorFactory {
 public:
  /**
   * @brief Construct an XC-integrator factory.
   * @param ex Host or device execution space.
   * @param timing_settings Rank-local timing behavior.
   * @param domain_batch_mode Complete-domain model batching policy.
   */
  explicit XCIntegratorFactory(
      ExecutionSpace ex, TimingSettings timing_settings = {},
      DomainBatchMode domain_batch_mode = DomainBatchMode::Conservative)
      : ex_(ex),
        timing_settings_(timing_settings),
        domain_batch_mode_(domain_batch_mode) {}

  /**
   * @brief Build an integrator from a model functional and weighted balancer.
   * @param func Functional carrying the model selector.
   * @param lb Load balancer with partitioned weights.
   * @return Ready-to-use XC integrator.
   */
  XCIntegrator<MatrixType> get_instance(const functional_type& func,
                                        const LoadBalancer& lb) {
    return XCIntegrator<MatrixType>(detail::make_integrator_core(
        ex_, func, lb, timing_settings_, domain_batch_mode_));
  }

 private:
  ExecutionSpace ex_;                  ///< Selected execution space.
  TimingSettings timing_settings_;     ///< Rank-local timing behavior.
  DomainBatchMode domain_batch_mode_;  ///< Complete-domain batching policy.
};

}  // namespace SkalaXC
