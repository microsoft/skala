/**
 * @file
 * @brief Non-template evaluation core hidden behind XCIntegrator<MatrixType>.
 *
 * ABI isolation contract: this header exposes no GauXC/LibTorch/Eigen types.
 * IntegratorCore confines the templated public facade from the GauXC/LibTorch
 * host driver: the facade marshals caller matrices to/from column-major double
 * buffers and forwards to these non-template raw-pointer methods, whose
 * definitions live in the implementation translation unit.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <skalaxc/skalaxc_export.h>

namespace SkalaXC {

// Forward declarations (full definitions live in <skalaxc/skalaxc.hpp>).
enum class ExecutionSpace;
enum class DomainBatchMode;
class functional_type;
class LoadBalancer;
struct DiagnosticsSnapshot;
struct TimingSettings;

namespace detail {

/** @brief Caller-reported extent of a column-major matrix buffer. */
struct MatrixExtent {
  std::int64_t rows;          ///< Number of rows.
  std::int64_t cols;          ///< Number of columns.
  std::int64_t outer_stride;  ///< Column stride; must equal rows (dense).
};

/** @brief Opaque, non-template SkalaXC host evaluation core. */
class SKALAXC_EXPORT IntegratorCore {
 public:
  struct Impl;
  explicit IntegratorCore(std::unique_ptr<Impl> impl);
  ~IntegratorCore();

  IntegratorCore(IntegratorCore&&) noexcept;
  IntegratorCore& operator=(IntegratorCore&&) noexcept;
  IntegratorCore(const IntegratorCore&) = delete;
  IntegratorCore& operator=(const IntegratorCore&) = delete;

  std::int64_t nbf() const;
  std::int64_t natoms() const;

  /**
   * Evaluate UKS EXC/VXC from column-major buffers with the given extents.
   *
   * Extents are validated by the backend inside its collective error handling,
   * so an invalid shape on one MPI rank fails every rank together instead of
   * leaving the others waiting in a collective.
   */
  double eval_exc_vxc(const double* Ps, MatrixExtent Ps_extent,
                      const double* Pz, MatrixExtent Pz_extent, double* VXCs,
                      MatrixExtent VXCs_extent, double* VXCz,
                      MatrixExtent VXCz_extent);

  /**
   * Evaluate the UKS XC nuclear gradient (atom-major xyz). @p gradient_size is
   * the caller's element count and is validated collectively like the
   * matrix extents.
   */
  void eval_exc_grad(const double* Ps, MatrixExtent Ps_extent, const double* Pz,
                     MatrixExtent Pz_extent, double* gradient,
                     std::size_t gradient_size);

  /// Return rank-local diagnostics without performing MPI collectives.
  DiagnosticsSnapshot diagnostics() const;

  /// Clear evaluation timings and counters while preserving setup data.
  void reset_diagnostics();

 private:
  std::unique_ptr<Impl> pimpl_;
};

/// Build a host evaluation core from a weighted load balancer and functional.
SKALAXC_EXPORT std::unique_ptr<IntegratorCore> make_integrator_core(
    ExecutionSpace ex, const functional_type& func, const LoadBalancer& lb,
    TimingSettings timing_settings, DomainBatchMode domain_batch_mode);

}  // namespace detail
}  // namespace SkalaXC
