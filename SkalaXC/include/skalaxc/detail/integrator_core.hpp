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

  /// Evaluate UKS EXC/VXC from column-major nbf x nbf buffers.
  double eval_exc_vxc(const double* Ps, const double* Pz, double* VXCs,
                      double* VXCz);

  /// Evaluate the UKS XC nuclear gradient (3 * natoms, atom-major xyz).
  void eval_exc_grad(const double* Ps, const double* Pz, double* gradient);

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
