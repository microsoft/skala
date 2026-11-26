/**
 * @file
 * @brief SkalaXC C API: XC integrator handle.
 *
 * Mirrors SkalaXC::XCIntegrator / SkalaXC::XCIntegratorFactory (UKS subset).
 * Part of the public C API; include <skalaxc/skalaxc.h> to get the whole API.
 */
#pragma once

#include <stdint.h>

#include <skalaxc/c/diagnostics.h>
#include <skalaxc/c/enums.h>
#include <skalaxc/c/functional.h>
#include <skalaxc/c/load_balancer.h>
#include <skalaxc/c/status.h>
#include <skalaxc/skalaxc_export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque XC-integrator handle (mirrors SkalaXC::XCIntegrator).
 *
 * One handle must not be used concurrently from multiple threads. Serialize
 * access to a shared handle or use a separate handle per calling thread.
 */
typedef struct skalaxc_xc_integrator* skalaxc_xc_integrator_t;

/** @brief XC-integrator construction settings. */
typedef struct skalaxc_integrator_settings {
  skalaxc_timing_settings_t timing; /**< Rank-local timing behavior. */
  enum SkalaXC_DomainBatchMode domain_batch_mode; /**< Complete-domain model
                                                     batching policy. */
} skalaxc_integrator_settings_t;

/**
 * @brief Fill XC-integrator settings with conservative defaults.
 * @param settings Output settings; NULL is tolerated.
 */
void skalaxc_integrator_settings_default(
    skalaxc_integrator_settings_t* settings) SKALAXC_EXPORT;

/**
 * @brief Create an XC integrator (mirrors
 * SkalaXC::XCIntegratorFactory::get_instance).
 *
 * The load balancer must carry partitioned weights (see
 * skalaxc_molecular_weights_modify_weights) before this call.
 * @param ex Execution space; device requires a CUDA-enabled build and runtime.
 * @param func Functional handle selecting the Skala model.
 * @param lb Load-balancer handle with partitioned weights.
 * @param out Output owning handle on success (free with
 *            skalaxc_xc_integrator_destroy).
 * @return SKALAXC_SUCCESS on success, or an error status.
 */
skalaxc_status_t skalaxc_xc_integrator_create(
    enum SkalaXC_ExecutionSpace ex, skalaxc_functional_t func,
    skalaxc_load_balancer_t lb, skalaxc_xc_integrator_t* out) SKALAXC_EXPORT;

/**
 * @brief Create an XC integrator with explicit timing behavior.
 *
 * This is the settings-bearing counterpart to skalaxc_xc_integrator_create.
 * Passing NULL for @p settings uses non-synchronizing timing defaults.
 * @param ex Execution space; device requires a CUDA-enabled build and runtime.
 * @param func Functional handle selecting the Skala model.
 * @param lb Load-balancer handle with partitioned weights.
 * @param settings Timing settings, or NULL for defaults.
 * @param out Output owning handle on success.
 * @return SKALAXC_SUCCESS on success, or an error status.
 */
skalaxc_status_t skalaxc_xc_integrator_create_with_timing(
    enum SkalaXC_ExecutionSpace ex, skalaxc_functional_t func,
    skalaxc_load_balancer_t lb, const skalaxc_timing_settings_t* settings,
    skalaxc_xc_integrator_t* out) SKALAXC_EXPORT;

/**
 * @brief Create an XC integrator with timing and domain batching settings.
 * @param ex Execution space for complete-domain model batching.
 * @param func Functional handle selecting the Skala model.
 * @param lb Load-balancer handle with partitioned weights.
 * @param settings Integrator settings, or NULL for conservative defaults.
 * @param out Output owning handle on success.
 * @return SKALAXC_SUCCESS on success, or an error status.
 */
skalaxc_status_t skalaxc_xc_integrator_create_with_settings(
    enum SkalaXC_ExecutionSpace ex, skalaxc_functional_t func,
    skalaxc_load_balancer_t lb, const skalaxc_integrator_settings_t* settings,
    skalaxc_xc_integrator_t* out) SKALAXC_EXPORT;

/**
 * @brief Return the number of basis functions, or -1 if xc is null.
 * @param xc XC-integrator handle.
 * @return Number of basis functions.
 */
int64_t skalaxc_xc_integrator_nbf(skalaxc_xc_integrator_t xc) SKALAXC_EXPORT;

/**
 * @brief Return the number of atoms, or -1 if xc is null.
 * @param xc XC-integrator handle.
 * @return Number of atoms.
 */
int64_t skalaxc_xc_integrator_natoms(skalaxc_xc_integrator_t xc) SKALAXC_EXPORT;

/**
 * @brief Evaluate the UKS ML exchange-correlation energy and potential.
 *
 * Ps/Pz are the input scalar/z spin-density matrices; VXCs/VXCz receive the
 * corresponding potentials. All are nbf x nbf, column-major, leading dim nbf.
 * @param xc XC-integrator handle.
 * @param Ps Scalar spin-density matrix.
 * @param Pz Z spin-density matrix.
 * @param VXCs Output scalar XC potential.
 * @param VXCz Output z XC potential.
 * @param exc_out Output exchange-correlation energy.
 * @return SKALAXC_SUCCESS on success, or an error status.
 */
skalaxc_status_t skalaxc_xc_integrator_eval_exc_vxc_uks(
    skalaxc_xc_integrator_t xc, const double* Ps, const double* Pz,
    double* VXCs, double* VXCz, double* exc_out) SKALAXC_EXPORT;

/**
 * @brief Evaluate the UKS ML exchange-correlation energy gradient.
 *
 * Ps/Pz are nbf x nbf column-major density matrices. The caller must provide
 * space for 3 * natoms doubles in gradient_out, ordered xyz per atom.
 * @param xc XC-integrator handle.
 * @param Ps Scalar spin-density matrix.
 * @param Pz Z spin-density matrix.
 * @param gradient_out Output atom-major Cartesian gradient.
 * @return SKALAXC_SUCCESS on success, or an error status.
 */
skalaxc_status_t skalaxc_xc_integrator_eval_exc_grad_uks(
    skalaxc_xc_integrator_t xc, const double* Ps, const double* Pz,
    double* gradient_out) SKALAXC_EXPORT;

/**
 * @brief Retrieve a rank-local diagnostics snapshot without MPI collectives.
 * @param xc XC-integrator handle.
 * @param out Output diagnostics snapshot.
 * @return SKALAXC_SUCCESS on success, or an error status.
 */
skalaxc_status_t skalaxc_xc_integrator_get_diagnostics(
    skalaxc_xc_integrator_t xc,
    skalaxc_diagnostics_snapshot_t* out) SKALAXC_EXPORT;

/**
 * @brief Clear evaluation timings and counters on this integrator.
 * @param xc XC-integrator handle.
 * @return SKALAXC_SUCCESS on success, or an error status.
 */
skalaxc_status_t skalaxc_xc_integrator_reset_diagnostics(
    skalaxc_xc_integrator_t xc) SKALAXC_EXPORT;

/**
 * @brief Destroy an XC-integrator handle (NULL is tolerated).
 * @param xc XC-integrator handle to destroy.
 */
void skalaxc_xc_integrator_destroy(skalaxc_xc_integrator_t xc) SKALAXC_EXPORT;

#ifdef __cplusplus
}  // extern "C"
#endif
