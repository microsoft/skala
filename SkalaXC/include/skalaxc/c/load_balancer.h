/**
 * @file
 * @brief SkalaXC C API: load-balancer handle.
 *
 * Mirrors SkalaXC::LoadBalancer / SkalaXC::LoadBalancerFactory. Part of the
 * public C API; include <skalaxc/skalaxc.h> to get the whole API.
 */
#pragma once

#include <skalaxc/c/basisset.h>
#include <skalaxc/c/enums.h>
#include <skalaxc/c/molecule.h>
#include <skalaxc/c/molgrid.h>
#include <skalaxc/c/runtime.h>
#include <skalaxc/c/status.h>
#include <skalaxc/skalaxc_export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque load-balancer handle (mirrors SkalaXC::LoadBalancer).
 */
typedef struct skalaxc_load_balancer* skalaxc_load_balancer_t;

/**
 * @brief Create a load balancer for the given system (mirrors
 * SkalaXC::LoadBalancerFactory::get_instance).
 * @param ex Execution space; device requires a CUDA-enabled build and runtime.
 * @param rt Runtime environment handle.
 * @param mol Molecule handle.
 * @param mg Molecular-grid handle.
 * @param basis Basis-set handle.
 * @param out Output owning handle on success (free with
 *            skalaxc_load_balancer_destroy).
 * @return SKALAXC_SUCCESS on success, or an error status.
 */
skalaxc_status_t skalaxc_load_balancer_create(
    enum SkalaXC_ExecutionSpace ex, skalaxc_runtime_environment_t rt,
    skalaxc_molecule_t mol, skalaxc_molgrid_t mg, skalaxc_basisset_t basis,
    skalaxc_load_balancer_t* out) SKALAXC_EXPORT;

/**
 * @brief Destroy a load-balancer handle (NULL is tolerated).
 * @param lb Load-balancer handle to destroy.
 */
void skalaxc_load_balancer_destroy(skalaxc_load_balancer_t lb) SKALAXC_EXPORT;

#ifdef __cplusplus
}  // extern "C"
#endif
