/**
 * @file
 * @brief SkalaXC C API: molecular-weights handle.
 *
 * Mirrors SkalaXC::MolecularWeights / SkalaXC::MolecularWeightsFactory. Part of
 * the public C API; include <skalaxc/skalaxc.h> to get the whole API.
 */
#pragma once

#include <skalaxc/c/enums.h>
#include <skalaxc/c/load_balancer.h>
#include <skalaxc/c/status.h>
#include <skalaxc/skalaxc_export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque molecular-weights handle (mirrors SkalaXC::MolecularWeights).
 */
typedef struct skalaxc_molecular_weights* skalaxc_molecular_weights_t;

/**
 * @brief Create a molecular-weights partitioner (mirrors
 * SkalaXC::MolecularWeightsFactory::get_instance).
 * @param ex Execution space; device requires a CUDA-enabled build and runtime.
 * @param weight_alg Weight partitioning scheme.
 * @param out Output owning handle on success (free with
 *            skalaxc_molecular_weights_destroy).
 * @return SKALAXC_SUCCESS on success, or an error status.
 */
skalaxc_status_t skalaxc_molecular_weights_create(
    enum SkalaXC_ExecutionSpace ex, enum SkalaXC_XCWeightAlg weight_alg,
    skalaxc_molecular_weights_t* out) SKALAXC_EXPORT;

/**
 * @brief Partition the quadrature weights stored on a load balancer in place.
 * @param mw Molecular-weights handle.
 * @param lb Load-balancer handle whose weights are modified.
 * @return SKALAXC_SUCCESS on success, or an error status.
 */
skalaxc_status_t skalaxc_molecular_weights_modify_weights(
    skalaxc_molecular_weights_t mw, skalaxc_load_balancer_t lb) SKALAXC_EXPORT;

/**
 * @brief Destroy a molecular-weights handle (NULL is tolerated).
 * @param mw Molecular-weights handle to destroy.
 */
void skalaxc_molecular_weights_destroy(skalaxc_molecular_weights_t mw)
    SKALAXC_EXPORT;

#ifdef __cplusplus
}  // extern "C"
#endif
