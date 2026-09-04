/**
 * @file
 * @brief SkalaXC C API: functional (model selector) handle.
 *
 * Mirrors SkalaXC::functional_type. Part of the public C API; include
 * <skalaxc/skalaxc.h> to get the whole API.
 */
#pragma once

#include <skalaxc/c/status.h>
#include <skalaxc/skalaxc_export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque functional handle (mirrors SkalaXC::functional_type).
 */
typedef struct skalaxc_functional* skalaxc_functional_t;

/**
 * @brief Create a functional from a Skala model selector.
 * @param model Model identifier ("LDA"/"PBE"/"TPSS") or a path to a .fun model.
 * @param out Output owning handle on success (free with
 *            skalaxc_functional_destroy).
 * @return SKALAXC_SUCCESS on success, or an error status.
 */
skalaxc_status_t skalaxc_functional_create(
    const char* model, skalaxc_functional_t* out) SKALAXC_EXPORT;

/**
 * @brief Destroy a functional handle (NULL is tolerated).
 * @param func Functional handle to destroy.
 */
void skalaxc_functional_destroy(skalaxc_functional_t func) SKALAXC_EXPORT;

#ifdef __cplusplus
}  // extern "C"
#endif
