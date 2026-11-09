/**
 * @file
 * @brief SkalaXC C API: molecular integration-grid settings.
 *
 * Part of the public C API; include <skalaxc/skalaxc.h> to get the whole API.
 */
#pragma once

#include <stdint.h>

#include <skalaxc/c/enums.h>
#include <skalaxc/skalaxc_export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Molecular integration-grid parameters.
 *
 * The equivalents of every argument to the default molecular-grid factory
 * (SkalaXC::MolGridFactory::create_default_molgrid). Initialize with
 * skalaxc_grid_settings_default() (do NOT zero-initialize: the defaults are not
 * all zero), then override individual fields. Passing NULL to
 * skalaxc_molgrid_create_default selects the built-in preset.
 */
typedef struct {
  enum SkalaXC_PruningScheme pruning;   ///< Pruning scheme
  int64_t batch_size;                   ///< Grid-point batch size (> 0)
  enum SkalaXC_RadialQuad radial_quad;  ///< Radial quadrature
  enum SkalaXC_AtomicGridSizeDefault atomic_grid;  ///< Atomic grid size preset
} skalaxc_grid_settings_t;

/**
 * @brief Initialize settings with the SkalaXC built-in preset.
 *
 * The preset is: unpruned, batch size 512, Mura-Knowles radial quadrature,
 * ultrafine atomic grid. No-op if settings is NULL.
 * @param settings Output grid settings to populate.
 */
void skalaxc_grid_settings_default(skalaxc_grid_settings_t* settings)
    SKALAXC_EXPORT;

#ifdef __cplusplus
}  // extern "C"
#endif
