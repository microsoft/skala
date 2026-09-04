/**
 * @file
 * @brief SkalaXC C API: molecular integration-grid handle.
 *
 * Mirrors SkalaXC::MolGrid / SkalaXC::MolGridFactory. Part of the public C API;
 * include <skalaxc/skalaxc.h> to get the whole API.
 */
#pragma once

#include <skalaxc/c/grid.h>
#include <skalaxc/c/molecule.h>
#include <skalaxc/c/status.h>
#include <skalaxc/skalaxc_export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque molecular-grid handle (mirrors SkalaXC::MolGrid).
 */
typedef struct skalaxc_molgrid* skalaxc_molgrid_t;

/**
 * @brief Create a default molecular grid for a molecule.
 * @param mol Molecule handle.
 * @param grid Grid settings, or NULL for the built-in preset.
 * @param out Output owning handle on success (free with
 *            skalaxc_molgrid_destroy).
 * @return SKALAXC_SUCCESS on success, or an error status.
 */
skalaxc_status_t skalaxc_molgrid_create_default(
    skalaxc_molecule_t mol, const skalaxc_grid_settings_t* grid,
    skalaxc_molgrid_t* out) SKALAXC_EXPORT;

/**
 * @brief Destroy a molecular-grid handle (NULL is tolerated).
 * @param mg Molecular-grid handle to destroy.
 */
void skalaxc_molgrid_destroy(skalaxc_molgrid_t mg) SKALAXC_EXPORT;

#ifdef __cplusplus
}  // extern "C"
#endif
