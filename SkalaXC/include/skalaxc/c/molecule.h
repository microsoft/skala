/**
 * @file
 * @brief SkalaXC C API: molecule handle.
 *
 * Mirrors SkalaXC::Molecule. Part of the public C API; include
 * <skalaxc/skalaxc.h> to get the whole API.
 */
#pragma once

#include <stdint.h>

#include <skalaxc/c/config.h>
#include <skalaxc/c/status.h>
#include <skalaxc/skalaxc_export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque molecule handle (mirrors SkalaXC::Molecule).
 */
typedef struct skalaxc_molecule* skalaxc_molecule_t;

/**
 * @brief Create an empty molecule.
 * @param out Output owning handle on success (free with
 *            skalaxc_molecule_destroy).
 * @return SKALAXC_SUCCESS on success, or an error status.
 */
skalaxc_status_t skalaxc_molecule_create(skalaxc_molecule_t* out)
    SKALAXC_EXPORT;

/**
 * @brief Append an atom to a molecule.
 * @param mol Molecule handle.
 * @param atomic_number Atomic number of the nucleus.
 * @param x Nuclear x-coordinate (bohr).
 * @param y Nuclear y-coordinate (bohr).
 * @param z Nuclear z-coordinate (bohr).
 * @return SKALAXC_SUCCESS on success, or an error status.
 */
skalaxc_status_t skalaxc_molecule_add_atom(skalaxc_molecule_t mol,
                                           int64_t atomic_number, double x,
                                           double y, double z) SKALAXC_EXPORT;

/**
 * @brief Create a molecule from native arrays.
 * @param natoms Number of atoms.
 * @param Z [natoms] Atomic numbers.
 * @param atom_xyz [3*natoms] Nuclear coordinates (bohr), atom-major.
 * @param out Output owning handle on success.
 * @return SKALAXC_SUCCESS on success, or an error status.
 */
skalaxc_status_t skalaxc_molecule_from_arrays(
    int64_t natoms, const int64_t* Z, const double* atom_xyz,
    skalaxc_molecule_t* out) SKALAXC_EXPORT;

#ifdef SKALAXC_HAS_HDF5
/**
 * @brief Create a molecule from an HDF5 record.
 * @param path Path to the HDF5 file.
 * @param dset Dataset/record name (e.g. "/MOLECULE").
 * @param out Output owning handle on success.
 * @return SKALAXC_SUCCESS on success, or an error status.
 */
skalaxc_status_t skalaxc_molecule_from_hdf5(
    const char* path, const char* dset, skalaxc_molecule_t* out) SKALAXC_EXPORT;
#endif

/**
 * @brief Return the number of atoms, or -1 if mol is null.
 * @param mol Molecule handle.
 * @return Number of atoms.
 */
int64_t skalaxc_molecule_natoms(skalaxc_molecule_t mol) SKALAXC_EXPORT;

/**
 * @brief Destroy a molecule handle (NULL is tolerated).
 * @param mol Molecule handle to destroy.
 */
void skalaxc_molecule_destroy(skalaxc_molecule_t mol) SKALAXC_EXPORT;

#ifdef __cplusplus
}  // extern "C"
#endif
