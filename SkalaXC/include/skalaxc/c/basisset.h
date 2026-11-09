/**
 * @file
 * @brief SkalaXC C API: basis-set handle.
 *
 * Mirrors SkalaXC::BasisSet<double>. Part of the public C API; include
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
 * @brief Opaque basis-set handle (mirrors SkalaXC::BasisSet<double>).
 */
typedef struct skalaxc_basisset* skalaxc_basisset_t;

/**
 * @brief Create an empty basis set.
 * @param out Output owning handle on success (free with
 *            skalaxc_basisset_destroy).
 * @return SKALAXC_SUCCESS on success, or an error status.
 */
skalaxc_status_t skalaxc_basisset_create(skalaxc_basisset_t* out)
    SKALAXC_EXPORT;

/**
 * @brief Append a contracted Gaussian shell to a basis set.
 * @param basis Basis-set handle.
 * @param l Angular momentum.
 * @param pure 1 = pure spherical, 0 = cartesian.
 * @param center_xyz [3] Shell center (bohr).
 * @param nprim Primitive count (1..32).
 * @param exponents [nprim] Primitive exponents.
 * @param coefficients [nprim] Contraction coefficients.
 * @param normalize 1 to normalize the shell inside the library, 0 to take the
 *                  coefficients as-is.
 * @return SKALAXC_SUCCESS on success, or an error status.
 */
skalaxc_status_t skalaxc_basisset_add_shell(
    skalaxc_basisset_t basis, int32_t l, int32_t pure, const double* center_xyz,
    int32_t nprim, const double* exponents, const double* coefficients,
    int32_t normalize) SKALAXC_EXPORT;

/**
 * @brief Create a basis set from native arrays.
 * @param nshells Number of basis shells.
 * @param shell_l [nshells] Angular momentum per shell.
 * @param shell_pure [nshells] 1 = pure spherical, 0 = cartesian.
 * @param shell_xyz [3*nshells] Shell centers (bohr), shell-major.
 * @param shell_nprim [nshells] Primitive count per shell.
 * @param prim_exp [sum(shell_nprim)] Primitive exponents, concatenated.
 * @param prim_coeff [sum(shell_nprim)] Contraction coefficients, concatenated.
 * @param out Output owning handle on success.
 * @return SKALAXC_SUCCESS on success, or an error status.
 */
skalaxc_status_t skalaxc_basisset_from_arrays(
    int64_t nshells, const int32_t* shell_l, const int32_t* shell_pure,
    const double* shell_xyz, const int32_t* shell_nprim, const double* prim_exp,
    const double* prim_coeff, skalaxc_basisset_t* out) SKALAXC_EXPORT;

#ifdef SKALAXC_HAS_HDF5
/**
 * @brief Create a basis set from an HDF5 record.
 * @param path Path to the HDF5 file.
 * @param dset Dataset/record name (e.g. "/BASIS").
 * @param out Output owning handle on success.
 * @return SKALAXC_SUCCESS on success, or an error status.
 */
skalaxc_status_t skalaxc_basisset_from_hdf5(
    const char* path, const char* dset, skalaxc_basisset_t* out) SKALAXC_EXPORT;
#endif

/**
 * @brief Return the number of basis functions, or -1 if basis is null.
 * @param basis Basis-set handle.
 * @return Number of basis functions.
 */
int64_t skalaxc_basisset_nbf(skalaxc_basisset_t basis) SKALAXC_EXPORT;

/**
 * @brief Destroy a basis-set handle (NULL is tolerated).
 * @param basis Basis-set handle to destroy.
 */
void skalaxc_basisset_destroy(skalaxc_basisset_t basis) SKALAXC_EXPORT;

#ifdef __cplusplus
}  // extern "C"
#endif
