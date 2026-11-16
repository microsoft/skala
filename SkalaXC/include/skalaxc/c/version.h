/**
 * @file
 * @brief SkalaXC C API: library version information.
 *
 * Part of the public C API; include <skalaxc/skalaxc.h> to get the whole API.
 */
#pragma once

#include <skalaxc/skalaxc_export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Return the SkalaXC semantic version.
 *
 * The returned null-terminated string has static storage duration and must not
 * be modified or freed.
 * @return Null-terminated semantic version string.
 */
const char* skalaxc_version(void) SKALAXC_EXPORT;

#ifdef __cplusplus
}  // extern "C"
#endif
