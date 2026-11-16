/**
 * @file
 * @brief SkalaXC C API: status codes and error reporting.
 *
 * Part of the public C API; include <skalaxc/skalaxc.h> to get the whole API.
 */
#pragma once

#include <skalaxc/skalaxc_export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Status codes returned by SkalaXC C API functions.
 */
typedef enum {
  SKALAXC_SUCCESS = 0,  ///< Success
  SKALAXC_ERROR = 1,    ///< Generic failure (see skalaxc_last_error_message)
  SKALAXC_INVALID_ARGUMENT =
      2  ///< A null pointer, invalid dimension, or invalid enum was passed
} skalaxc_status_t;

/**
 * @brief Thread-local message describing the most recent failure on the calling
 * thread.
 *
 * The returned pointer remains valid until the next failure on the same thread.
 * Long implementation or dependency messages may be truncated. The caller must
 * not modify or free the returned storage.
 * @return Null-terminated error string (empty if no error has occurred).
 */
const char* skalaxc_last_error_message(void) SKALAXC_EXPORT;

#ifdef __cplusplus
}  // extern "C"
#endif
