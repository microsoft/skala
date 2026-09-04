/**
 * @file
 * @brief SkalaXC symbol-visibility macro.
 *
 * SkalaXC is built with -fvisibility=hidden; only symbols annotated with
 * SKALAXC_EXPORT are placed in the shared library's dynamic symbol table. This
 * is the compile-time half of the ABI-isolation guarantee (the link-time half
 * is the version script + --exclude-libs,ALL that hide the statically embedded
 * GauXC / LibTorch symbols).
 */
#pragma once

#if defined(_WIN32) || defined(__CYGWIN__)
#ifdef SKALAXC_BUILDING_LIBRARY
#define SKALAXC_EXPORT __declspec(dllexport)
#else
#define SKALAXC_EXPORT __declspec(dllimport)
#endif
#else
#define SKALAXC_EXPORT __attribute__((visibility("default")))
#endif
