# Verify that the built shared library exposes only SkalaXC's public C and C++
# ABI. The CTest registration supplies the platform's nm executable, the
# library path, and either MACHO or ELF mode. This script normalizes the two nm
# output conventions, rejects leaked dependency/private symbols, and confirms
# that both the C and C++ API surfaces have at least one exported symbol.

if(NOT NM_EXECUTABLE OR NOT LIBRARY_PATH)
  message(FATAL_ERROR "NM_EXECUTABLE and LIBRARY_PATH are required")
endif()

if(NM_MODE STREQUAL "MACHO")
  set(nm_arguments -gU -P)
  # Depending on whether CMAKE_NM resolves to Apple nm or llvm-nm, and whether
  # POSIX output mode canonicalizes Mach-O names, an Itanium C++ symbol can be
  # reported as __ZN..., _ZN..., or ZN.... Match those equivalent spellings
  # directly rather than stripping an underscore that may already be absent.
  set(c_symbol_pattern "^_*(skalaxc_|SkalaXC_)")
  set(cxx_symbol_pattern
      "^_*Z(N7SkalaXC|NK7SkalaXC|TIN7SkalaXC|TSN7SkalaXC|TVN7SkalaXC)")
else()
  set(nm_arguments -D --defined-only --format=posix)
  set(c_symbol_pattern "^(__skalaxc_|skalaxc_|SkalaXC_)")
  set(cxx_symbol_pattern
      "^(_ZN7SkalaXC|_ZNK7SkalaXC|_ZTIN7SkalaXC|_ZTSN7SkalaXC|_ZTVN7SkalaXC)")
endif()

execute_process(
  COMMAND "${NM_EXECUTABLE}" ${nm_arguments} "${LIBRARY_PATH}"
  RESULT_VARIABLE nm_result
  OUTPUT_VARIABLE nm_output
  ERROR_VARIABLE nm_error)
if(NOT nm_result EQUAL 0)
  message(FATAL_ERROR "nm failed for ${LIBRARY_PATH}: ${nm_error}")
endif()

string(REPLACE "\n" ";" symbol_lines "${nm_output}")
set(c_symbol_count 0)
set(cxx_symbol_count 0)
set(unexpected_symbols)
foreach(symbol_line IN LISTS symbol_lines)
  string(STRIP "${symbol_line}" symbol_line)
  if(symbol_line STREQUAL "")
    continue()
  endif()

  string(REGEX MATCH "^([^ \t]+)[ \t]" symbol_match "${symbol_line}")
  set(symbol "${CMAKE_MATCH_1}")
  if(symbol MATCHES "${c_symbol_pattern}")
    math(EXPR c_symbol_count "${c_symbol_count} + 1")
  elseif(symbol MATCHES "${cxx_symbol_pattern}")
    math(EXPR cxx_symbol_count "${cxx_symbol_count} + 1")
  else()
    list(APPEND unexpected_symbols "${symbol}")
  endif()
endforeach()

if(unexpected_symbols)
  list(JOIN unexpected_symbols "\n  " unexpected_lines)
  message(FATAL_ERROR
    "${LIBRARY_PATH} exports symbols outside the SkalaXC ABI:\n  "
    "${unexpected_lines}")
endif()
if(c_symbol_count EQUAL 0 OR cxx_symbol_count EQUAL 0)
  message(FATAL_ERROR
    "Expected both C and C++ SkalaXC exports, found ${c_symbol_count} C and "
    "${cxx_symbol_count} C++ symbols")
endif()

message(STATUS
  "Verified ${c_symbol_count} C and ${cxx_symbol_count} C++ SkalaXC exports")
