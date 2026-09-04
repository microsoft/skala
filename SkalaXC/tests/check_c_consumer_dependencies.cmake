if(NOT CONSUMER_PATH OR (NOT READELF_EXECUTABLE AND NOT OTOOL_EXECUTABLE))
  message(FATAL_ERROR
    "CONSUMER_PATH and either READELF_EXECUTABLE or OTOOL_EXECUTABLE are required")
endif()

if(OTOOL_EXECUTABLE)
  execute_process(
    COMMAND "${OTOOL_EXECUTABLE}" -L "${CONSUMER_PATH}"
    RESULT_VARIABLE inspector_result
    OUTPUT_VARIABLE inspector_output
    ERROR_VARIABLE inspector_error)
  set(required_dependency "libskalaxc[^\n]*[.]dylib")
  set(private_dependencies "gauxc|torch|c10|libc[+][+]")
  set(inspector_name "otool")
else()
  execute_process(
    COMMAND "${READELF_EXECUTABLE}" -d "${CONSUMER_PATH}"
    RESULT_VARIABLE inspector_result
    OUTPUT_VARIABLE inspector_output
    ERROR_VARIABLE inspector_error)
  set(required_dependency "\\(needed\\)[^\n]*\\[libskalaxc[^]]*\\]")
  set(private_dependencies
    "\\(needed\\)[^\n]*\\[[^]]*(gauxc|torch|c10|stdc[+][+])[^]]*\\]")
  set(inspector_name "readelf")
endif()
if(NOT inspector_result EQUAL 0)
  message(FATAL_ERROR
    "${inspector_name} failed for ${CONSUMER_PATH}: ${inspector_error}")
endif()

string(TOLOWER "${inspector_output}" dependencies)
if(NOT dependencies MATCHES "${required_dependency}")
  message(FATAL_ERROR "${CONSUMER_PATH} does not directly depend on libskalaxc")
endif()
if(dependencies MATCHES "${private_dependencies}")
  message(FATAL_ERROR
    "${CONSUMER_PATH} directly exposes a private C++ dependency:\n"
    "${inspector_output}")
endif()

message(STATUS "Verified pure-C consumer dynamic dependency isolation")
