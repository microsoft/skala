if(NOT LIBRARY_PATH OR (NOT READELF_EXECUTABLE AND NOT OTOOL_EXECUTABLE))
  message(FATAL_ERROR
    "LIBRARY_PATH and either READELF_EXECUTABLE or OTOOL_EXECUTABLE are required")
endif()

if(OTOOL_EXECUTABLE)
  execute_process(
    COMMAND "${OTOOL_EXECUTABLE}" -L "${LIBRARY_PATH}"
    RESULT_VARIABLE inspector_result
    OUTPUT_VARIABLE inspector_output
    ERROR_VARIABLE inspector_error)
  set(private_dependencies "gauxc|exchcxx|integratorxx|eigen|nlohmann")
  set(inspector_name "otool")
else()
  execute_process(
    COMMAND "${READELF_EXECUTABLE}" -d "${LIBRARY_PATH}"
    RESULT_VARIABLE inspector_result
    OUTPUT_VARIABLE inspector_output
    ERROR_VARIABLE inspector_error)
  set(private_dependencies
    "\\(needed\\)[^\n]*\\[[^]]*(gauxc|exchcxx|integratorxx|eigen|nlohmann)[^]]*\\]")
  set(inspector_name "readelf")
endif()
if(NOT inspector_result EQUAL 0)
  message(FATAL_ERROR
    "${inspector_name} failed for ${LIBRARY_PATH}: ${inspector_error}")
endif()

string(TOLOWER "${inspector_output}" dependencies)
if(dependencies MATCHES "${private_dependencies}")
  message(FATAL_ERROR
    "${LIBRARY_PATH} dynamically exposes an embedded private dependency:\n"
    "${inspector_output}")
endif()

message(STATUS "Verified libskalaxc dynamic dependency isolation")
