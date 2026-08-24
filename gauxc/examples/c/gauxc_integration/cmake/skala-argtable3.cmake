if(NOT DEFINED Skala_Argtable3_URL)
  include(skala-dep-versions)
endif()
find_package(Argtable3 QUIET CONFIG)
if(NOT Argtable3_FOUND)
  include(FetchContent)

  message(STATUS "Could not find Argtable3... Building Argtable3 from source")
  message(STATUS "Argtable3 URL: ${Skala_Argtable3_URL}")

  FetchContent_Declare(
    argtable3
    URL ${Skala_Argtable3_URL}
    URL_HASH SHA256=${Skala_Argtable3_SHA256}
    DOWNLOAD_EXTRACT_TIMESTAMP ON
  )

  # The example only needs the library target. Disabling upstream tests/examples
  # avoids extra tooling requirements in CI (e.g., dos2unix).
  set(ARGTABLE3_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  set(ARGTABLE3_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(argtable3)
endif()
