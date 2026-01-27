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
  FetchContent_MakeAvailable(Argtable3)
endif()
