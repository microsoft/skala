if(NOT DEFINED Skala_CLI11_URL)
  include(skala-dep-versions)
endif()
find_package(CLI11 QUIET CONFIG)
if(NOT CLI11_FOUND)
  message(STATUS "Could not find CLI11... Building CLI11 from source")
  message(STATUS "CLI11 URL: ${Skala_CLI11_URL}")

  FetchContent_Declare(
    cli11
    URL ${Skala_CLI11_URL}
    URL_HASH SHA256=${Skala_CLI11_SHA256}
    DOWNLOAD_EXTRACT_TIMESTAMP ON
  )

  FetchContent_GetProperties(cli11)
  if(NOT cli11_POPULATED)
    FetchContent_Populate(cli11)
  endif()

  add_library( CLI11::CLI11 INTERFACE IMPORTED )
  set_target_properties( CLI11::CLI11 PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${cli11_SOURCE_DIR}/include
  )
endif()