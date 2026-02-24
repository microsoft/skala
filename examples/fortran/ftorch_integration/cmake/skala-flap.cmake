if(NOT DEFINED Skala_FLAP_URL)
  include(skala-dep-versions)
endif()
find_package(FLAP QUIET CONFIG)
if(NOT FLAP_FOUND)
  include(FetchContent)

  message(STATUS "Could not find FLAP... Building FLAP from source")
  message(STATUS "FLAP URL: ${Skala_FLAP_URL}")

  FetchContent_Declare(
    flap
    URL ${Skala_FLAP_URL}
    URL_HASH SHA256=${Skala_FLAP_SHA256}
    DOWNLOAD_EXTRACT_TIMESTAMP ON
  )
  FetchContent_MakeAvailable(flap)
endif()

if(NOT TARGET FLAP::FLAP)
  add_library(FLAP::FLAP INTERFACE IMPORTED)
  target_link_libraries(FLAP::FLAP INTERFACE FLAP)
endif()