if(NOT DEFINED Skala_SDFTD3_URL)
  include(skala-dep-versions)
endif()
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(SDFTD3 QUIET "s-dftd3")
else()
  set(SDFTD3_FOUND OFF)
endif()
if(SDFTD3_FOUND)
  message(STATUS "Found s-dftd3 via pkg-config")

  add_library("s-dftd3::s-dftd3" INTERFACE IMPORTED)
  target_link_libraries(
    "s-dftd3::s-dftd3"
    INTERFACE
    "${SDFTD3_LINK_LIBRARIES}"
  )
  target_include_directories(
    "s-dftd3::s-dftd3"
    INTERFACE
    "${SDFTD3_INCLUDE_DIRS}"
  )
else()
  include(FetchContent)

  message(STATUS "Could not find s-dftd3... Building s-dftd3 from source")
  message(STATUS "S-DFTD3 URL: ${Skala_SDFTD3_URL}")

  set("s-dftd3-dependency-method" "subproject" CACHE STRING "Method to acquire s-dftd3 dependencies" FORCE)

  FetchContent_Declare(
    sdftd3
    URL ${Skala_SDFTD3_URL}
    URL_HASH SHA256=${Skala_SDFTD3_SHA256}
    DOWNLOAD_EXTRACT_TIMESTAMP ON
  )
  FetchContent_MakeAvailable(sdftd3)

  if(NOT TARGET "s-dftd3::s-dftd3")
    add_library("s-dftd3::s-dftd3" INTERFACE IMPORTED)
    target_link_libraries(
      "s-dftd3::s-dftd3"
      INTERFACE
      s-dftd3
    )
  endif()
endif()