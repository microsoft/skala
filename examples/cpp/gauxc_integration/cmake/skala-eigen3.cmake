if(NOT DEFINED Skala_Eigen3_URL)
  include(skala-dep-versions)
endif()
find_package(Eigen3 CONFIG HINTS ${EIGEN3_ROOT_DIR})
if(NOT Eigen3_FOUND)
  message(STATUS "Could Not Find Eigen3... Building Eigen3 from source")
  message(STATUS "EIGEN3 REPO = ${Skala_Eigen3_URL}")

  FetchContent_Declare(
    eigen3
    URL ${Skala_Eigen3_URL}
    URL_HASH SHA256=${Skala_Eigen3_SHA256}
    DOWNLOAD_EXTRACT_TIMESTAMP ON
  )

  FetchContent_GetProperties(eigen3)
  if(NOT eigen3_POPULATED)
    FetchContent_Populate(eigen3)
  endif()

  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(
    Eigen3::Eigen
    PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${eigen3_SOURCE_DIR}
  )
endif()

