if(NOT DEFINED Skala_FTorch_URL)
  include(skala-dep-versions)
endif()
find_package(FTorch QUIET)
if(NOT FTorch_FOUND)
  include(FetchContent)

  message(STATUS "Could not find FTorch... Building FTorch from source")
  message(STATUS "FTorch URL: ${Skala_FTorch_URL}")

  FetchContent_Declare(
    ftorch
    URL ${Skala_FTorch_URL}
    URL_HASH SHA256=${Skala_FTorch_SHA256}
    DOWNLOAD_EXTRACT_TIMESTAMP ON
  )
  FetchContent_MakeAvailable(ftorch)
endif()