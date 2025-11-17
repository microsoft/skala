find_package(nlohmann_json)
if( NOT nlohmann_json_FOUND )

  message( STATUS "Could Not Find nlohmann_json... Building" )
  message( STATUS "NLOHMANN_JSON REPO = ${GAUXC_NLOHMANN_JSON_REPOSITORY}" )

  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY ${GAUXC_NLOHMANN_JSON_REPOSITORY}
    GIT_TAG        ${GAUXC_NLOHMANN_JSON_REVISION}
  )

  FetchContent_GetProperties( nlohmann_json )
  if( NOT nlohmann_json_POPULATED )
    FetchContent_Populate( nlohmann_json )
  endif()

  add_library( nlohmann_json::nlohmann_json INTERFACE IMPORTED )
  set_target_properties( nlohmann_json::nlohmann_json PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${nlohmann_json_SOURCE_DIR}/include
  )
endif()

# store and restore CMAKE_CUDA_ARCHITECTURES if Torch clobbers it
set(_PREV_CUDA_ARCHS "${CMAKE_CUDA_ARCHITECTURES}")
find_package(Torch REQUIRED)
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "OFF")
  set(CMAKE_CUDA_ARCHITECTURES "${_PREV_CUDA_ARCHS}" CACHE STRING "Restore CUDA archs after Torch override" FORCE)
  message(WARNING "Torch set CMAKE_CUDA_ARCHITECTURES to OFF. Restored previous value: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
list(REMOVE_ITEM TORCH_LIBRARIES torch::nvtoolsext)
message(STATUS "Torch libraries without nvtoolsext: ${TORCH_LIBRARIES}")
