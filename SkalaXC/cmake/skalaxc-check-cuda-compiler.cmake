# SPDX-License-Identifier: MIT

# Match the host-compiler bounds used by torch.utils.cpp_extension:
# https://github.com/pytorch/pytorch/blob/v2.13.0/torch/utils/cpp_extension.py
# PyTorch derives these bounds from the CUDA toolkit's crt/host_config.h and
# https://gist.github.com/ax3l/9489132. Upper bounds are exclusive. CUDA 13
# uses the conservative CUDA 13.0 Clang bound; newer toolkit headers may accept
# a newer Clang release.
function(_skalaxc_check_cuda_compiler_compatibility cuda_version)
  unset(_skalaxc_min_compiler)
  unset(_skalaxc_max_compiler)

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(_skalaxc_min_compiler "6")
    if(cuda_version VERSION_GREATER_EQUAL "12.0" AND cuda_version VERSION_LESS "12.3")
      set(_skalaxc_max_compiler "13")
    elseif(cuda_version VERSION_GREATER_EQUAL "12.3" AND cuda_version VERSION_LESS "12.8")
      set(_skalaxc_max_compiler "14")
    elseif(cuda_version VERSION_GREATER_EQUAL "12.8" AND cuda_version VERSION_LESS "13.0")
      set(_skalaxc_max_compiler "15")
    elseif(cuda_version VERSION_GREATER_EQUAL "13.0" AND cuda_version VERSION_LESS "14.0")
      set(_skalaxc_max_compiler "16")
    endif()
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(_skalaxc_min_compiler "7")
    if(cuda_version VERSION_GREATER_EQUAL "12.0" AND cuda_version VERSION_LESS "12.2")
      set(_skalaxc_max_compiler "15")
    elseif(cuda_version VERSION_GREATER_EQUAL "12.2" AND cuda_version VERSION_LESS "12.4")
      set(_skalaxc_max_compiler "16")
    elseif(cuda_version VERSION_GREATER_EQUAL "12.4" AND cuda_version VERSION_LESS "12.5")
      set(_skalaxc_max_compiler "17")
    elseif(cuda_version VERSION_GREATER_EQUAL "12.5" AND cuda_version VERSION_LESS "12.7")
      set(_skalaxc_max_compiler "18")
    elseif(cuda_version VERSION_GREATER_EQUAL "12.7" AND cuda_version VERSION_LESS "13.0")
      set(_skalaxc_max_compiler "19")
    elseif(cuda_version VERSION_GREATER_EQUAL "13.0" AND cuda_version VERSION_LESS "14.0")
      set(_skalaxc_max_compiler "21")
    endif()
  else()
    return()
  endif()

  if(NOT DEFINED _skalaxc_max_compiler)
    message(WARNING
      "No SkalaXC ${CMAKE_CXX_COMPILER_ID} host-compiler bounds are known for "
      "CUDA ${cuda_version}; CUDA compiler identification will perform the "
      "toolkit's native compatibility check.")
    return()
  endif()

    if("${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS "${_skalaxc_min_compiler}" OR
      NOT "${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS "${_skalaxc_max_compiler}")
    message(FATAL_ERROR
      "CUDA ${cuda_version} requires ${CMAKE_CXX_COMPILER_ID} host compiler "
      ">=${_skalaxc_min_compiler},<${_skalaxc_max_compiler}, but "
      "${CMAKE_CXX_COMPILER} is version ${CMAKE_CXX_COMPILER_VERSION}.")
  endif()

  message(STATUS
    "CUDA ${cuda_version} host compiler ${CMAKE_CXX_COMPILER_ID} "
    "${CMAKE_CXX_COMPILER_VERSION} satisfies "
    ">=${_skalaxc_min_compiler},<${_skalaxc_max_compiler}")
endfunction()