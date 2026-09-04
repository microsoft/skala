# skalaxc-gauxc.cmake
#
# Brings in an UNMODIFIED GauXC master source tree as a build-tree dependency so
# SkalaXC can reuse GauXC's internal components (LocalWorkDriver, LoadBalancer,
# MolecularWeights, collocation, HDF5 I/O, ...).
#
# GauXC is vendored as a git submodule at SkalaXC/external/GauXC, pinned to the
# exact commit the ML port was validated against. Initialize it with:
#     git submodule update --init SkalaXC/external/GauXC
#
# GauXC's `gauxc` target publishes ${GauXC}/src on its PUBLIC BUILD_INTERFACE
# include path, so a source-tree (add_subdirectory) build is what grants SkalaXC
# access to GauXC's private headers. An installed GauXC would NOT expose them --
# that is intentional: SkalaXC only reuses internals, it never ships them.
#
# GauXC is never modified. All SkalaXC-specific code lives under skala/SkalaXC.

# Path to the GauXC master source tree. Defaults to the bundled submodule; can
# be overridden to point at an existing checkout.
set(SKALAXC_GAUXC_SOURCE_DIR "${PROJECT_SOURCE_DIR}/external/GauXC"
    CACHE PATH "Path to an unmodified GauXC master source tree")

# Resolved path to GauXC test fixtures used by SkalaXC integration tests.
# This may differ from SKALAXC_GAUXC_SOURCE_DIR when GauXC is acquired through
# FetchContent fallback.
set(SKALAXC_GAUXC_REF_DATA_DIR "${SKALAXC_GAUXC_SOURCE_DIR}/tests/ref_data"
  CACHE PATH "Path to GauXC reference fixture directory")

# Commit the submodule is pinned to; also used to pin the FetchContent fallback
# so both acquisition paths yield an identical, reproducible GauXC.
set(SKALAXC_GAUXC_GIT_TAG "554bef7495b2a93f16a2fdedabf4e1cdcb3a1faf"
    CACHE STRING "GauXC commit to use when fetching (fallback only)")

# GauXC build options. Master has no ONEDFT/skala option -- SkalaXC provides the
# ML functional itself. Match GauXC's HDF5 support to the public SkalaXC option
# because the SkalaXC molecule and basis readers reuse GauXC's implementation.
set(GAUXC_ENABLE_TESTS  OFF                     CACHE BOOL "" FORCE)
set(GAUXC_ENABLE_C      OFF                     CACHE BOOL "" FORCE)
set(GAUXC_ENABLE_HDF5   ${SKALAXC_ENABLE_HDF5} CACHE BOOL "" FORCE)
set(GAUXC_ENABLE_OPENMP ${SKALAXC_ENABLE_OPENMP} CACHE BOOL "" FORCE)
set(GAUXC_ENABLE_MPI    ${SKALAXC_ENABLE_MPI}    CACHE BOOL "" FORCE)
set(GAUXC_ENABLE_CUDA   ${SKALAXC_ENABLE_CUDA}   CACHE BOOL "" FORCE)
set(GAUXC_ENABLE_MAGMA  OFF                      CACHE BOOL "" FORCE)

# GauXC currently declares HighFive with an older revision that can be slow or
# unreliable to fetch on some networks. Bootstrap a shallow clone here and
# force GauXC's FetchContent to reuse this local source tree, without modifying
# the vendored GauXC source files.
include(FetchContent)
if(SKALAXC_ENABLE_HDF5 AND
  NOT DEFINED FETCHCONTENT_SOURCE_DIR_HIGHFIVE)
  # Force these cache entries so existing build trees recover from stale values
  # left by previous configure attempts.
  set(HIGHFIVE_USE_BOOST OFF CACHE BOOL "" FORCE)
  set(HIGHFIVE_UNIT_TESTS OFF CACHE STRING "" FORCE)
  set(HIGHFIVE_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(HIGHFIVE_BUILD_DOCS OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(
    skalaxc_highfive_bootstrap
    GIT_REPOSITORY https://github.com/highfive-devs/HighFive.git
    GIT_TAG        v2.9.0
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE
    # Populate only; GauXC adds HighFive from this source tree below.
    SOURCE_SUBDIR  _skalaxc_source_only
  )
  FetchContent_MakeAvailable(skalaxc_highfive_bootstrap)
  set(FETCHCONTENT_SOURCE_DIR_HIGHFIVE
      "${skalaxc_highfive_bootstrap_SOURCE_DIR}"
      CACHE PATH
      "Local HighFive source directory used by GauXC FetchContent"
      FORCE)
endif()

# GauXC transitively uses deprecated CMake APIs and fetches an older HighFive
# release whose minimum CMake version triggers a deprecation warning on CMake
# 4.x. Suppress developer and deprecation warnings only while configuring this
# external dependency tree.
if(DEFINED CACHE{CMAKE_WARN_DEPRECATED})
  set(_skalaxc_cmake_warn_deprecated_defined TRUE)
  get_property(_skalaxc_cmake_warn_deprecated
    CACHE CMAKE_WARN_DEPRECATED PROPERTY VALUE)
else()
  set(_skalaxc_cmake_warn_deprecated_defined FALSE)
endif()
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL
    "Suppress deprecation warnings while configuring GauXC dependencies" FORCE)

if(DEFINED CACHE{CMAKE_SUPPRESS_DEVELOPER_WARNINGS})
  set(_skalaxc_cmake_suppress_developer_warnings_defined TRUE)
  get_property(_skalaxc_cmake_suppress_developer_warnings
    CACHE CMAKE_SUPPRESS_DEVELOPER_WARNINGS PROPERTY VALUE)
else()
  set(_skalaxc_cmake_suppress_developer_warnings_defined FALSE)
endif()
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE BOOL
    "Suppress developer warnings while configuring GauXC dependencies" FORCE)

# LibXC resets its policy baseline to CMake 3.1, where CMP0063 is unset.
# Apply modern visibility handling only while configuring this dependency tree.
if(DEFINED CMAKE_POLICY_DEFAULT_CMP0063)
  set(_skalaxc_policy_default_cmp0063_defined TRUE)
  set(_skalaxc_policy_default_cmp0063 "${CMAKE_POLICY_DEFAULT_CMP0063}")
else()
  set(_skalaxc_policy_default_cmp0063_defined FALSE)
endif()
set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)

# CMake 4 no longer accepts HighFive 2.9's CMake 3.1 policy baseline.
if(DEFINED CMAKE_POLICY_VERSION_MINIMUM)
  set(_skalaxc_policy_version_minimum_defined TRUE)
  set(_skalaxc_policy_version_minimum "${CMAKE_POLICY_VERSION_MINIMUM}")
else()
  set(_skalaxc_policy_version_minimum_defined FALSE)
endif()
if(NOT DEFINED CMAKE_POLICY_VERSION_MINIMUM OR
   CMAKE_POLICY_VERSION_MINIMUM VERSION_LESS 3.5)
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
endif()

if(EXISTS "${SKALAXC_GAUXC_SOURCE_DIR}/CMakeLists.txt")
  message(STATUS "SkalaXC: using GauXC source tree at ${SKALAXC_GAUXC_SOURCE_DIR}")
  set(_skalaxc_gauxc_source_dir "${SKALAXC_GAUXC_SOURCE_DIR}")
  set(SKALAXC_GAUXC_REF_DATA_DIR "${SKALAXC_GAUXC_SOURCE_DIR}/tests/ref_data"
      CACHE PATH "Path to GauXC reference fixture directory" FORCE)
  add_subdirectory(
    ${SKALAXC_GAUXC_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}/gauxc-master
    EXCLUDE_FROM_ALL
  )
else()
  # The submodule has not been initialized (and no override was given). Fall back
  # to fetching the pinned commit so the build still works (e.g. from a source
  # tarball) -- but `git submodule update --init` is the intended path.
  message(STATUS
    "SkalaXC: GauXC not found at ${SKALAXC_GAUXC_SOURCE_DIR} "
    "(run: git submodule update --init SkalaXC/external/GauXC). "
    "Falling back to FetchContent at ${SKALAXC_GAUXC_GIT_TAG}.")
  include(FetchContent)
  set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "Disable FC Updates")
  FetchContent_Declare(
    gauxc
    GIT_REPOSITORY https://github.com/wavefunction91/GauXC.git
    GIT_TAG        ${SKALAXC_GAUXC_GIT_TAG}
  )
  FetchContent_MakeAvailable(gauxc)
  set(_skalaxc_gauxc_source_dir "${gauxc_SOURCE_DIR}")
  set(SKALAXC_GAUXC_REF_DATA_DIR "${gauxc_SOURCE_DIR}/tests/ref_data"
      CACHE PATH "Path to GauXC reference fixture directory" FORCE)
endif()

# The IntegratorXX revision pinned by GauXC uses std::back_inserter without
# including <iterator>. libstdc++ currently exposes it transitively, but libc++
# does not. Patch only the populated build-tree dependency until this is fixed
# upstream; the bundled GauXC source tree remains unmodified.
FetchContent_GetProperties(integratorxx
  POPULATED _skalaxc_integratorxx_populated
  SOURCE_DIR _skalaxc_integratorxx_source_dir)
if(_skalaxc_integratorxx_populated)
  set(_skalaxc_integratorxx_batcher
      "${_skalaxc_integratorxx_source_dir}/include/integratorxx/batch/spherical_micro_batcher.hpp")
  if(EXISTS "${_skalaxc_integratorxx_batcher}")
    file(READ "${_skalaxc_integratorxx_batcher}"
         _skalaxc_integratorxx_batcher_contents)
    if(NOT _skalaxc_integratorxx_batcher_contents MATCHES
       "#include[ \t]*<[Ii][Tt][Ee][Rr][Aa][Tt][Oo][Rr]>")
      string(REPLACE
        "#include <integratorxx/type_traits.hpp>"
        "#include <integratorxx/type_traits.hpp>\n#include <iterator>"
        _skalaxc_integratorxx_batcher_patched
        "${_skalaxc_integratorxx_batcher_contents}")
      if(_skalaxc_integratorxx_batcher_patched STREQUAL
         _skalaxc_integratorxx_batcher_contents)
        message(FATAL_ERROR
          "Could not add the missing <iterator> include to IntegratorXX")
      endif()
      file(WRITE "${_skalaxc_integratorxx_batcher}"
           "${_skalaxc_integratorxx_batcher_patched}")
      message(STATUS "SkalaXC: added missing <iterator> include to IntegratorXX")
    endif()
  endif()
endif()
unset(_skalaxc_integratorxx_batcher)
unset(_skalaxc_integratorxx_batcher_contents)
unset(_skalaxc_integratorxx_batcher_patched)
unset(_skalaxc_integratorxx_populated)
unset(_skalaxc_integratorxx_source_dir)

if(_skalaxc_policy_version_minimum_defined)
  set(CMAKE_POLICY_VERSION_MINIMUM "${_skalaxc_policy_version_minimum}")
else()
  unset(CMAKE_POLICY_VERSION_MINIMUM)
endif()
unset(_skalaxc_policy_version_minimum)
unset(_skalaxc_policy_version_minimum_defined)

if(_skalaxc_policy_default_cmp0063_defined)
  set(CMAKE_POLICY_DEFAULT_CMP0063 "${_skalaxc_policy_default_cmp0063}")
else()
  unset(CMAKE_POLICY_DEFAULT_CMP0063)
endif()
unset(_skalaxc_policy_default_cmp0063)
unset(_skalaxc_policy_default_cmp0063_defined)

if(_skalaxc_cmake_warn_deprecated_defined)
  set(CMAKE_WARN_DEPRECATED "${_skalaxc_cmake_warn_deprecated}"
      CACHE BOOL "Whether to issue warnings for deprecated functionality" FORCE)
else()
  set(CMAKE_WARN_DEPRECATED ON CACHE BOOL
      "Whether to issue warnings for deprecated functionality" FORCE)
  unset(CMAKE_WARN_DEPRECATED CACHE)
endif()
unset(_skalaxc_cmake_warn_deprecated)
unset(_skalaxc_cmake_warn_deprecated_defined)

if(_skalaxc_cmake_suppress_developer_warnings_defined)
  set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS
      "${_skalaxc_cmake_suppress_developer_warnings}" CACHE BOOL
      "Suppress CMake developer warnings" FORCE)
else()
  set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS OFF CACHE BOOL
      "Suppress CMake developer warnings" FORCE)
  unset(CMAKE_SUPPRESS_DEVELOPER_WARNINGS CACHE)
endif()
unset(_skalaxc_cmake_suppress_developer_warnings)
unset(_skalaxc_cmake_suppress_developer_warnings_defined)

if(NOT TARGET gauxc)
  message(FATAL_ERROR "SkalaXC: GauXC target `gauxc` was not created")
endif()

# The pinned GauXC revision uses the removed C++17 `register` specifier in
# active CUDA sources and headers. Substitute patched build-tree copies until
# the fixes are available in the pinned upstream revision; never modify the
# GauXC source tree itself.
if(SKALAXC_ENABLE_CUDA)
  set(_skalaxc_gauxc_kernel_dir
      "${_skalaxc_gauxc_source_dir}/src/xc_integrator/local_work_driver/device/cuda/kernels")
  set(_skalaxc_gauxc_patch_dir
      "${CMAKE_CURRENT_BINARY_DIR}/skalaxc-gauxc-compat")
  file(MAKE_DIRECTORY "${_skalaxc_gauxc_patch_dir}")

  function(_skalaxc_remove_gauxc_register file_name)
    file(READ "${_skalaxc_gauxc_kernel_dir}/${file_name}"
         _skalaxc_gauxc_register_contents)
    string(REPLACE "register " ""
      _skalaxc_gauxc_register_patched_contents
      "${_skalaxc_gauxc_register_contents}")
    if(_skalaxc_gauxc_register_patched_contents STREQUAL
       _skalaxc_gauxc_register_contents)
      message(FATAL_ERROR
        "Could not apply the GauXC register-specifier compatibility patch to ${file_name}")
    endif()
    file(WRITE
      "${_skalaxc_gauxc_patch_dir}/${file_name}"
      "${_skalaxc_gauxc_register_patched_contents}")
  endfunction()

  _skalaxc_remove_gauxc_register(grid_to_center.cu)
  _skalaxc_remove_gauxc_register(uvvars_lda.hpp)
  _skalaxc_remove_gauxc_register(uvvars_gga.hpp)
  _skalaxc_remove_gauxc_register(uvvars_mgga.hpp)

  foreach(_skalaxc_gauxc_support_file IN ITEMS
          cuda_extensions.hpp grid_to_center.hpp uvvars.cu)
    configure_file(
      "${_skalaxc_gauxc_kernel_dir}/${_skalaxc_gauxc_support_file}"
      "${_skalaxc_gauxc_patch_dir}/${_skalaxc_gauxc_support_file}"
      COPYONLY)
  endforeach()

  get_target_property(_skalaxc_gauxc_sources gauxc SOURCES)
  set(_skalaxc_gauxc_patched_sources)
  set(_skalaxc_gauxc_grid_replaced FALSE)
  set(_skalaxc_gauxc_uvvars_replaced FALSE)
  foreach(_skalaxc_gauxc_source IN LISTS _skalaxc_gauxc_sources)
    if(_skalaxc_gauxc_source MATCHES "(^|/)kernels/grid_to_center\\.cu$")
      list(APPEND _skalaxc_gauxc_patched_sources
           "${_skalaxc_gauxc_patch_dir}/grid_to_center.cu")
      set(_skalaxc_gauxc_grid_replaced TRUE)
    elseif(_skalaxc_gauxc_source MATCHES "(^|/)kernels/uvvars\\.cu$")
      list(APPEND _skalaxc_gauxc_patched_sources
           "${_skalaxc_gauxc_patch_dir}/uvvars.cu")
      set(_skalaxc_gauxc_uvvars_replaced TRUE)
    else()
      list(APPEND _skalaxc_gauxc_patched_sources "${_skalaxc_gauxc_source}")
    endif()
  endforeach()
  if(NOT _skalaxc_gauxc_grid_replaced)
    message(FATAL_ERROR
      "Could not locate GauXC grid_to_center.cu in the gauxc target")
  endif()
  if(NOT _skalaxc_gauxc_uvvars_replaced)
    message(FATAL_ERROR
      "Could not locate GauXC uvvars.cu in the gauxc target")
  endif()
  set_property(TARGET gauxc PROPERTY SOURCES "${_skalaxc_gauxc_patched_sources}")
  message(STATUS
    "SkalaXC: patched GauXC register specifiers in build-tree copies")

  unset(_skalaxc_gauxc_grid_replaced)
  unset(_skalaxc_gauxc_kernel_dir)
  unset(_skalaxc_gauxc_patch_dir)
  unset(_skalaxc_gauxc_patched_sources)
  unset(_skalaxc_gauxc_source)
  unset(_skalaxc_gauxc_sources)
  unset(_skalaxc_gauxc_support_file)
  unset(_skalaxc_gauxc_uvvars_replaced)
endif()
unset(_skalaxc_gauxc_source_dir)

# Keep warnings actionable by silencing the unmodified external GauXC target
# without affecting SkalaXC sources or consumers.
if(MSVC)
  set(_skalaxc_gauxc_host_disable_warnings /w)
else()
  set(_skalaxc_gauxc_host_disable_warnings -w)
endif()
target_compile_options(gauxc PRIVATE
  "$<$<COMPILE_LANGUAGE:C>:${_skalaxc_gauxc_host_disable_warnings}>"
  "$<$<COMPILE_LANGUAGE:CXX>:${_skalaxc_gauxc_host_disable_warnings}>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-w>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${_skalaxc_gauxc_host_disable_warnings}>"
)
unset(_skalaxc_gauxc_host_disable_warnings)

# SkalaXC privately embeds GauXC and ExchCXX static CUDA archives in its shared
# library, so their host and device objects must be PIC.
set_target_properties(gauxc PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(TARGET exchcxx)
  set_target_properties(exchcxx PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# Treat third-party dependency headers as system includes for downstream
# consumers to reduce warning noise from external templates.
function(_skalaxc_mark_interface_includes_system target_name)
  if(NOT TARGET ${target_name})
    return()
  endif()

  get_target_property(_skalaxc_iface_includes ${target_name} INTERFACE_INCLUDE_DIRECTORIES)
  if(_skalaxc_iface_includes)
    set_target_properties(${target_name} PROPERTIES
      INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_skalaxc_iface_includes}")
  endif()
endfunction()

_skalaxc_mark_interface_includes_system(integratorxx)
_skalaxc_mark_interface_includes_system(exchcxx)
_skalaxc_mark_interface_includes_system(gauxc)

# Sanity: master must NOT carry the skala/onedft surface API -- SkalaXC owns that.
if(GAUXC_HAS_ONEDFT)
  message(FATAL_ERROR
    "SkalaXC: GauXC source tree reports GAUXC_HAS_ONEDFT. SkalaXC must build "
    "against an unmodified GauXC master (no onedft/skala surface API).")
endif()
