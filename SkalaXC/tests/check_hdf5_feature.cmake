if(NOT DEFINED EXPECT_HDF5 OR
   NOT DEFINED PUBLIC_CONFIG_HEADER OR
   NOT DEFINED INTERNAL_CONFIG_HEADER OR
   NOT DEFINED PACKAGE_CONFIG)
  message(FATAL_ERROR
    "EXPECT_HDF5, PUBLIC_CONFIG_HEADER, INTERNAL_CONFIG_HEADER, and "
    "PACKAGE_CONFIG are required")
endif()

foreach(_skalaxc_file IN ITEMS
    "${PUBLIC_CONFIG_HEADER}"
    "${INTERNAL_CONFIG_HEADER}"
    "${PACKAGE_CONFIG}")
  if(NOT EXISTS "${_skalaxc_file}")
    message(FATAL_ERROR "Expected generated file does not exist: ${_skalaxc_file}")
  endif()
endforeach()

file(READ "${PUBLIC_CONFIG_HEADER}" _skalaxc_public_config)
file(READ "${INTERNAL_CONFIG_HEADER}" _skalaxc_internal_config)
file(READ "${PACKAGE_CONFIG}" _skalaxc_package_config)

if(EXPECT_HDF5)
  set(_skalaxc_expected_header
      "#[ \t]*define[ \t]+SKALAXC_HAS_HDF5([ \t]+1)?")
  set(_skalaxc_expected_package
      "set\\(SkalaXC_HDF5_ENABLED[ \t]+(1|ON|TRUE)\\)")
else()
  set(_skalaxc_expected_header
      "#[ \t]*undef[ \t]+SKALAXC_HAS_HDF5")
  set(_skalaxc_expected_package
      "set\\(SkalaXC_HDF5_ENABLED[ \t]+(0|OFF|FALSE)\\)")
endif()

foreach(_skalaxc_header IN ITEMS
    _skalaxc_public_config
    _skalaxc_internal_config)
  if(NOT "${${_skalaxc_header}}" MATCHES "${_skalaxc_expected_header}")
    message(FATAL_ERROR
      "${_skalaxc_header} does not report the expected HDF5 feature state")
  endif()
endforeach()

if(NOT _skalaxc_package_config MATCHES "${_skalaxc_expected_package}")
  message(FATAL_ERROR
    "SkalaXCConfig.cmake does not report the expected HDF5 feature state")
endif()

message(STATUS "Verified generated HDF5 feature state: ${EXPECT_HDF5}")