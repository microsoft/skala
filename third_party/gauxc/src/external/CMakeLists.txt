#
# GauXC Copyright (c) 2020-2024, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy).
#
# (c) 2024-2025, Microsoft Corporation
#
# All rights reserved.
#
# See LICENSE.txt for details
#
if( GAUXC_ENABLE_HDF5 )
  include(FetchContent)
  find_package(HDF5)
  if(HDF5_FOUND)
    set(GAUXC_HAS_HDF5 TRUE CACHE BOOL "" FORCE)
    message(STATUS "Enabling HDF5 Bindings")
    message(STATUS "HighFive REPO = ${GAUXC_HIGHFIVE_REPOSITORY}")
    message(STATUS "HighFive REV  = ${GAUXC_HIGHFIVE_REVISION}  ")
    FetchContent_Declare( HighFive
      GIT_REPOSITORY ${GAUXC_HIGHFIVE_REPOSITORY}
      GIT_TAG        ${GAUXC_HIGHFIVE_REVISION}  
    )
    
    set(HIGHFIVE_USE_BOOST OFF CACHE BOOL "" )
    set(HIGHFIVE_UNIT_TESTS OFF CACHE BOOL "" )
    set(HIGHFIVE_EXAMPLES OFF CACHE BOOL "" )
    #set(HIGHFIVE_PARALLEL_HDF5 ON CACHE BOOL "" )
    set(HIGHFIVE_BUILD_DOCS OFF CACHE BOOL "" )
    FetchContent_MakeAvailable( HighFive )
    
    target_sources( gauxc PRIVATE hdf5_write.cxx hdf5_read.cxx )
    target_link_libraries( gauxc PUBLIC HighFive )
  else()
    message(WARNING "GAUXC_ENABLE_HDF5 was enabled, but HDF5 was not found, Disabling HDF5 Bindings")
  endif()
else()
  message(STATUS "Disabling HDF5 Bindings")
endif()
