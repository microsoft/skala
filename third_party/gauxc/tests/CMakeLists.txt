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
if(NOT GAUXC_HAS_HDF5)
  message(WARNING "GauXC Tests were ENABLED but HDF5 was not FOUND/ENABLED. Please enable and ensure discovery of HDF5. Skipping Tests")
  return()
endif()
message( STATUS "Building GauXC Tests" )
add_library( gauxc_catch2 INTERFACE )

find_package( Catch2 CONFIG QUIET )
if( NOT Catch2_FOUND )

  FetchContent_Declare(
    catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v2.13.10
  )
  
  set(CATCH_BUILD_TESTING OFF CACHE BOOL "Build SelfTest project" FORCE)
  set(CATCH_INSTALL_DOCS OFF CACHE BOOL "Install documentation alongside library" FORCE)
  set(CATCH_INSTALL_HELPERS OFF CACHE BOOL "Install contrib alongside library" FORCE)

  FetchContent_MakeAvailable( catch2 )
  target_link_libraries( gauxc_catch2 INTERFACE Catch2::Catch2 )

else()

  target_link_libraries( gauxc_catch2 INTERFACE Catch2::Catch2 )

endif()

include( gauxc-eigen3 )
include( gauxc-cereal )

#add_library( catch2_main STATIC ut_main.cxx)
#target_link_libraries( catch2_main PUBLIC gauxc_catch2 )
#
#add_executable( gauxc_moltypes_test moltypes_test.cxx )
#target_link_libraries( gauxc_moltypes_test PUBLIC gauxc catch2_main )
#
#add_executable( gauxc_grid_test grid_test.cxx )
#target_link_libraries( gauxc_grid_test PUBLIC gauxc catch2_main )

add_executable( gauxc_test 
  ut_main.cxx 
  moltypes_test.cxx 
  grid_test.cxx 
  molgrid_test.cxx 
  basisset_test.cxx 
  load_balancer_test.cxx 
  xc_integrator.cxx 
  environment.cxx
  collocation.cxx
  weights.cxx
  weight_derivative_test.cxx
  standards.cxx 
  runtime.cxx
  basis/parse_basis.cxx
  dd_psi_potential_test.cxx
  onedft_test.cxx
  2nd_derivative_test.cxx
)
target_link_libraries( gauxc_test PUBLIC gauxc gauxc_catch2 Eigen3::Eigen cereal )
if(GAUXC_ENABLE_CUTLASS)
  include(gauxc-cutlass)
  target_link_libraries(gauxc_test PUBLIC gauxc_cutlass)
endif()


set( GAUXC_REF_DATA_PATH "${PROJECT_SOURCE_DIR}/tests/ref_data" )
set( GAUXC_ONEDFT_MODEL_PATH "${PROJECT_SOURCE_DIR}/data/onedft_models" )
configure_file( ut_common.hpp.in ${PROJECT_BINARY_DIR}/tests/ut_common.hpp )
target_include_directories( gauxc_test PRIVATE ${PROJECT_BINARY_DIR}/tests )
target_include_directories( gauxc_test PRIVATE ${PROJECT_SOURCE_DIR}/tests )


add_executable( standalone_driver standalone_driver.cxx standards.cxx basis/parse_basis.cxx ini_input.cxx )
target_link_libraries( standalone_driver PUBLIC gauxc gauxc_catch2 Eigen3::Eigen cereal )
target_include_directories( standalone_driver PRIVATE ${PROJECT_BINARY_DIR}/tests )
target_include_directories( standalone_driver PRIVATE ${PROJECT_SOURCE_DIR}/tests )

#add_executable( grid_opt grid_opt.cxx standards.cxx basis/parse_basis.cxx ini_input.cxx )
#target_link_libraries( grid_opt PUBLIC gauxc gauxc_catch2 Eigen3::Eigen cereal )
#target_include_directories( grid_opt PRIVATE ${PROJECT_BINARY_DIR}/tests )
#target_include_directories( grid_opt PRIVATE ${PROJECT_SOURCE_DIR}/tests )

#add_executable( conv_cereal_to_hdf5 conv_cereal_to_hdf5.cxx standards.cxx basis/parse_basis.cxx )
#target_link_libraries( conv_cereal_to_hdf5 PUBLIC gauxc gauxc_catch2 Eigen3::Eigen cereal )
#target_include_directories( conv_cereal_to_hdf5 PRIVATE ${PROJECT_BINARY_DIR}/tests )
#target_include_directories( conv_cereal_to_hdf5 PRIVATE ${PROJECT_SOURCE_DIR}/tests )

add_test( NAME GAUXC_SERIAL_TEST COMMAND $<TARGET_FILE:gauxc_test> )
if( GAUXC_ENABLE_MPI )
  add_test( NAME GAUXC_MPI_TEST
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:gauxc_test> ${MPIEXEC_POSTFLAGS}
  )
endif()
