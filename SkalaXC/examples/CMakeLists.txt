# SkalaXC examples: minimal standalone consumer programs, one per language
# binding. HDF5 / HighFive are used purely to load each example's own density
# input (the consumer's job), never to reach GauXC. Built only when
# SKALAXC_BUILD_EXAMPLES is ON.
#
# Run against the bundled fixtures, e.g.:
#   ./skalaxc_eval_cpp     ../tests/ref_data/skala_he_def2qzvp_pbe_uks.hdf5 PBE
#   ./skalaxc_eval_c       ../tests/ref_data/skala_he_def2qzvp_lda_uks.hdf5 LDA
#   ./skalaxc_eval_fortran ../tests/ref_data/skala_he_def2qzvp_tpss_uks.hdf5 TPSS

if(NOT SKALAXC_HAS_HDF5)
  message(STATUS "SkalaXC examples: HDF5 support disabled; skipping HDF5 examples")
  return()
endif()

# --- C++ example ------------------------------------------------------------
add_executable(skalaxc_eval_cpp cpp/skalaxc_eval.cpp)
target_link_libraries(skalaxc_eval_cpp PRIVATE skalaxc HighFive Eigen3::Eigen)
target_compile_features(skalaxc_eval_cpp PRIVATE cxx_std_17)
if(SKALAXC_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  target_link_libraries(skalaxc_eval_cpp PRIVATE MPI::MPI_CXX)
endif()

# --- C example --------------------------------------------------------------
find_package(HDF5 QUIET COMPONENTS C)
if(HDF5_FOUND AND HDF5_C_LIBRARIES)
  add_executable(skalaxc_eval_c c/skalaxc_eval.c)
  # `m` for sqrt(): unlike C++/Fortran, a C program must link libm explicitly.
  target_link_libraries(skalaxc_eval_c PRIVATE skalaxc ${HDF5_C_LIBRARIES} m)
  target_include_directories(skalaxc_eval_c PRIVATE ${HDF5_INCLUDE_DIRS})
  if(SKALAXC_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS C)
    target_link_libraries(skalaxc_eval_c PRIVATE MPI::MPI_C)
  endif()
else()
  message(STATUS "SkalaXC examples: HDF5 C component not found; skipping C example")
endif()

# --- Fortran example --------------------------------------------------------
if(SKALAXC_BUILD_FORTRAN)
  find_package(HDF5 QUIET COMPONENTS Fortran)
  if(HDF5_FOUND AND HDF5_Fortran_FOUND)
    add_executable(skalaxc_eval_fortran fortran/skalaxc_eval.F90)
    set_source_files_properties(fortran/skalaxc_eval.F90 PROPERTIES
      Fortran_PREPROCESS ON)
    if(SKALAXC_ENABLE_MPI)
      find_package(MPI REQUIRED COMPONENTS Fortran)
      target_compile_definitions(skalaxc_eval_fortran PRIVATE SKALAXC_HAS_MPI)
      target_link_libraries(skalaxc_eval_fortran PRIVATE MPI::MPI_Fortran)
    endif()
    target_link_libraries(skalaxc_eval_fortran PRIVATE
      skalaxc_fortran ${HDF5_Fortran_LIBRARIES})
    target_include_directories(skalaxc_eval_fortran PRIVATE ${HDF5_Fortran_INCLUDE_DIRS})
  else()
    message(STATUS "SkalaXC examples: HDF5 Fortran component not found; skipping Fortran example")
  endif()
endif()
