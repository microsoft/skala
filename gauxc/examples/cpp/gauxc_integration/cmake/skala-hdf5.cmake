find_package(HDF5 REQUIRED COMPONENTS C HL)

# Some released GauXC/HighFive exports refer to a plain "hdf5-shared" link
# item. Map it to the resolved imported HDF5 target so link lines remain
# portable across package layouts.
if(TARGET HDF5::HDF5 AND NOT TARGET hdf5-shared)
  add_library(hdf5-shared INTERFACE IMPORTED)
  target_link_libraries(hdf5-shared INTERFACE HDF5::HDF5)
endif()
