#ifndef SKALAXC_EXAMPLE_DENSITY_IO_H
#define SKALAXC_EXAMPLE_DENSITY_IO_H

#include <hdf5.h>
#include <stdint.h>
#include <stdio.h>

static int density_matrix_size(int64_t nbf, size_t* count) {
  if (nbf <= 0 || (uint64_t)nbf > SIZE_MAX ||
      (size_t)nbf > SIZE_MAX / (size_t)nbf / sizeof(double)) {
    fprintf(stderr, "Invalid or oversized density matrix dimension\n");
    return -1;
  }
  *count = (size_t)nbf * (size_t)nbf;
  return 0;
}

static int read_density_matrix(hid_t file, const char* name, size_t nbf,
                               double* buffer) {
  hid_t dataset = -1, file_space = -1, memory_space = -1;
  hsize_t dimensions[2];
  int result = -1;
  dataset = H5Dopen2(file, name, H5P_DEFAULT);
  if (dataset < 0) goto cleanup;
  file_space = H5Dget_space(dataset);
  if (file_space < 0) goto cleanup;
  if (H5Sget_simple_extent_ndims(file_space) != 2 ||
      H5Sget_simple_extent_dims(file_space, dimensions, NULL) < 0 ||
      dimensions[0] != nbf || dimensions[1] != nbf) {
    fprintf(stderr, "%s: expected density shape (%zu, %zu)\n", name, nbf, nbf);
    goto cleanup;
  }
  memory_space = H5Screate_simple(2, dimensions, NULL);
  if (memory_space < 0) goto cleanup;
  if (H5Dread(dataset, H5T_NATIVE_DOUBLE, memory_space, file_space, H5P_DEFAULT,
              buffer) < 0)
    goto cleanup;
  result = 0;

cleanup:
  if (memory_space >= 0) H5Sclose(memory_space);
  if (file_space >= 0) H5Sclose(file_space);
  if (dataset >= 0) H5Dclose(dataset);
  if (result != 0) fprintf(stderr, "HDF5 density read failed: %s\n", name);
  return result;
}

#endif
