#include "../examples/c/density_io.h"

#include <assert.h>

int main(void) {
  const char* names[] = {"/DENSITY_SCALAR", "/DENSITY_Z"};
  const hsize_t shapes[][2] = {{2, 2}, {3, 3}, {1, 1}, {1, 4}, {4, 1}};
  const double values[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  size_t count = 0;
  assert(density_matrix_size(2, &count) == 0 && count == 4);
  assert(density_matrix_size(0, &count) != 0);
  assert(density_matrix_size(-1, &count) != 0);
  assert(density_matrix_size(INT64_MAX, &count) != 0);
  hid_t access = H5Pcreate(H5P_FILE_ACCESS);
  assert(access >= 0);
  assert(H5Pset_fapl_core(access, 4096, 0) >= 0);
  hid_t file =
      H5Fcreate("density-io-test.h5", H5F_ACC_TRUNC, H5P_DEFAULT, access);
  assert(file >= 0);
  assert(H5Pclose(access) >= 0);

  for (size_t name = 0; name < 2; ++name) {
    for (size_t shape = 0; shape < 6; ++shape) {
      int rank = shape == 5 ? 1 : 2;
      const hsize_t* dims = shape == 5 ? shapes[4] : shapes[shape];
      hid_t space = H5Screate_simple(rank, dims, NULL);
      assert(space >= 0);
      hid_t dataset = H5Dcreate2(file, names[name], H5T_NATIVE_DOUBLE, space,
                                 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      assert(dataset >= 0);
      assert(H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                      values) >= 0);
      assert(H5Dclose(dataset) >= 0);
      assert(H5Sclose(space) >= 0);
      double guarded[6] = {-99, -99, -99, -99, -99, -99};
      int result = read_density_matrix(file, names[name], 2, guarded + 1);
      assert((result == 0) == (shape == 0));
      assert(guarded[0] == -99 && guarded[5] == -99);
      for (size_t i = 1; i < 5; ++i)
        assert(guarded[i] == (shape == 0 ? values[i - 1] : -99));
      assert(H5Fget_obj_count(file, H5F_OBJ_ALL) == 1);
      assert(H5Ldelete(file, names[name], H5P_DEFAULT) >= 0);
    }
  }
  assert(H5Fclose(file) >= 0);
  return 0;
}
