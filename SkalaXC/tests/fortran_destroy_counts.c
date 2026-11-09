#include <stddef.h>

static int counts[8];

#define TRACK_DESTROY(name, index)                     \
  void __real_skalaxc_##name##_destroy(void *handle);  \
  void __wrap_skalaxc_##name##_destroy(void *handle) { \
    if (handle != NULL) ++counts[index];               \
    __real_skalaxc_##name##_destroy(handle);           \
  }

TRACK_DESTROY(runtime_environment, 0)
TRACK_DESTROY(molecule, 1)
TRACK_DESTROY(basisset, 2)
TRACK_DESTROY(molgrid, 3)
TRACK_DESTROY(load_balancer, 4)
TRACK_DESTROY(molecular_weights, 5)
TRACK_DESTROY(functional, 6)
TRACK_DESTROY(xc_integrator, 7)

int skalaxc_test_destroy_counts_equal(int expected) {
  for (int i = 0; i < 8; ++i)
    if (counts[i] != expected) return 0;
  return 1;
}
