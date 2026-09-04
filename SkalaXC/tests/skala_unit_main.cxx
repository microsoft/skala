#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>
#include <skalaxc/skalaxc_config.hpp>

#ifdef SKALAXC_HAS_CUDA
#include <c10/cuda/CUDACachingAllocator.h>

namespace {

/**
 * @brief Release unused LibTorch CUDA allocations between Catch2 cases.
 *
 * Catch2 v2 executes cases serially within this process. CTest may still run
 * separate test processes in parallel; each process has its own allocator,
 * although their live allocations still share the physical GPU. If in-process
 * parallel test execution is introduced, this process-wide cleanup must be
 * revisited because it can contend with concurrent cases and defeat caching.
 */
class CudaCacheCleanupListener : public Catch::TestEventListenerBase {
 public:
  using Catch::TestEventListenerBase::TestEventListenerBase;

  void testCaseEnded(const Catch::TestCaseStats&) override {
    c10::cuda::CUDACachingAllocator::emptyCache();
  }
};

}  // namespace

CATCH_REGISTER_LISTENER(CudaCacheCleanupListener)
#endif

#ifdef SKALAXC_HAS_MPI
#include <mpi.h>
#endif

int main(int argc, char* argv[]) {
#ifdef SKALAXC_HAS_MPI
  MPI_Init(&argc, &argv);
#endif

  const int rc = Catch::Session().run(argc, argv);

#ifdef SKALAXC_HAS_MPI
  MPI_Finalize();
#endif

  return rc;
}
