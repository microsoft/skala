#pragma once

#include <gauxc/runtime_environment.hpp>
#include <skalaxc/skalaxc.hpp>

#include <array>
#include <cstdio>
#include <exception>
#include <limits>
#include <string>
#include <utility>

namespace SkalaXC::mpi {

/**
 * @brief Run rank-local work and collectively report evaluation failures.
 *
 * All ranks in the runtime communicator must participate in every corresponding
 * call, even when their local work is empty. Ranks may arrive at different
 * times, but must not skip or reorder calls. After local work finishes, ranks
 * agree on success before the caller proceeds to numerical reductions. On
 * failure, the lowest failing runtime rank broadcasts its error message
 * (truncated to 2047 bytes), and every rank throws SkalaXC::Exception.
 * Single-rank calls rethrow the original exception instead.
 *
 * @tparam Function Callable accepting no arguments; its return value is
 * ignored.
 * @param runtime Runtime defining the communicator used for error coordination.
 * @param function Rank-local work that must not perform MPI collectives.
 * @throws SkalaXC::Exception If any rank fails in a multi-rank call.
 * @note Only exceptions reaching the calling thread are caught. OpenMP worker
 * exceptions, process failures, and errors before entering this function are
 * not handled.
 */
template <typename Function>
void collective_try(const GauXC::RuntimeEnvironment& runtime,
                    Function&& function) {
  std::exception_ptr exception;
  try {
    std::forward<Function>(function)();
  } catch (...) {
    exception = std::current_exception();
  }

#ifdef GAUXC_HAS_MPI
  if (runtime.comm_size() > 1) {
    int failing_rank =
        exception ? runtime.comm_rank() : std::numeric_limits<int>::max();
    MPI_Allreduce(MPI_IN_PLACE, &failing_rank, 1, MPI_INT, MPI_MIN,
                  runtime.comm());
    if (failing_rank != std::numeric_limits<int>::max()) {
      std::array<char, 2048> message{};
      if (runtime.comm_rank() == failing_rank) {
        try {
          std::rethrow_exception(exception);
        } catch (const std::exception& error) {
          std::snprintf(message.data(), message.size(), "%s", error.what());
        } catch (...) {
          std::snprintf(message.data(), message.size(), "%s", "unknown error");
        }
      }
      MPI_Bcast(message.data(), static_cast<int>(message.size()), MPI_CHAR,
                failing_rank, runtime.comm());
      throw Exception("Runtime rank " + std::to_string(failing_rank) +
                      " evaluation failed: " + message.data());
    }
    return;
  }
#else
  (void)runtime;
#endif
  if (exception) std::rethrow_exception(exception);
}

}  // namespace SkalaXC::mpi