#pragma once

#include "index_types.hpp"

#include <skalaxc/skalaxc.hpp>

#include <cstdio>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace SkalaXC::detail {

/**
 * @brief Format one rank-prefixed diagnostics line.
 * @param backend Evaluation backend.
 * @param rank Runtime-communicator rank.
 * @param size Runtime-communicator size.
 * @param phase Diagnostic phase name.
 * @param message Diagnostic text.
 * @return Complete line including its trailing newline.
 */
std::string format_debug_line(ExecutionSpace backend,
                              types::CommunicatorRank rank,
                              types::CommunicatorSize size,
                              std::string_view phase, std::string_view message);

/** @brief Best-effort rank-local diagnostics writer. */
class DebugLogger {
 public:
  /**
   * @brief Configure a diagnostics writer.
   * @param settings Timing and logging settings.
   * @param backend Evaluation backend.
   * @param rank Runtime-communicator rank.
   * @param size Runtime-communicator size.
   * @param output Destination stream, or `nullptr` to discard output.
   */
  DebugLogger(TimingSettings settings, ExecutionSpace backend,
              types::CommunicatorRank rank, types::CommunicatorSize size,
              std::FILE* output = stderr) noexcept;

  /** @return Whether debug logging is enabled. */
  bool enabled() const noexcept { return enabled_; }

  /**
   * @brief Write one already-formatted message.
   * @param phase Diagnostic phase name.
   * @param message Diagnostic text.
   */
  void log(std::string_view phase, std::string_view message) const noexcept;

  /**
   * @brief Build and write one message without propagating formatting errors.
   * @tparam Writer Callable accepting an `std::ostream&`.
   * @param phase Diagnostic phase name.
   * @param write_message Message-writing callable.
   */
  template <typename Writer>
  void log(std::string_view phase, Writer&& write_message) const noexcept {
    if (!enabled_) return;
    try {
      std::ostringstream message;
      std::forward<Writer>(write_message)(message);
      emit(phase, message.str());
    } catch (...) {
      return;
    }
  }

 private:
  void emit(std::string_view phase, std::string_view message) const;

  bool enabled_ = false;
  ExecutionSpace backend_ = ExecutionSpace::Host;
  types::CommunicatorRank rank_{};
  types::CommunicatorSize size_{1};
  std::FILE* output_ = nullptr;
};

}  // namespace SkalaXC::detail
