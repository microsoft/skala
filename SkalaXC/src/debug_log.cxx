#include "debug_log.hpp"

#include <cstdio>
#include <sstream>

namespace SkalaXC::detail {

std::string format_debug_line(ExecutionSpace backend,
                              types::CommunicatorRank rank,
                              types::CommunicatorSize size,
                              std::string_view phase,
                              std::string_view message) {
  std::ostringstream line;
  line << "[SkalaXC][debug][backend="
       << (backend == ExecutionSpace::Device ? "device" : "host")
       << "][rank=" << rank.raw() << '/' << size.raw() << "][phase=" << phase
       << "] ";
  for (const char character : message)
    line << (character == '\n' || character == '\r' ? ' ' : character);
  line << '\n';
  return line.str();
}

DebugLogger::DebugLogger(TimingSettings settings, ExecutionSpace backend,
                         types::CommunicatorRank rank,
                         types::CommunicatorSize size,
                         std::FILE* output) noexcept
    : enabled_(settings.debug_logging),
      backend_(backend),
      rank_(rank),
      size_(size),
      output_(output) {}

void DebugLogger::log(std::string_view phase,
                      std::string_view message) const noexcept {
  if (!enabled_ || !output_) return;
  try {
    emit(phase, message);
  } catch (...) {
    return;
  }
}

void DebugLogger::emit(std::string_view phase, std::string_view message) const {
  const auto line = format_debug_line(backend_, rank_, size_, phase, message);
  std::fwrite(line.data(), 1, line.size(), output_);
  std::fflush(output_);
}

}  // namespace SkalaXC::detail