#pragma once

#include <skalaxc/skalaxc.hpp>

#include <sstream>
#include <string>

namespace SkalaXC::detail {

[[noreturn]] inline void throw_exception(const char* file, const char* function,
                                         int line, std::string message) {
  std::ostringstream stream;
  stream << "SkalaXC Exception (" << message << ")\n"
         << "  File     " << file << '\n'
         << "  Function " << function << '\n'
         << "  Line     " << line << '\n';
  throw Exception(stream.str());
}

}  // namespace SkalaXC::detail

#define SKALAXC_EXCEPTION(message)                                            \
  ::SkalaXC::detail::throw_exception(__FILE__, __PRETTY_FUNCTION__, __LINE__, \
                                     (message))
