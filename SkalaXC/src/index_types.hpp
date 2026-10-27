#pragma once

#include <skalaxc/skalaxc.hpp>

#include <cstddef>
#include <cstdint>

namespace SkalaXC::types {

using AtomIndex = NamedType<std::size_t, struct AtomIndexTag>;
using TaskIndex = NamedType<std::size_t, struct TaskIndexTag>;
using GridPointOffset = NamedType<std::int64_t, struct GridPointOffsetTag>;
using GridPointCount =
    AdditiveNamedType<std::int64_t, struct GridPointCountTag>;
using BasisFunctionCount =
    AdditiveNamedType<std::int64_t, struct BasisFunctionCountTag>;
using AtomCount = AdditiveNamedType<std::uint64_t, struct AtomCountTag>;
using TaskCount = AdditiveNamedType<std::uint64_t, struct TaskCountTag>;
using ModelBatchCount =
    AdditiveNamedType<std::uint64_t, struct ModelBatchCountTag>;
using DomainCount = AdditiveNamedType<std::uint64_t, struct DomainCountTag>;
using CommunicatorRank = NamedType<int, struct CommunicatorRankTag>;
using CommunicatorSize = NamedType<int, struct CommunicatorSizeTag>;
using DeviceId = NamedType<int, struct DeviceIdTag>;
using OpenMPThreadCount = NamedType<int, struct OpenMPThreadCountTag>;
using PermutationIndex = NamedType<std::int64_t, struct PermutationIndexTag>;
using WorkEstimate = NamedType<std::uint64_t, struct WorkEstimateTag>;

constexpr GridPointOffset operator+(GridPointOffset offset,
                                    GridPointCount count) {
  return GridPointOffset{offset.raw() + count.raw()};
}

constexpr GridPointOffset& operator+=(GridPointOffset& offset,
                                      GridPointCount count) {
  offset = offset + count;
  return offset;
}

template <typename Count>
/** @brief Inclusive observed range for a strongly typed count. @tparam Count
   Count type. */
struct CountRange {
  Count minimum{};  ///< Smallest observed count.
  Count maximum{};  ///< Largest observed count.
};

}  // namespace SkalaXC::types
