#pragma once

#include "eigen_types.hpp"
#include "index_types.hpp"
#include "spin_gradient.hpp"

#include <gauxc/gauxc_config.hpp>
#include <gauxc/runtime_environment.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace SkalaXC::mpi {

at::Tensor spin_gradient_to_torch(const SpinGradient& gradient);

SpinGradient torch_to_spin_gradient(const at::Tensor& tensor,
                                    Eigen::Index expected_points);

void broadcast_string(std::string& value, const GauXC::RuntimeEnvironment& rt,
                      int root = 0);

/** @brief Validated MPI counts and displacements for variable collectives. */
class CollectiveLayout {
 public:
  /** @brief Construct an empty layout. */
  CollectiveLayout() = default;

  /**
   * @brief Construct packed displacements from per-rank counts.
   * @param counts Nonnegative element counts in rank order.
   */
  explicit CollectiveLayout(std::vector<int> counts)
      : counts_(std::move(counts)), displacements_(counts_.size()) {
    int displacement = 0;
    for (std::size_t rank = 0; rank < counts_.size(); ++rank) {
      if (counts_[rank] < 0 ||
          counts_[rank] > std::numeric_limits<int>::max() - displacement)
        throw std::invalid_argument("Invalid collective counts");
      displacements_[rank] = displacement;
      displacement += counts_[rank];
    }
  }

  /**
   * @brief Construct from explicit per-rank metadata.
   * @param counts Nonnegative element counts in rank order.
   * @param displacements Nonnegative displacements in rank order.
   */
  CollectiveLayout(std::vector<int> counts, std::vector<int> displacements)
      : counts_(std::move(counts)), displacements_(std::move(displacements)) {
    validate();
  }

  /** @return Per-rank element counts. */
  const std::vector<int>& counts() const noexcept { return counts_; }
  /** @return Per-rank element displacements. */
  const std::vector<int>& displacements() const noexcept {
    return displacements_;
  }

  /** @return Minimum buffer extent covering all rank intervals. */
  int extent() const noexcept {
    int result = 0;
    for (std::size_t rank = 0; rank < counts_.size(); ++rank)
      result = std::max(result, displacements_[rank] + counts_[rank]);
    return result;
  }

  /**
   * @brief Scale every count and displacement.
   * @param factor Positive scale factor.
   * @return Scaled validated layout.
   */
  CollectiveLayout scaled(int factor) const {
    if (factor <= 0) throw std::invalid_argument("Invalid collective scale");
    std::vector<int> counts(counts_.size());
    std::vector<int> displacements(displacements_.size());
    for (std::size_t rank = 0; rank < counts_.size(); ++rank) {
      if (counts_[rank] > std::numeric_limits<int>::max() / factor ||
          displacements_[rank] > std::numeric_limits<int>::max() / factor)
        throw std::invalid_argument("Collective layout overflow");
      counts[rank] = counts_[rank] * factor;
      displacements[rank] = displacements_[rank] * factor;
    }
    return {std::move(counts), std::move(displacements)};
  }

 private:
  void validate() const {
    if (counts_.size() != displacements_.size())
      throw std::invalid_argument("Collective metadata size mismatch");
    for (std::size_t rank = 0; rank < counts_.size(); ++rank) {
      if (counts_[rank] < 0 || displacements_[rank] < 0 ||
          counts_[rank] >
              std::numeric_limits<int>::max() - displacements_[rank])
        throw std::invalid_argument("Invalid collective metadata");
    }
  }

  std::vector<int> counts_;
  std::vector<int> displacements_;
};

at::Tensor gather_torch_gradient(
    const SpinGradient& local_gradient, const CollectiveLayout& point_layout,
    const std::vector<SkalaXC::types::PermutationIndex>&
        rank_to_atom_permutation,
    const GauXC::RuntimeEnvironment& rt, int root = 0);

SpinGradient scatter_torch_gradient(
    const at::Tensor& root_tensor, Eigen::Index local_points,
    const CollectiveLayout& point_layout,
    const std::vector<SkalaXC::types::PermutationIndex>&
        atom_to_rank_permutation,
    const GauXC::RuntimeEnvironment& rt, int root = 0);

#ifdef GAUXC_HAS_MPI

namespace detail {

template <typename T>
struct ContiguousBuffer {
  using value_type = std::remove_const_t<T>;

  T* data;
  std::size_t size;
};

template <typename T>
MPI_Datatype datatype();

template <>
inline MPI_Datatype datatype<double>() {
  return MPI_DOUBLE;
}

template <>
inline MPI_Datatype datatype<int>() {
  return MPI_INT;
}

template <>
inline MPI_Datatype datatype<std::int64_t>() {
  return MPI_INT64_T;
}

template <typename Derived>
void assert_contiguous(const Eigen::DenseBase<Derived>& values) {
  static_assert((Derived::Flags & Eigen::DirectAccessBit) != 0,
                "MPI buffers require direct Eigen storage access");
  if (values.size() == 0) return;
  assert(values.innerStride() == 1);
  assert(values.outerStride() ==
         (Derived::IsRowMajor ? values.cols() : values.rows()));
}

template <typename Derived>
auto contiguous_buffer(const Eigen::DenseBase<Derived>& values) {
  assert_contiguous(values);
  return ContiguousBuffer<const typename Derived::Scalar>{
      values.derived().data(), static_cast<std::size_t>(values.size())};
}

template <typename Derived>
auto contiguous_buffer(Eigen::DenseBase<Derived>& values) {
  assert_contiguous(values);
  return ContiguousBuffer<typename Derived::Scalar>{
      values.derived().data(), static_cast<std::size_t>(values.size())};
}

template <typename T, typename Allocator>
auto contiguous_buffer(const std::vector<T, Allocator>& values) {
  return ContiguousBuffer<const T>{values.data(), values.size()};
}

template <typename T, typename Allocator>
auto contiguous_buffer(std::vector<T, Allocator>& values) {
  return ContiguousBuffer<T>{values.data(), values.size()};
}

inline int mpi_count(std::size_t size) {
  if (size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("MPI buffer exceeds the supported count range");
  return static_cast<int>(size);
}

inline void assert_collective_metadata(
    [[maybe_unused]] const CollectiveLayout& layout,
    const GauXC::RuntimeEnvironment& rt, int root) {
  if (rt.comm_rank() != root) return;
  assert(layout.counts().size() == static_cast<std::size_t>(rt.comm_size()));
}

}  // namespace detail

template <typename Source, typename Destination>
void gather(const Source& source, Destination& root_destination,
            const GauXC::RuntimeEnvironment& rt, int root = 0) {
  const auto source_buffer = detail::contiguous_buffer(source);
  auto destination_buffer = detail::contiguous_buffer(root_destination);
  using value_type = typename decltype(source_buffer)::value_type;
  static_assert(
      std::is_same_v<value_type,
                     typename decltype(destination_buffer)::value_type>);
  static_assert(!std::is_const_v<
                    std::remove_pointer_t<decltype(destination_buffer.data)>>,
                "MPI destination must be mutable");
  if (rt.comm_rank() == root)
    assert(destination_buffer.size == source_buffer.size * rt.comm_size());

  const int count = detail::mpi_count(source_buffer.size);
  MPI_Gather(source_buffer.data, count, detail::datatype<value_type>(),
             destination_buffer.data, count, detail::datatype<value_type>(),
             root, rt.comm());
}

template <typename Source, typename Destination>
void gatherv(const Source& source, Destination& root_destination,
             const CollectiveLayout& layout,
             const GauXC::RuntimeEnvironment& rt, int root = 0) {
  const auto source_buffer = detail::contiguous_buffer(source);
  auto destination_buffer = detail::contiguous_buffer(root_destination);
  using value_type = typename decltype(source_buffer)::value_type;
  static_assert(
      std::is_same_v<value_type,
                     typename decltype(destination_buffer)::value_type>);
  static_assert(!std::is_const_v<
                    std::remove_pointer_t<decltype(destination_buffer.data)>>,
                "MPI destination must be mutable");
  detail::assert_collective_metadata(layout, rt, root);
  if (rt.comm_rank() == root) {
    assert(source_buffer.size ==
           static_cast<std::size_t>(layout.counts()[root]));
    assert(destination_buffer.size >=
           static_cast<std::size_t>(layout.extent()));
  }

  MPI_Gatherv(source_buffer.data, detail::mpi_count(source_buffer.size),
              detail::datatype<value_type>(), destination_buffer.data,
              layout.counts().data(), layout.displacements().data(),
              detail::datatype<value_type>(), root, rt.comm());
}

template <typename Source, typename Destination>
void scatterv(const Source& root_source, const CollectiveLayout& layout,
              Destination& destination, const GauXC::RuntimeEnvironment& rt,
              int root = 0) {
  const auto source_buffer = detail::contiguous_buffer(root_source);
  auto destination_buffer = detail::contiguous_buffer(destination);
  using value_type = typename decltype(source_buffer)::value_type;
  static_assert(
      std::is_same_v<value_type,
                     typename decltype(destination_buffer)::value_type>);
  static_assert(!std::is_const_v<
                    std::remove_pointer_t<decltype(destination_buffer.data)>>,
                "MPI destination must be mutable");
  detail::assert_collective_metadata(layout, rt, root);
  if (rt.comm_rank() == root) {
    assert(destination_buffer.size ==
           static_cast<std::size_t>(layout.counts()[root]));
    assert(source_buffer.size >= static_cast<std::size_t>(layout.extent()));
  }

  MPI_Scatterv(source_buffer.data, layout.counts().data(),
               layout.displacements().data(), detail::datatype<value_type>(),
               destination_buffer.data,
               detail::mpi_count(destination_buffer.size),
               detail::datatype<value_type>(), root, rt.comm());
}

template <typename Values>
void allreduce_sum(Values& values, const GauXC::RuntimeEnvironment& rt) {
  auto buffer = detail::contiguous_buffer(values);
  using value_type = typename decltype(buffer)::value_type;
  static_assert(!std::is_const_v<std::remove_pointer_t<decltype(buffer.data)>>,
                "MPI reduction buffer must be mutable");
  if (buffer.size == 0) return;
  MPI_Allreduce(MPI_IN_PLACE, buffer.data, detail::mpi_count(buffer.size),
                detail::datatype<value_type>(), MPI_SUM, rt.comm());
}

#endif

}  // namespace SkalaXC::mpi
