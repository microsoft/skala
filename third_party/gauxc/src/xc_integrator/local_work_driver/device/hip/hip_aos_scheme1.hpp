/**
 * GauXC Copyright (c) 2020-2024, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy).
 *
 * (c) 2024-2025, Microsoft Corporation
 *
 * All rights reserved.
 *
 * See LICENSE.txt for details
 */
#pragma once
#include "device/scheme1_base.hpp"
#include "device/scheme1_magma_base.hpp"

namespace GauXC {

namespace alg_constants {

struct HipAoSScheme1 {
  static constexpr uint32_t weight_unroll = 4;
  static constexpr uint32_t weight_thread_block = 640;
  static constexpr uint32_t weight_thread_block_per_sm = 2;
  static constexpr uint32_t max_submat_blocks = 10;
};

}

template <typename Base = AoSScheme1Base>
struct HipAoSScheme1 : public Base {

  // API Overrides
  void partition_weights( XCDeviceData* ) override final;

  std::unique_ptr<XCDeviceData> create_device_data(const DeviceRuntimeEnvironment&) override final;

  struct Data;

};

extern template struct HipAoSScheme1<AoSScheme1Base>;
#ifdef GAUXC_HAS_MAGMA
extern template struct HipAoSScheme1<AoSScheme1MAGMABase>;
#endif


template <typename Base>
struct HipAoSScheme1<Base>::Data : public Base::Data {

  virtual ~Data() noexcept;
  Data() = delete;
  Data(const DeviceRuntimeEnvironment& rt);

  // Final overrides
  size_t get_submat_chunk_size(int32_t,int32_t) override final;
  size_t get_ldatoms() override final;
  size_t get_rab_align() override final;
  int get_points_per_subtask() override final;

};

extern template struct HipAoSScheme1<AoSScheme1Base>::Data;
#ifdef GAUXC_HAS_MAGMA
extern template struct HipAoSScheme1<AoSScheme1MAGMABase>::Data;
#endif


}
