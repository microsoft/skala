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
#include "ut_common.hpp"
#include <gauxc/runtime_environment.hpp>
#include <gauxc/exceptions.hpp>

using namespace GauXC;

template <typename RuntimeType, typename... Args>
void test_basic_check(Args&&... args) {
      RuntimeType rt(std::forward<Args>(args)...);
      REQUIRE( rt.shared_usage_count() == 1 );

      SECTION("MPI Data") {
        #ifdef GAUXC_HAS_MPI
        REQUIRE( rt.comm() == MPI_COMM_WORLD );
        int world_rank, world_size;
        MPI_Comm_rank( MPI_COMM_WORLD, &world_rank );
        MPI_Comm_size( MPI_COMM_WORLD, &world_size );
        REQUIRE( rt.comm_rank() == world_rank );
        REQUIRE( rt.comm_size() == world_size );
        #else
        REQUIRE( rt.comm_rank() == 0 );
        REQUIRE( rt.comm_size() == 1 );
        #endif
      }

      SECTION("Copy") {
          RuntimeType cpy(rt);
          GAUXC_MPI_CODE( REQUIRE( cpy.comm() == rt.comm() ); )
          REQUIRE( cpy.comm_rank() == rt.comm_rank() );
          REQUIRE( cpy.comm_size() == rt.comm_size() );
          REQUIRE( cpy.shared_usage_count() == 2 );
      }

      SECTION("Move") {
          GAUXC_MPI_CODE(auto c = rt.comm();)
          auto r = rt.comm_rank();
          auto s = rt.comm_size();
          RuntimeType cpy(std::move(rt));
          GAUXC_MPI_CODE( REQUIRE( cpy.comm() == c ); )
          REQUIRE( cpy.comm_rank() == r );
          REQUIRE( cpy.comm_size() == s );
          REQUIRE( cpy.shared_usage_count() == 1 );
      }

}

TEST_CASE("Runtime", "[runtime]") {
    SECTION("Host") {
       test_basic_check<RuntimeEnvironment>(GAUXC_MPI_CODE(MPI_COMM_WORLD)); 
    }

    #ifdef GAUXC_HAS_DEVICE
    SECTION("Device") {

      SECTION("Memory Wrapper") {
        void* p   = (void*)0x6666DEADBEEF6666;
        size_t sz = 40;
        auto rt = DeviceRuntimeEnvironment(GAUXC_MPI_CODE(MPI_COMM_WORLD,) p, sz);
        REQUIRE_FALSE( rt.owns_memory() );
        REQUIRE( rt.device_memory() == p );
        REQUIRE( rt.device_memory_size() == sz );
      }

      SECTION("Owns Memory") {
        auto rt = DeviceRuntimeEnvironment( GAUXC_MPI_CODE(MPI_COMM_WORLD,) 0.2 );

        auto p = rt.device_memory();
        auto sz = rt.device_memory_size();

        REQUIRE( p != nullptr );
        REQUIRE( sz > 0 );
        REQUIRE( rt.owns_memory() );
        REQUIRE( rt.shared_usage_count() == 1 );

        SECTION("Copy") {
          DeviceRuntimeEnvironment cpy(rt);
          REQUIRE( cpy.device_memory() == p );
          REQUIRE( cpy.device_memory_size() == sz );
          REQUIRE( cpy.owns_memory() );
          REQUIRE( cpy.shared_usage_count() == 2 );
          // Sanity check
          REQUIRE( rt.device_memory() == p );
          REQUIRE( rt.device_memory_size() == sz );
          REQUIRE( rt.owns_memory() );
          REQUIRE( rt.shared_usage_count() == 2 );
        }

        SECTION("Move") {
          DeviceRuntimeEnvironment cpy(std::move(rt));
          REQUIRE( cpy.device_memory() == p );
          REQUIRE( cpy.device_memory_size() == sz );
          REQUIRE( cpy.owns_memory() );
          REQUIRE( cpy.shared_usage_count() == 1 );
          // Sanity check
          REQUIRE_THROWS_AS( rt.device_memory(), generic_gauxc_exception );
          REQUIRE_THROWS_AS( rt.device_memory_size(), generic_gauxc_exception );
          REQUIRE_THROWS_AS( rt.owns_memory(), generic_gauxc_exception );
        }
      }

      SECTION("Host != Device") {
        auto rt = RuntimeEnvironment(GAUXC_MPI_CODE(MPI_COMM_WORLD));
        REQUIRE_THROWS_AS(detail::as_device_runtime(rt), generic_gauxc_exception);
      }

      SECTION("as_device_runtime") {
        auto d_rt = DeviceRuntimeEnvironment(GAUXC_MPI_CODE(MPI_COMM_WORLD,) 0.2);
        auto p  = d_rt.device_memory();
        auto sz = d_rt.device_memory_size();
        REQUIRE(d_rt.owns_memory());


        SECTION("Dynamic Cast") {
          auto d_rt_new = detail::as_device_runtime(d_rt);
          REQUIRE(d_rt_new.device_memory() == p);
          REQUIRE(d_rt_new.device_memory_size() == sz);
          REQUIRE(d_rt_new.owns_memory());
          REQUIRE(d_rt_new.shared_usage_count() == 2);
        }

        SECTION("Host Copy") {
          RuntimeEnvironment h_rt(d_rt);
          REQUIRE(h_rt.shared_usage_count() == 2);
          auto d_rt_new = detail::as_device_runtime(h_rt);
          REQUIRE(d_rt_new.device_memory() == p);
          REQUIRE(d_rt_new.device_memory_size() == sz);
          REQUIRE(d_rt_new.owns_memory());
          REQUIRE(d_rt_new.shared_usage_count() == 3);
        }
      }


      SECTION("Basic Checks") {
        test_basic_check<DeviceRuntimeEnvironment>(GAUXC_MPI_CODE(MPI_COMM_WORLD,)0.2);
      }
    }
    #endif
}
