#include "atomic_domain_load_balancer.hpp"

#include <catch2/catch.hpp>

#include <gauxc/external/hdf5.hpp>
#include <gauxc/molgrid/defaults.hpp>
#include <gauxc/runtime_environment.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#ifdef GAUXC_HAS_MPI
#include <mpi.h>
#endif

TEST_CASE("atomic domains belong to one runtime rank",
          "[skala][mpi][atomic-domain-ownership][mpi-only]") {
#ifdef GAUXC_HAS_MPI
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm runtime_communicator = MPI_COMM_NULL;
  MPI_Comm_split(MPI_COMM_WORLD, world_rank % 2, world_rank,
                 &runtime_communicator);
  GauXC::RuntimeEnvironment runtime(runtime_communicator);
#else
  GauXC::RuntimeEnvironment runtime;
#endif

  const std::string fixture =
      std::string(SKALAXC_GAUXC_REF_DATA_PATH) + "/h2o2_def2-tzvp.hdf5";
  GauXC::Molecule molecule;
  GauXC::BasisSet<double> basis;
  GauXC::read_hdf5_record(molecule, fixture, "/MOLECULE");
  GauXC::read_hdf5_record(basis, fixture, "/BASIS");
  REQUIRE(molecule.natoms() == 4);
  auto grid = GauXC::MolGridFactory::create_default_molgrid(
      molecule, GauXC::PruningScheme::Unpruned, GauXC::BatchSize(512),
      GauXC::RadialQuad::MuraKnowles, GauXC::AtomicGridSizeDefault::FineGrid);
  auto load_balancer = SkalaXC::detail::make_atomic_domain_load_balancer(
      runtime, molecule, grid, basis, "Default");

  std::array<int, 4> local_owners{};
  for (const auto& task : load_balancer.get_tasks()) {
    if (task.iParent < 0 ||
        static_cast<std::size_t>(task.iParent) >= molecule.natoms()) {
      FAIL("load-balancer task has an invalid parent atom");
      continue;
    }
    local_owners[static_cast<std::size_t>(task.iParent)] = 1;
  }

  std::array<int, 4> owner_counts{};
#ifdef GAUXC_HAS_MPI
  MPI_Allreduce(local_owners.data(), owner_counts.data(),
                static_cast<int>(local_owners.size()), MPI_INT, MPI_SUM,
                runtime_communicator);
#else
  owner_counts = local_owners;
#endif
  for (const auto owners : owner_counts) CHECK(owners == 1);

#ifdef GAUXC_HAS_MPI
  MPI_Comm_free(&runtime_communicator);
#endif
}