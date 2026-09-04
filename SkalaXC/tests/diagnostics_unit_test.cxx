#include "debug_log.hpp"
#include "diagnostics.hpp"

#include <catch2/catch.hpp>

#include <chrono>
#include <cstdio>
#include <string>

TEST_CASE("debug lines contain backend rank and phase", "[diagnostics]") {
  const auto line = SkalaXC::detail::format_debug_line(
      SkalaXC::ExecutionSpace::Device, SkalaXC::types::CommunicatorRank{2},
      SkalaXC::types::CommunicatorSize{4}, "setup", "tasks=7");
  CHECK(line ==
        "[SkalaXC][debug][backend=device][rank=2/4][phase=setup] "
        "tasks=7\n");
}

TEST_CASE("debug lines replace embedded line breaks", "[diagnostics]") {
  const auto line = SkalaXC::detail::format_debug_line(
      SkalaXC::ExecutionSpace::Host, SkalaXC::types::CommunicatorRank{0},
      SkalaXC::types::CommunicatorSize{1}, "model", "selector=a\nb\r");
  CHECK(line ==
        "[SkalaXC][debug][backend=host][rank=0/1][phase=model] "
        "selector=a b \n");
}

TEST_CASE("disabled debug logger does not invoke message formatting",
          "[diagnostics]") {
  SkalaXC::detail::DebugLogger logger(
      SkalaXC::TimingSettings{}, SkalaXC::ExecutionSpace::Host,
      SkalaXC::types::CommunicatorRank{0}, SkalaXC::types::CommunicatorSize{1});
  bool invoked = false;
  logger.log("setup", [&](std::ostream&) { invoked = true; });
  CHECK_FALSE(invoked);
}

TEST_CASE("enabled debug logger emits one complete line", "[diagnostics]") {
  std::FILE* output = std::tmpfile();
  REQUIRE(output != nullptr);
  SkalaXC::TimingSettings settings;
  settings.debug_logging = true;
  SkalaXC::detail::DebugLogger logger(settings, SkalaXC::ExecutionSpace::Host,
                                      SkalaXC::types::CommunicatorRank{1},
                                      SkalaXC::types::CommunicatorSize{3},
                                      output);
  logger.log("setup", [](std::ostream& message) { message << "tasks=5"; });
  std::rewind(output);
  char buffer[256]{};
  const auto bytes = std::fread(buffer, 1, sizeof(buffer) - 1, output);
  std::fclose(output);
  const std::string line(buffer, bytes);
  CHECK(line ==
        "[SkalaXC][debug][backend=host][rank=1/3][phase=setup] tasks=5\n");
}

TEST_CASE("default diagnostics record host metrics", "[diagnostics]") {
  SkalaXC::detail::DiagnosticsRegistry registry(
      SkalaXC::TimingSettings{}, SkalaXC::ExecutionSpace::Host,
      SkalaXC::types::CommunicatorRank{3});

  registry.record(SkalaXC::TimingMetric::ModelForward,
                  std::chrono::nanoseconds(17));
  registry.increment_exc_vxc_calls();
  registry.set_local_workload(
      {SkalaXC::types::TaskCount{5}, SkalaXC::types::GridPointCount{100}});
  registry.record_model_batch(SkalaXC::types::DomainCount{2});

  const auto snapshot = registry.snapshot();
  CHECK(snapshot.rank == 3);
  CHECK(snapshot.timing(SkalaXC::TimingMetric::ModelForward).status ==
        SkalaXC::TimingStatus::Complete);
  CHECK(snapshot.exc_vxc_calls == 1);
  CHECK(snapshot.tasks == 5);
  CHECK(snapshot.points == 100);
  CHECK(snapshot.model_batches == 1);
}

TEST_CASE("host timing scope can finish before destruction", "[diagnostics]") {
  SkalaXC::detail::DiagnosticsRegistry registry(
      SkalaXC::TimingSettings{}, SkalaXC::ExecutionSpace::Host,
      SkalaXC::types::CommunicatorRank{0});
  {
    SkalaXC::detail::HostTimingScope timer(registry,
                                           SkalaXC::TimingMetric::TotalEXCVXC);
    timer.finish();
    CHECK(registry.snapshot()
              .timing(SkalaXC::TimingMetric::TotalEXCVXC)
              .call_count == 1);
  }
  CHECK(registry.snapshot()
            .timing(SkalaXC::TimingMetric::TotalEXCVXC)
            .call_count == 1);
}

TEST_CASE("host diagnostics accumulate and reset evaluation metrics",
          "[diagnostics]") {
  SkalaXC::TimingSettings settings;
  SkalaXC::detail::DiagnosticsRegistry registry(
      settings, SkalaXC::ExecutionSpace::Host,
      SkalaXC::types::CommunicatorRank{1});

  registry.record(SkalaXC::TimingMetric::ModelLoad,
                  std::chrono::nanoseconds(11));
  registry.record(SkalaXC::TimingMetric::ModelForward,
                  std::chrono::nanoseconds(13));
  registry.record(SkalaXC::TimingMetric::ModelForward,
                  std::chrono::nanoseconds(17));
  registry.increment_exc_vxc_calls();
  registry.set_parallel_setup({SkalaXC::types::CommunicatorSize{4},
                               SkalaXC::types::DeviceId{2},
                               SkalaXC::types::OpenMPThreadCount{8}, 0.75,
                               SkalaXC::DomainBatchMode::Aggressive});
  registry.set_local_workload(
      {SkalaXC::types::TaskCount{7}, SkalaXC::types::GridPointCount{101}});
  registry.set_model_workload(
      {SkalaXC::types::AtomCount{3},
       SkalaXC::types::ModelBatchCount{2},
       {SkalaXC::types::GridPointCount{5}, SkalaXC::types::GridPointCount{23}},
       {SkalaXC::types::BasisFunctionCount{2},
        SkalaXC::types::BasisFunctionCount{11}},
       {SkalaXC::types::GridPointCount{17}, SkalaXC::types::GridPointCount{51}},
       SkalaXC::types::DomainCount{3}});
  registry.record_model_batch(SkalaXC::types::DomainCount{4});

  auto snapshot = registry.snapshot();
  const auto& model_load = snapshot.timing(SkalaXC::TimingMetric::ModelLoad);
  const auto& model_forward =
      snapshot.timing(SkalaXC::TimingMetric::ModelForward);
  CHECK(model_load.total_nanoseconds == 11);
  CHECK(model_load.call_count == 1);
  CHECK(model_forward.last_nanoseconds == 17);
  CHECK(model_forward.total_nanoseconds == 30);
  CHECK(model_forward.call_count == 2);
  CHECK(snapshot.exc_vxc_calls == 1);
  CHECK(snapshot.tasks == 7);
  CHECK(snapshot.points == 101);
  CHECK(snapshot.model_batches == 1);
  CHECK(snapshot.domains == 4);

  registry.reset_evaluation();
  snapshot = registry.snapshot();
  CHECK(snapshot.timing(SkalaXC::TimingMetric::ModelLoad).total_nanoseconds ==
        11);
  CHECK(snapshot.timing(SkalaXC::TimingMetric::ModelForward).status ==
        SkalaXC::TimingStatus::Unavailable);
  CHECK(snapshot.exc_vxc_calls == 0);
  CHECK(snapshot.tasks == 7);
  CHECK(snapshot.points == 101);
  CHECK(snapshot.model_batches == 0);
  CHECK(snapshot.communicator_size == 4);
  CHECK(snapshot.device_id == 2);
  CHECK(snapshot.configured_model_batches == 2);
  CHECK(snapshot.openmp_threads == 8);
  CHECK(snapshot.domain_batch_mode == SkalaXC::DomainBatchMode::Aggressive);
  CHECK(snapshot.local_atoms == 3);
  CHECK(snapshot.model_batch_points_max == 51);
}