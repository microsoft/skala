cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(SkalaXCInstalledConsumers LANGUAGES CXX)
find_package(SkalaXC REQUIRED CONFIG COMPONENTS CXX)
enable_testing()

add_executable(installed_consumer_cpp consumer.cpp)
target_link_libraries(installed_consumer_cpp PRIVATE SkalaXC::skalaxc)
target_compile_features(installed_consumer_cpp PRIVATE cxx_std_17)
add_test(NAME installed_consumer_cpp COMMAND installed_consumer_cpp)
set_tests_properties(installed_consumer_cpp PROPERTIES
  ENVIRONMENT "SKALAXC_MODEL_PATH=${SkalaXC_MODEL_DIR}")
