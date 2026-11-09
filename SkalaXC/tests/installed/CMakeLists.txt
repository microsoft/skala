cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(SkalaXCInstalledConsumers LANGUAGES C CXX)
find_package(SkalaXC REQUIRED CONFIG COMPONENTS C CXX)
enable_testing()

add_executable(installed_consumer_cpp consumer.cpp)
target_link_libraries(installed_consumer_cpp PRIVATE SkalaXC::skalaxc)
target_compile_features(installed_consumer_cpp PRIVATE cxx_std_17)
add_test(NAME installed_consumer_cpp COMMAND installed_consumer_cpp)
set_tests_properties(installed_consumer_cpp PROPERTIES
  ENVIRONMENT "SKALAXC_MODEL_PATH=${SkalaXC_MODEL_DIR}")

add_executable(installed_consumer_c consumer.c)
target_link_libraries(installed_consumer_c PRIVATE SkalaXC::skalaxc)
add_test(NAME installed_consumer_c COMMAND installed_consumer_c)
set_tests_properties(installed_consumer_c PROPERTIES
  ENVIRONMENT "SKALAXC_MODEL_PATH=${SkalaXC_MODEL_DIR}")
