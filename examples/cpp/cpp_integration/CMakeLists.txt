cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(
  "skala_cpp_integration"
  VERSION "0.0.1"
  LANGUAGES CXX
)

find_package(nlohmann_json REQUIRED)
find_package(Torch REQUIRED)

add_executable(
  ${PROJECT_NAME}
  ./main.cpp
)
target_link_libraries(
  ${PROJECT_NAME}
  "${TORCH_LIBRARIES}"
)
set_property(
  TARGET ${PROJECT_NAME}
  PROPERTY CXX_STANDARD 17
)