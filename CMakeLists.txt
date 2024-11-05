cmake_minimum_required(VERSION 3.15.0)
project(
  cs-228-project
  VERSION 0.1
  DESCRIPTION ""
  HOMEPAGE_URL ""
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(Torch_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtorch/share/cmake/Torch)

find_package(Torch REQUIRED)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/parlaylib/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann-json/include
)

add_compile_options(
  # -O3
  -Wall -Wextra
)

add_executable(temp temp.cpp)
target_link_libraries(temp "${TORCH_LIBRARIES}")
