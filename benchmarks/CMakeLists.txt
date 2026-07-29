# Standalone benchmark project.
#
# Deliberately not wired into the top-level build: it needs only Eigen and the vendored GeoFIK, so
# it configures and runs without libfranka, ruckig or a robot.
#
#   cmake -S benchmarks -B build/benchmarks -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/benchmarks
#   ./build/benchmarks/bench_manipulability

cmake_minimum_required(VERSION 3.11)
project(franky-benchmarks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

find_package(Eigen3 3.4.0 REQUIRED NO_MODULE)

add_executable(bench_manipulability
        bench_manipulability.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/geofik/geofik.cpp)
target_include_directories(bench_manipulability PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/geofik)
target_link_libraries(bench_manipulability PRIVATE Eigen3::Eigen)
target_compile_options(bench_manipulability PRIVATE -O2 -DNDEBUG)
