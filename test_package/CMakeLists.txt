cmake_minimum_required( VERSION 3.17 )
project(PackageTest CXX)

# Integrate with LLVM/clang tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(p2300 REQUIRED)
find_package(Threads REQUIRED)

add_executable(test_p2300 test.cpp)
target_link_libraries(test_p2300 PUBLIC P2300::p2300)
