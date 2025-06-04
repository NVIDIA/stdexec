cmake_minimum_required(VERSION 3.25.0)
project(PackageTest)
enable_testing()

find_package(P2300 REQUIRED)

add_executable(test_stdexec test.cpp)
target_link_libraries(test_stdexec P2300::P2300)
add_test(test_stdexec test_stdexec)
