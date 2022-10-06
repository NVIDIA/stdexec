cmake_minimum_required( VERSION 3.17 )
project(PackageTest CXX)

find_package(Threads REQUIRED)
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(test_stdexec test.cpp)
target_link_libraries(test_stdexec ${CONAN_LIBS} Threads::Threads)
