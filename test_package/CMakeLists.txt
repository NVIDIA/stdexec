cmake_minimum_required(VERSION 3.25.0)
project(PackageTest)
enable_testing()

find_package(P2300 REQUIRED)

add_executable(test_stdexec test.cpp)
target_link_libraries(test_stdexec STDEXEC::stdexec $<TARGET_NAME_IF_EXISTS:STDEXEC::system_context>)
if(TARGET STDEXEC::system_context)
  target_compile_definitions(test_stdexec PRIVATE STDEXEC_TEST_SYSTEM_CONTEXT)
endif()
add_test(test_stdexec test_stdexec)
