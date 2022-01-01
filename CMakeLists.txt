cmake_minimum_required(VERSION 3.17.0)

# Define the project
project(P2300-dev LANGUAGES CXX)

if (EXISTS ${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    conan_basic_setup(TARGETS)
else ()
    message(ERROR "The file conanbuildinfo.cmake doesn't exist, you have to run conan install first")
endif ()

# Set the version of the project
project(P2300-dev VERSION "${CONAN_PACKAGE_VERSION}")

# Print CMake configuration
message(STATUS "System           : ${CMAKE_SYSTEM}")
message(STATUS "System name      : ${CMAKE_SYSTEM_NAME}")
message(STATUS "System ver       : ${CMAKE_SYSTEM_VERSION}")
message(STATUS)

# The main library
add_library(P2300 INTERFACE)

# Set the version and current build date
set(P2300_VERSION "${PROJECT_VERSION}")
set(P2300_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
string(TIMESTAMP P2300_BUILD_DATE "%Y-%m-%d")
string(TIMESTAMP P2300_BUILD_YEAR "%Y")

message(STATUS "Library ver      : ${P2300_VERSION}")
message(STATUS "Build date       : ${P2300_BUILD_DATE}")
message(STATUS "Build year       : ${P2300_BUILD_YEAR}")
message(STATUS)

# Set library version
set_target_properties(P2300 PROPERTIES
                      VERSION "${P2300_VERSION}"
                      SOVERSION "${P2300_VERSION_MAJOR}")

# Declare the public include directories
target_include_directories(P2300 INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                           $<INSTALL_INTERFACE:include>
                           )

# Ensure that we link with the threading library
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
find_package(Threads REQUIRED)
target_link_libraries(P2300 INTERFACE Threads::Threads)

# Use C++20 standard
target_compile_features(P2300 INTERFACE cxx_std_20)

# Prevent linking errors with CXX11 or older ABI (visible when linking with rapidcheck)
target_compile_definitions(P2300 INTERFACE _GLIBCXX_USE_CXX11_ABI=1)

add_library(P2300::p2300 ALIAS P2300)

# Now, set up test executable
enable_testing()

set(test_sourceFiles
    "test/test_main.cpp"
    "test/test_include.cpp"
    )

add_executable(test.P2300 ${test_sourceFiles})

target_link_libraries(test.P2300 P2300)
target_include_directories(test.P2300 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test.P2300 CONAN_PKG::catch2)

# Discover the Catch2 test built by the application
include(CTest)
include(Catch)
catch_discover_tests(test.P2300)

# Turn all warnings
target_compile_options(test.P2300 PUBLIC
                       $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
                       -Wall>
                       $<$<CXX_COMPILER_ID:MSVC>:
                       /W4>)


# Set up examples
function(def_example target sourceFile)
    add_executable(${target} ${sourceFile})
    target_link_libraries(${target} P2300)
    target_include_directories(${target} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
endfunction()

def_example(example.hello_world "examples/hello_world.cpp")
def_example(example.hello_coro "examples/hello_coro.cpp")
def_example(example.retry "examples/retry.cpp")
def_example(example.server_theme.let_value "examples/server_theme/let_value.cpp")
def_example(example.server_theme.on_transfer "examples/server_theme/on_transfer.cpp")
