#!/bin/bash

# This is called inside the container to build the code in the desired configuration.

set -ex

BUILDDIR=$1

function do_build {
  MODE=$1
  CXXFLAGS=$2

  # Configure
  cmake -S . -B ${BUILDDIR} \
    -DCMAKE_BUILD_TYPE=${MODE} \
    -DCMAKE_CXX_FLAGS="${CXXFLAGS}" \
    -DSTDEXEC_ENABLE_TBB:BOOL=false \
    ;

  # Compile
  cmake --build ${BUILDDIR} -v;

  # Tests
  ctest --test-dir ${BUILDDIR} --verbose --output-on-failure --timeout 60;
}

case $2 in
    "debug" )
        do_build "Debug" "";;
    "release" )
        do_build "Release" "";;
    "tsan" )
        do_build "Release" "-fsanitize=thread";;
    "asan" )
        do_build "Release" "-fsanitize=address";;
esac
