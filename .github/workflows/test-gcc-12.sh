#!/bin/bash

realpath() {
    [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}
CURDIR=$(realpath $(dirname "$0"))
source ${CURDIR}/_test_common.sh

startTest "gcc-12 compilation"

# Create a temporary directory to store results between runs
BUILDDIR="build/gh-checks/gcc-12/"
mkdir -p "${CURDIR}/../../${BUILDDIR}"

# Run docker with action-cxx-toolkit to check our code
docker run ${DOCKER_RUN_PARAMS} \
    -e INPUT_BUILDDIR="/github/workspace/${BUILDDIR}" \
    -e INPUT_MAKEFLAGS='-j 4' \
    -e INPUT_CC='gcc-12' \
    -e INPUT_CHECKS='build test install' \
    -e INPUT_PREBUILD_COMMAND="\
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add -; \
apt update && apt install -y --no-install-recommends git cmake" \
    lucteo/action-cxx-toolkit.gcc12:latest
status=$?
printStatus $status
