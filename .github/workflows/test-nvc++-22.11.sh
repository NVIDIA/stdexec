#!/bin/bash

realpath() {
    [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}
CURDIR=$(realpath $(dirname "$0"))
source ${CURDIR}/_test_common.sh

startTest "nvc++ compilation"

# Create a temporary directory to store results between runs
BUILDDIR="build/gh-checks/nvc++-22.11/"
mkdir -p "${CURDIR}/../../${BUILDDIR}"

# Run docker with action-cxx-toolkit to check our code
docker run ${DOCKER_RUN_PARAMS} \
    --runtime=nvidia \
    -e NVLOCALRC="/opt/nvidia/localrc" \
    -e INPUT_BUILDDIR="/github/workspace/${BUILDDIR}" \
    -e INPUT_MAKEFLAGS="-j$(nproc --ignore=2)" \
    -e INPUT_IGNORE_CONAN='true' \
    -e INPUT_CC='mpicc' \
    -e INPUT_CHECKS='build test install' \
    -e INPUT_PREBUILD_COMMAND='makelocalrc -d /opt/nvidia -x "$(dirname $(which nvc++))";' \
    ghcr.io/trxcllnt/action-cxx-toolkit:gcc11-cuda11.8-nvhpc22.11-ubuntu22.04
status=$?
printStatus $status
