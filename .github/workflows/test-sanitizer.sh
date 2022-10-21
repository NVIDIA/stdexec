#!/bin/bash

realpath() {
    [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}
CURDIR=$(realpath $(dirname "$0"))
source ${CURDIR}/_test_common.sh

startTest "Run the tests with sanitizer"

# Create a temporary directory to store results between runs
BUILDDIR="build/gh-checks/sanitizer/"
mkdir -p "${CURDIR}/../../${BUILDDIR}"

# Run docker with action-cxx-toolkit to check our code
docker run ${DOCKER_RUN_PARAMS} \
    -e INPUT_BUILDDIR="/github/workspace/${BUILDDIR}" \
    -e INPUT_CC='clang-13' \
    -e INPUT_MAKEFLAGS='-j 4' \
    -e INPUT_CHECKS='sanitize=address sanitize=undefined' \
    ghcr.io/trxcllnt/action-cxx-toolkit:main
status=$?
printStatus $status
