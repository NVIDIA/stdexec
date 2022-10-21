#!/bin/bash

realpath() {
    [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}
CURDIR=$(realpath $(dirname "$0"))
source ${CURDIR}/_test_common.sh

startTest "clang-format test"

# Run docker with action-cxx-toolkit to check our code
docker run ${DOCKER_RUN_PARAMS} \
    -e INPUT_CHECKS='clang-format' \
    -e INPUT_CLANGFORMATDIRS='include test examples' \
    ghcr.io/trxcllnt/action-cxx-toolkit:main
status=$?
printStatus $status
