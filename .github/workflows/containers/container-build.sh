#!/bin/sh

# Run this as:
# ./container-build.sh <image> <build_type>
# 
# where <image> can be `gcc11`, and <build_type> can be `debug`, `release`, `tsan`, `asan`.

set -ex

IMG=$1
BUILD_TYPE=$2

_realpath() {
    [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}
_CURDIR=$(_realpath $(dirname "$0"))
ROOTDIR="${_CURDIR}/../../../"


# Ensure the image is built
docker build -t ${IMG} -f ${_CURDIR}/Dockerfile.${IMG} .

# Run the container, building the code in the desired configuration.
mkdir -p "${ROOTDIR}/build/${IMG}-${BUILD_TYPE}/"
docker run --rm -it \
    -v ${ROOTDIR}:/github/workspace \
    ${IMG} \
    "build/${IMG}-${BUILD_TYPE}" "${BUILD_TYPE}"
