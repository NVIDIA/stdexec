#!/bin/bash

_realpath() {
    [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}
_CURDIR=$(_realpath $(dirname "$0"))
ROOTDIR="${_CURDIR}/../../"

# Export the general params that need to be added do 'docker run'
DOCKER_RUN_PARAMS="--rm -it --workdir /github/workspace -v ${ROOTDIR}:/github/workspace"

RED="\033[0;31m"
GREEN="\033[0;32m"
ORANGE="\033[0;33m"
BLUE="\033[0;34m"
PURPLE="\033[0;35m"
CYAN="\033[0;36m"
LIGHTGRAY="\033[0;37m"
DARKGRAY="\033[1;30m"
LIGHTRED="\033[1;31m"
LIGHTGREEN="\033[1;32m"
YELLOW="\033[1;33m"
LIGHTBLUE="\033[1;34m"
LIGHTPURPLE="\033[1;35m"
LIGHTCYAN="\033[1;36m"
WHITE="\033[1;37m"
CLEAR="\033[0m"

CUR_TEST=""

# Prints the status of a test
printStatus() {
    STATUS=$1

    if [ $STATUS -eq 0 ]; then
        echo
        echo -e "${CUR_TEST}:  [   ${LIGHTGREEN}OK${CLEAR}   ]"
        echo
    else
        echo
        echo -e "${CUR_TEST} [ ${LIGHTRED}FAILED${CLEAR} ]"
        echo
        exit 1
    fi
}

# Prints the current test name
startTest() {
    CUR_TEST=$1
    echo
    echo -e "${LIGHTBLUE}Starting test: ${WHITE}${CUR_TEST}${CLEAR}"
    echo
}
