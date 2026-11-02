# =============================================================================
# Copyright 2026 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

# Tests that a source file fails to build with an expected error.
#
# The source file states the expected error in one or more comments of the form
#
#   // build error: <regex>
#
# and the test passes when building the file fails and the build output matches
# at least one of those regular expressions.
#
# This file is used in two ways: included, it defines
# stdexec_add_build_failure_test(); run with `cmake -P`, it is the test itself.

if(CMAKE_SCRIPT_MODE_FILE)
  # Run as the test: cmake -DTARGET=<target> -DSOURCE=<file> -DBINARY_DIR=<dir>
  # [-DCONFIG=<config>] -P BuildFailureTests.cmake
  if(CONFIG)
    set(config_args --config ${CONFIG})
  endif()

  # Some compilers (MSVC) report errors on stdout, so collect both streams.
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build ${BINARY_DIR} --target ${TARGET}
            ${config_args}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE output)

  if(result EQUAL 0)
    message(FATAL_ERROR "Building '${TARGET}' succeeded, but it should fail.")
  endif()

  file(STRINGS ${SOURCE} expected_errors ENCODING UTF-8
       REGEX "//[ ]*build error:")
  set(tried_errors "")
  foreach(expected_error IN LISTS expected_errors)
    string(REGEX REPLACE ".*//[ ]*build error:[ \t]*" "" expected_error
                         "${expected_error}")
    # An empty regex would match any output.
    if("${expected_error}" STREQUAL "")
      continue()
    endif()
    if(output MATCHES "${expected_error}")
      message("Building '${TARGET}' failed with the expected error: "
              "${expected_error}")
      return()
    endif()
    string(APPEND tried_errors "\n  ${expected_error}")
  endforeach()

  if(tried_errors STREQUAL "")
    message(FATAL_ERROR "${SOURCE} has no '// build error: <regex>' comment.")
  endif()
  message("${output}")
  message(
    FATAL_ERROR
      "Building '${TARGET}' failed, but the output above matches none of the "
      "expected errors in ${SOURCE}:${tried_errors}")
endif()

# stdexec_add_build_failure_test(NAME <name> SOURCE <file> [LIBRARIES <lib>...])
#
# Adds an executable target <name> that is built only by the test <name>.
function(stdexec_add_build_failure_test)
  cmake_parse_arguments(ARG "" "NAME;SOURCE" "LIBRARIES" ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS OR NOT ARG_NAME OR NOT ARG_SOURCE)
    message(
      FATAL_ERROR
        "Usage: stdexec_add_build_failure_test(NAME <name> SOURCE <file> "
        "[LIBRARIES <lib>...]), got: ${ARGN}")
  endif()

  add_executable(${ARG_NAME} EXCLUDE_FROM_ALL ${ARG_SOURCE})
  set_target_properties(${ARG_NAME} PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD TRUE
                                               FOLDER test)
  target_link_libraries(${ARG_NAME} PRIVATE ${ARG_LIBRARIES})

  get_filename_component(source "${ARG_SOURCE}" ABSOLUTE)
  add_test(
    NAME ${ARG_NAME}
    COMMAND
      ${CMAKE_COMMAND} -DTARGET=${ARG_NAME} -DSOURCE=${source}
      -DBINARY_DIR=${CMAKE_BINARY_DIR} -DCONFIG=$<CONFIG> -P
      ${CMAKE_CURRENT_FUNCTION_LIST_FILE})
  # These tests run the build tool on the build tree, so run them one at a time.
  set_tests_properties(${ARG_NAME} PROPERTIES LABELS build-failure
                                              RESOURCE_LOCK stdexec_build_tree)
endfunction()
