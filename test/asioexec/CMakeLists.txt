# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#                         Copyright (c) 2025 Robert Leahy. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(asioexec_test_sources
    ../test_main.cpp
    test_completion_token.cpp
    test_use_sender.cpp
    )

add_executable(test.asioexec ${asioexec_test_sources})
target_link_libraries(test.asioexec
    PUBLIC
    STDEXEC::stdexec
    $<TARGET_NAME_IF_EXISTS:STDEXEC::asioexec_boost>
    $<TARGET_NAME_IF_EXISTS:STDEXEC::asioexec_asio>
    stdexec_executable_flags
    Catch2::Catch2
    PRIVATE
    common_test_settings)

catch_discover_tests(test.asioexec)
