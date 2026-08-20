/*
 * Copyright (c) 2026 NVIDIA Corporation
 *
 * Licensed under the Apache License Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// stdexec/__detail/__parallel_scheduler_default_impl_entry.hpp is excluded
// from VERIFY_INTERFACE_HEADER_SETS (see the root CMakeLists.txt) because it
// has a documented precondition: includers must define
// STDEXEC_PARALLEL_SCHEDULER_INLINE before including it. That precondition
// is otherwise only exercised when STDEXEC_BUILD_PARALLEL_SCHEDULER is
// enabled, which CI does not currently do.
//
// This translation unit exists solely to verify that, given the documented
// precondition, the header is otherwise self-contained. It is not linked
// into any test executable or library; see the verify_parallel_scheduler_
// default_impl_entry OBJECT library target in this directory's CMakeLists
// wiring, which is hooked into all_verify_interface_header_sets as an extra
// dependency.

#define STDEXEC_PARALLEL_SCHEDULER_INLINE inline
#include "stdexec/__detail/__parallel_scheduler_default_impl_entry.hpp"
