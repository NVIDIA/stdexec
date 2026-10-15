/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <catch2/catch_all.hpp>

#include <stdexec/execution.hpp>

namespace ex = STDEXEC;
using cb     = ex::__completion_behavior;

static_assert(!cb::__is_affine(cb::__asynchronous));
static_assert(cb::__is_affine(cb::__asynchronous_affine));
static_assert(cb::__is_affine(cb::__inline_completion));

static_assert(cb::__asynchronous != cb::__asynchronous_affine);
static_assert(cb::__asynchronous != cb::__inline_completion);
static_assert(cb::__asynchronous_affine != cb::__inline_completion);

static_assert((cb::__asynchronous | cb::__asynchronous) == cb::__asynchronous);
static_assert((cb::__asynchronous_affine | cb::__asynchronous_affine) == cb::__asynchronous_affine);
static_assert((cb::__inline_completion | cb::__inline_completion) == cb::__inline_completion);

static_assert((cb::__asynchronous | cb::__asynchronous_affine) == cb::__asynchronous);
static_assert((cb::__asynchronous | cb::__inline_completion) == cb::__unknown);
static_assert(int((cb::__asynchronous_affine | cb::__inline_completion).value)
              == (cb::__async_ | cb::__inline_));

static_assert(!cb::__is_affine(cb::__asynchronous));
static_assert(cb::__is_affine(cb::__inline_completion));
static_assert(cb::__is_affine(cb::__asynchronous_affine));

static_assert(!cb::__is_affine(cb::__asynchronous | cb::__inline_completion));
static_assert(!cb::__is_affine(cb::__asynchronous_affine | cb::__asynchronous));
static_assert(cb::__is_affine(cb::__asynchronous_affine | cb::__inline_completion));

static_assert(cb::__is_always_asynchronous(cb::__asynchronous));
static_assert(!cb::__is_always_asynchronous(cb::__inline_completion));
static_assert(cb::__is_always_asynchronous(cb::__asynchronous_affine));

static_assert(!cb::__is_always_asynchronous(cb::__asynchronous | cb::__inline_completion));
static_assert(cb::__is_always_asynchronous(cb::__asynchronous_affine | cb::__asynchronous));
static_assert(!cb::__is_always_asynchronous(cb::__asynchronous_affine | cb::__inline_completion));

static_assert(cb::__may_be_asynchronous(cb::__asynchronous));
static_assert(!cb::__may_be_asynchronous(cb::__inline_completion));
static_assert(cb::__may_be_asynchronous(cb::__asynchronous_affine));

static_assert(cb::__may_be_asynchronous(cb::__asynchronous | cb::__inline_completion));
static_assert(cb::__may_be_asynchronous(cb::__asynchronous_affine | cb::__asynchronous));
static_assert(cb::__may_be_asynchronous(cb::__asynchronous_affine | cb::__inline_completion));

// read_env completes in its start(), so get_scheduler() may claim inline completion
// for set_value_t -- but only for the tags it can actually send. Advertising
// inline completion for a tag it never sends made a task that co_awaits
// get_scheduler() | let_value(schedule) look like it completed inline, and the
// awaiter then kept the composed operation state in its own stack frame while a
// nested schedule() operation was still queued (see #2098).
using __get_scheduler_sndr_t  = decltype(ex::get_scheduler());
using __get_scheduler_attrs_t = ex::env_of_t<__get_scheduler_sndr_t>;

static_assert(ex::__completes_inline<ex::set_value_t, __get_scheduler_attrs_t>);
static_assert(!ex::__completes_inline<ex::set_error_t, __get_scheduler_attrs_t>);
static_assert(!ex::__completes_inline<ex::set_stopped_t, __get_scheduler_attrs_t>);

// The same expression from #2098, composed: the successor sender schedule(sch) may
// complete asynchronously, so the composition must not be treated as completing
// inline.
inline constexpr auto __reschedule = [](auto __sch)
{
  return ex::schedule(__sch);
};
inline constexpr auto __reschedule_on_current = ex::get_scheduler() | ex::let_value(__reschedule);

static_assert(
  !ex::__as_awaitable::__completes_inline<decltype(__reschedule_on_current), ex::env<>>);
