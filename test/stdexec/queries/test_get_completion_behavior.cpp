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

#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>

namespace ex = STDEXEC;
using cb = ex::__completion_behavior;

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
static_assert(int((cb::__asynchronous_affine | cb::__inline_completion).value) == (cb::__async_ | cb::__inline_));

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
