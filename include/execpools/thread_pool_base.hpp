/*
 * Copyright (c) 2023 Ben FrantzDale
 * Copyright (c) 2021-2023 Facebook, Inc. and its affiliates.
 * Copyright (c) 2021-2024 NVIDIA Corporation
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
#pragma once

#include "../stdexec/__detail/__config.hpp"

#if STDEXEC_MSVC()
#  pragma message(                                                                                 \
    "WARNING: The header <execpools/thread_pool_base.hpp> is deprecated. Please include <exec/thread_pool_base.hpp> instead.")
#else
#  warning                                                                                         \
    "The header <execpools/thread_pool_base.hpp> is deprecated. Please include <exec/thread_pool_base.hpp> instead."
#endif

#include "../exec/thread_pool_base.hpp"

namespace execpools
{
  template <class DerivedPoolType>
  using thread_pool_base
    [[deprecated("execpools::thread_pool_base has been renamed to "
                 "exec::thread_pool_base")]] = exec::thread_pool_base<DerivedPoolType>;
}  // namespace execpools
