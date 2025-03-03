/*
 * Copyright (c) David Eles 2024
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

#include <execpools/tbb/tbb_thread_pool.hpp>

#warning Deprecated header file, please include the <execpools/tbb/tbb_thread_pool.hpp> header file instead and use the execpools::tbb_thread_pool class that is identical as tbbexec::thread_pool class.

namespace tbbexec {
  using tbb_thread_pool = execpools::tbb_thread_pool;
} // namespace tbbexec
