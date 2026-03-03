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

#include <tbb/task_arena.h>

#include "../thread_pool_base.hpp"

namespace experimental::execution::tbb
{
  class tbb_thread_pool : public exec::thread_pool_base<tbb_thread_pool>
  {
   public:
    //! Constructor forwards to ::tbb::task_arena constructor:
    template <class... Args>
      requires STDEXEC::__std::constructible_from<::tbb::task_arena, Args...>
    explicit tbb_thread_pool(Args&&... args)
      : arena_{std::forward<Args>(args)...}
    {
      arena_.initialize();
    }

    [[nodiscard]]
    auto available_parallelism() const -> std::uint32_t
    {
      return static_cast<std::uint32_t>(arena_.max_concurrency());
    }

   private:
    friend exec::thread_pool_base<tbb_thread_pool>;

    template <class PoolType, class Receiver>
    friend struct exec::_pool_::opstate;

    [[nodiscard]]
    static constexpr auto forward_progress_guarantee() -> STDEXEC::forward_progress_guarantee
    {
      return STDEXEC::forward_progress_guarantee::parallel;
    }

    void enqueue(exec::_pool_::task_base* task, std::uint32_t tid = 0) noexcept
    {
      arena_.enqueue([task, tid] { task->execute_(task, /*tid=*/tid); });
    }

    ::tbb::task_arena arena_{::tbb::task_arena::attach{}};
  };
}  // namespace experimental::execution::tbb

namespace exec = experimental::execution;
