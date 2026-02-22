/*
 * Copyright (c) 2023 Ben FrantzDale
 * Copyright (c) 2024 David Eles
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
#pragma once

#include <taskflow/taskflow.hpp>

#include "../../exec/thread_pool_base.hpp"

namespace execpools
{
  class taskflow_thread_pool : public exec::thread_pool_base<taskflow_thread_pool>
  {
   public:
    //! Constructor forwards to tbb::task_arena constructor:
    template <class... Args>
      requires STDEXEC::__std::constructible_from<tf::Executor, Args...>
    explicit taskflow_thread_pool(Args&&... args)
      : executor_(std::forward<Args>(args)...)
    {}

    [[nodiscard]]
    auto available_parallelism() const -> std::uint32_t
    {
      return static_cast<std::uint32_t>(executor_.num_workers());
    }
   private:
    [[nodiscard]]
    static constexpr auto forward_progress_guarantee() -> STDEXEC::forward_progress_guarantee
    {
      return STDEXEC::forward_progress_guarantee::parallel;
    }

    friend exec::thread_pool_base<taskflow_thread_pool>;

    template <class PoolType, class Receiver>
    friend struct exec::_pool_::opstate;

    void enqueue(exec::_pool_::task_base* task, std::uint32_t tid = 0) noexcept
    {
      executor_.silent_async([task, tid] { task->execute_(task, /*tid=*/tid); });
    }

    tf::Executor executor_;
  };
}  // namespace execpools
