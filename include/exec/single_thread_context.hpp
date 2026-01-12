/*
 * Copyright (c) 2021-2022 Facebook, Inc. and its affiliates.
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

#include "../stdexec/execution.hpp"

#include <thread>

namespace exec {
  class single_thread_context {
    STDEXEC::run_loop loop_;
    std::thread thread_;

   public:
    single_thread_context()
      : loop_()
      , thread_([this] { loop_.run(); }) {
    }

    ~single_thread_context() {
      loop_.finish();
      thread_.join();
    }

    auto get_scheduler() noexcept {
      return loop_.get_scheduler();
    }

    [[nodiscard]]
    auto get_thread_id() const noexcept -> std::thread::id {
      return thread_.get_id();
    }
  };
} // namespace exec
