/*
 * Copyright (c) 2022 NVIDIA Corporation
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

// clang-format Language: Cpp

#pragma once

#include <cstddef>
#include <memory_resource>
#include <thread>

#include "config.cuh"
#include "cuda_atomic.cuh" // IWYU pragma: keep
#include "throw_on_cuda_error.cuh"
#include "memory.cuh"

namespace nvexec::_strm::queue {
  struct task_base_t {
    using fn_t = void(task_base_t*) noexcept;

    task_base_t* next_{};
    task_base_t** atom_next_{};
    fn_t* execute_{};
    fn_t* free_{};
  };

  using task_ref = ::cuda::atomic_ref<task_base_t*, ::cuda::thread_scope_system>;
  using atom_task_ref = ::cuda::atomic_ref<task_base_t*, ::cuda::thread_scope_device>;

  struct producer_t {
    task_base_t** tail_;

    STDEXEC_ATTRIBUTE(host, device)
    void operator()(task_base_t* task) {
      atom_task_ref tail_ref(*tail_);
      task_base_t* old_tail = tail_ref.load(::cuda::memory_order_acquire);

      while (true) {
        atom_task_ref atom_next_ref(*old_tail->atom_next_);

        task_base_t* expected = nullptr;
        if (atom_next_ref.compare_exchange_weak(
              expected, task, ::cuda::memory_order_relaxed, ::cuda::memory_order_relaxed)) {
          task_ref next_ref(old_tail->next_);
          next_ref.store(task, ::cuda::memory_order_release);
          break;
        }

        old_tail = tail_ref.load(::cuda::memory_order_acquire);
      }

      tail_ref.compare_exchange_strong(old_tail, task);
    }
  };

  struct root_task_t : task_base_t {
    root_task_t() {
      this->execute_ = [](task_base_t*) noexcept {};
      this->free_ = [](task_base_t* t) noexcept {
        STDEXEC_ASSERT_CUDA_API(cudaFree(static_cast<void*>(t->atom_next_)));
      };
      this->next_ = nullptr;

      constexpr std::size_t ptr_size = sizeof(this->atom_next_);
      STDEXEC_TRY_CUDA_API(cudaMalloc(reinterpret_cast<void**>(&this->atom_next_), ptr_size));
      STDEXEC_TRY_CUDA_API(cudaMemset(static_cast<void*>(this->atom_next_), 0, ptr_size));
    }
  };

  struct poller_t {
    task_base_t* head_;
    std::thread poller_;
    ::cuda::std::atomic_flag stopped_{};

    poller_t(int dev_id, task_base_t* head)
      : head_(head) {
      poller_ = std::thread([dev_id, this] {
        cudaSetDevice(dev_id);

        task_base_t* current = head_;

        while (true) {
          task_ref next_ref(current->next_);

          while (next_ref.load(::cuda::memory_order_relaxed) == nullptr) {
            if (stopped_.test()) {
              current->free_(current);
              return;
            }
            std::this_thread::yield();
          }
          task_base_t* next = next_ref.load(::cuda::memory_order_acquire);
          current->free_(current);
          current = next;
          current->execute_(current);
        }
      });
    }

    ~poller_t() {
      stopped_.test_and_set();
      poller_.join();
    }
  };

  struct task_hub_t {
    cudaError_t status_{cudaSuccess};
    host_ptr<root_task_t> head_;
    device_ptr<task_base_t*> tail_ptr_;
    poller_t poller_;

    task_hub_t(int dev_id, std::pmr::memory_resource* pinned_resource)
      : head_(make_host<root_task_t>(status_, pinned_resource))
      , tail_ptr_(make_device<task_base_t*>(status_, head_.get()))
      , poller_(dev_id, head_.get()) {
    }

    auto producer() -> producer_t {
      return producer_t{tail_ptr_.get()};
    }
  };
} // namespace nvexec::_strm::queue
