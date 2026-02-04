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
#include "memory.cuh"
#include "throw_on_cuda_error.cuh"

namespace nvexec::_strm::queue {
  struct task_base {
    using fn_t = void(task_base*) noexcept;

    task_base* next_{};
    task_base** atom_next_{};
    fn_t* execute_{};
    fn_t* free_{};
  };

  using task_ref_t = ::cuda::atomic_ref<task_base*, ::cuda::thread_scope_system>;
  using atom_task_ref_t = ::cuda::atomic_ref<task_base*, ::cuda::thread_scope_device>;

  struct producer {
    task_base** tail_;

    STDEXEC_ATTRIBUTE(host, device)
    void operator()(task_base* task) {
      atom_task_ref_t tail_ref(*tail_);
      task_base* old_tail = tail_ref.load(::cuda::memory_order_acquire);

      while (true) {
        atom_task_ref_t atom_next_ref(*old_tail->atom_next_);

        task_base* expected = nullptr;
        if (atom_next_ref.compare_exchange_weak(
              expected, task, ::cuda::memory_order_relaxed, ::cuda::memory_order_relaxed)) {
          task_ref_t next_ref(old_tail->next_);
          next_ref.store(task, ::cuda::memory_order_release);
          break;
        }

        old_tail = tail_ref.load(::cuda::memory_order_acquire);
      }

      tail_ref.compare_exchange_strong(old_tail, task);
    }
  };

  struct root_task : task_base {
    root_task() {
      this->execute_ = [](task_base*) noexcept {
      };
      this->free_ = [](task_base* t) noexcept {
        STDEXEC_ASSERT_CUDA_API(cudaFree(static_cast<void*>(t->atom_next_)));
      };
      this->next_ = nullptr;

      constexpr std::size_t ptr_size = sizeof(this->atom_next_);
      STDEXEC_TRY_CUDA_API(cudaMalloc(reinterpret_cast<void**>(&this->atom_next_), ptr_size));
      STDEXEC_TRY_CUDA_API(cudaMemset(static_cast<void*>(this->atom_next_), 0, ptr_size));
    }
  };

  struct poller {
    task_base* head_;
    std::thread poller_;
    ::cuda::std::atomic_flag stopped_{};

    poller(int dev_id, task_base* head)
      : head_(head) {
      poller_ = std::thread([dev_id, this] {
        cudaSetDevice(dev_id);

        task_base* current = head_;

        while (true) {
          task_ref_t next_ref(current->next_);

          while (next_ref.load(::cuda::memory_order_relaxed) == nullptr) {
            if (stopped_.test()) {
              current->free_(current);
              return;
            }
            std::this_thread::yield();
          }
          task_base* next = next_ref.load(::cuda::memory_order_acquire);
          current->free_(current);
          current = next;
          current->execute_(current);
        }
      });
    }

    ~poller() {
      stopped_.test_and_set();
      poller_.join();
    }
  };

  struct task_hub {
    cudaError_t status_{cudaSuccess};
    host_ptr_t<root_task> head_;
    device_ptr_t<task_base*> tail_ptr_;
    poller poller_;

    task_hub(int dev_id, std::pmr::memory_resource* pinned_resource)
      : head_(host_allocate<root_task>(status_, pinned_resource))
      , tail_ptr_(device_allocate<task_base*>(status_, head_.get()))
      , poller_(dev_id, head_.get()) {
    }

    auto producer() -> queue::producer {
      return queue::producer{tail_ptr_.get()};
    }
  };
} // namespace nvexec::_strm::queue
