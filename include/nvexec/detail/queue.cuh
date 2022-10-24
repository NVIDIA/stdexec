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
#pragma once

#include "../../stdexec/execution.hpp"

#include <type_traits>

#include "config.cuh"
#include "cuda_atomic.cuh"
#include "throw_on_cuda_error.cuh"

namespace nvexec::STDEXEC_STREAM_DETAIL_NS {
  namespace queue {
    struct task_base_t {
      using fn_t = void(task_base_t*) noexcept;

      task_base_t* next_{};
      task_base_t** atom_next_{};
      fn_t* execute_{};
      fn_t* free_{};
    };

    struct device_deleter_t {
      template <class T>
      void operator()(T *ptr) {
        STDEXEC_DBG_ERR(cudaFree(ptr));
      }
    };

    template <class T>
      using device_ptr = std::unique_ptr<T, device_deleter_t>;

    template <class T, class... As>
      device_ptr<T> make_device(cudaError_t &status, As&&... as) {
        static_assert(std::is_trivially_copyable_v<T>);

        if (status == cudaSuccess) {
          T* ptr{};
          if (status = STDEXEC_DBG_ERR(cudaMalloc(&ptr, sizeof(T))); status == cudaSuccess) {
            T h((As&&)as...);
            status = STDEXEC_DBG_ERR(cudaMemcpy(ptr, &h, sizeof(T), cudaMemcpyHostToDevice)); 
            return device_ptr<T>(ptr);
          }
        }

        return device_ptr<T>();
      }

    struct host_deleter_t {
      template <class T>
      void operator()(T *ptr) {
        STDEXEC_DBG_ERR(cudaFreeHost(ptr));
      }
    };

    template <class T>
      using host_ptr = std::unique_ptr<T, host_deleter_t>;
    using task_ref = ::cuda::atomic_ref<task_base_t*, ::cuda::thread_scope_system>;
    using atom_task_ref = ::cuda::atomic_ref<task_base_t*, ::cuda::thread_scope_device>;

    template <class T, class... As>
      host_ptr<T> make_host(cudaError_t &status, As&&... as) {
        T* ptr{};

        if (status == cudaSuccess) {
          if (status = STDEXEC_DBG_ERR(cudaMallocHost(&ptr, sizeof(T))); status == cudaSuccess) {
            new (ptr) T((As&&)as...);
            return host_ptr<T>(ptr);
          }
        }

        return host_ptr<T>();
      }

    struct producer_t {
      task_base_t** tail_;

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
        this->execute_ = [](task_base_t* t) noexcept {};
        this->free_ = [](task_base_t* t) noexcept {
          STDEXEC_DBG_ERR(cudaFree(t->atom_next_));
        };
        this->next_ = nullptr;

        constexpr std::size_t ptr_size = sizeof(this->atom_next_);
        STDEXEC_DBG_ERR(cudaMalloc(&this->atom_next_, ptr_size));
        STDEXEC_DBG_ERR(cudaMemset(this->atom_next_, 0, ptr_size));
      }
    };

    struct poller_t {
      task_base_t *head_;
      std::thread poller_;
      ::cuda::std::atomic_flag stopped_ = ATOMIC_FLAG_INIT;

      poller_t(task_base_t* head) : head_(head) {
        poller_ = std::thread([this] {
          task_base_t* current = head_;

          while (true) {
            task_ref next_ref(current->next_);

            while(next_ref.load(::cuda::memory_order_relaxed) == nullptr) {
              if (stopped_.test()) {
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

      task_hub_t()
        : head_(make_host<root_task_t>(status_))
        , tail_ptr_(make_device<task_base_t*>(status_, head_.get()))
        , poller_(head_.get()) {
      }

      producer_t producer() {
        return producer_t{tail_ptr_.get()};
      }
    };
  }
}

