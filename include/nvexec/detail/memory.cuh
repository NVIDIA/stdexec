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

#include <memory_resource>
#include <new>
#include <type_traits>

#include "config.cuh"
#include "throw_on_cuda_error.cuh"

namespace nvexec::STDEXEC_STREAM_DETAIL_NS {

  struct device_deleter {
    template <class T>
    void operator()(T* ptr) {
      // ptr->~T();
      STDEXEC_DBG_ERR(::cudaFree(ptr));
    }
  };

  template <class T>
  using device_ptr = std::unique_ptr<T, device_deleter>;

  template <class T, class... As>
  device_ptr<T> make_device(cudaError_t& status, As&&... as) {
    static_assert(STDEXEC_IS_TRIVIALLY_COPYABLE(T));

    if (status == cudaSuccess) {
      T* ptr = nullptr;
      if (status = STDEXEC_DBG_ERR(::cudaMalloc(&ptr, sizeof(T))); status == cudaSuccess) {
        try {
          T h((As&&) as...);        
          status = STDEXEC_DBG_ERR(::cudaMemcpy(ptr, &h, sizeof(T), cudaMemcpyHostToDevice));
          if (status == cudaSuccess) {
            return device_ptr<T>(ptr);
          }
        } catch(...) {
          status = cudaErrorUnknown;
          STDEXEC_DBG_ERR(::cudaFree(ptr));
        }
      }
    }

    return device_ptr<T>();
  }

  template <class T = void, class A>
    requires same_as<T, void>
  device_ptr<__decay_t<A>> make_device(cudaError_t& status, A&& t) {
    return make_device<__decay_t<A>>(status, (A&&) t);
  }

  template <class T>
  struct host_deleter {
    std::pmr::memory_resource* resource_{nullptr};

    void operator()(T* ptr) const {
      if (ptr) {
        ptr->~T();
        resource_->deallocate(ptr, sizeof(T), alignof(T));
      }
    }
  };

  template <class T>
  using host_ptr = std::unique_ptr<T, host_deleter<T>>;

  template <class T, class... As>
  host_ptr<T> make_host(cudaError_t& status, std::pmr::memory_resource* resource, As&&... as) {
    T* ptr = nullptr;

    if (status == cudaSuccess) {
      try {
        ptr = static_cast<T*>(resource->allocate(sizeof(T), alignof(T)));
        ::new (static_cast<void*>(ptr)) T((As&&) as...);
        return host_ptr<T>(ptr, {resource});
      } catch (...) {
        if (ptr) {
          status = cudaErrorUnknown;
          resource->deallocate(ptr, sizeof(T), alignof(T));
        } else {
          status = cudaErrorMemoryAllocation;
        }
      }
    }

    return host_ptr<T>();
  }

  template <class T = void, class A>
    requires same_as<T, void>
  host_ptr<__decay_t<A>>
    make_host(cudaError_t& status, std::pmr::memory_resource* resource, A&& t) {
      return make_host<__decay_t<A>>(status, resource, (A&&) t);
    }

}
