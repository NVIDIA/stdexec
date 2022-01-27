/*
 * Copyright (c) NVIDIA
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

#include <algorithm>
#include <new>

#include <schedulers/detail/graph/consumer.hpp>
#include <schedulers/detail/graph/environment.hpp>
#include <schedulers/detail/graph/graph_instance.hpp>
#include <schedulers/detail/storage.hpp>

enum class device_type
{
  host,
  device
};

#ifdef _NVHPC_CUDA
#include <nv/target>

__host__ __device__ inline device_type get_device_type()
{
  if target (nv::target::is_host)
  {
    return device_type::host;
  }
  else
  {
    return device_type::device;
  }
}
#elif defined(__clang__) && defined(__CUDA__)
__host__ inline device_type get_device_type() { return device_type::host; }
__device__ inline device_type get_device_type() { return device_type::device; }
#endif

inline __host__ __device__ bool is_on_gpu()
{
  return get_device_type() == device_type::device;
}

template <int N = 1>
class flags_storage_t
{
  static_assert(N > 0);

  int *flags_{};

public:
  class flags_t
  {
    int *flags_{};

    flags_t(int *flags)
        : flags_(flags)
    {}

  public:
    __device__ void set(int i = 0) const { atomicAdd(flags_ + i, 1); }

    friend flags_storage_t;
  };

  flags_storage_t(const flags_storage_t &) = delete;
  flags_storage_t(flags_storage_t &&) = delete;

  void operator()(const flags_storage_t &) = delete;
  void operator()(flags_storage_t &&) = delete;

  flags_t get() { return {flags_}; }

  flags_storage_t()
  {
    cudaMalloc(&flags_, sizeof(int) * N);
    cudaMemset(flags_, 0, sizeof(int) * N);
  }

  ~flags_storage_t()
  {
    cudaFree(flags_);
    flags_ = nullptr;
  }

  bool all_set_once()
  {
    int flags[N];
    cudaMemcpy(flags, flags_, sizeof(int) * N, cudaMemcpyDeviceToHost);

    return std::count(std::begin(flags), std::end(flags), 1) == N;
  }

  bool all_unset() { return !all_set_once(); }
};

class receiver_tracer_t
{
  struct state_t
  {
    cudaGraph_t graph_{};

    std::size_t set_value_was_called_{};
    std::size_t set_stopped_was_called_{};
    std::size_t set_error_was_called_{};

    std::size_t num_nodes_{};
    std::size_t num_edges_{};

    state_t() { cudaGraphCreate(&graph_, 0); }

    ~state_t() { cudaGraphDestroy(graph_); }
  };

  state_t state_{};

public:
  struct receiver_t
  {
    state_t &state_;

    receiver_t(state_t &state)
        : state_(state)
    {}

    friend void tag_invoke(std::execution::set_value_t,
                           receiver_t &&self,
                           std::span<cudaGraphNode_t>) noexcept
    {
      cudaGraph_t graph = self.state_.graph_;
      cudaGraphGetNodes(graph, nullptr, &self.state_.num_nodes_);
      cudaGraphGetEdges(graph, nullptr, nullptr, &self.state_.num_edges_);

      self.state_.set_value_was_called_++;
    }

    friend void tag_invoke(std::execution::set_stopped_t,
                           receiver_t &&self) noexcept
    {
      self.state_.set_stopped_was_called_++;
    }

    friend void tag_invoke(std::execution::set_error_t,
                           receiver_t &&self,
                           std::exception_ptr) noexcept
    {
      self.state_.set_error_was_called_++;
    }

    [[nodiscard]] example::cuda::graph::detail::sink_consumer_t
    get_consumer() const noexcept
    {
      return {};
    }

    friend auto tag_invoke(std::execution::get_env_t, const receiver_t &self)
      -> example::cuda::graph::detail::env_t<std::execution::no_env>
    {
      return example::cuda::graph::detail::env_t<std::execution::no_env>{
        {},
        nullptr,
        self.state_.graph_};
    }

    static constexpr bool is_cuda_graph_api = true;
  };

  receiver_t get() { return {state_}; }

  [[nodiscard]] bool set_value_was_called() const
  {
    return state_.set_value_was_called_ > 0;
  }
  [[nodiscard]] bool set_stopped_was_called() const
  {
    return state_.set_stopped_was_called_ > 0;
  }
  [[nodiscard]] bool set_error_was_called() const
  {
    return state_.set_error_was_called_ > 0;
  }
  [[nodiscard]] bool set_value_was_called_once() const
  {
    return state_.set_value_was_called_ == 1;
  }
  [[nodiscard]] bool set_stopped_was_called_once() const
  {
    return state_.set_stopped_was_called_ == 1;
  }
  [[nodiscard]] bool set_error_was_called_once() const
  {
    return state_.set_error_was_called_ == 1;
  }

  [[nodiscard]] std::size_t num_nodes() const { return state_.num_nodes_; }
  [[nodiscard]] std::size_t num_edges() const { return state_.num_edges_; }
};

class tracer_t
{
  struct state_t
  {
    std::size_t n_copy_constructors{};
    std::size_t n_copy_assignments{};
  };

  struct state_storage_t
  {
    state_t *state_{};

    state_storage_t(const state_storage_t&) = delete;
    state_storage_t(state_storage_t&&) = delete;
    void operator=(const state_storage_t&) = delete;
    void operator=(state_storage_t&&) = delete;

    state_storage_t()
    {
      cudaMallocManaged(&state_, sizeof(state_t));
      new (state_) state_t{};
    }

    ~state_storage_t()
    {
      cudaFree(state_);
    }
  };

  state_storage_t state_storage_{};

public:
  struct accessor_t
  {
    state_t *state_;

    explicit accessor_t(state_t *state)
        : state_(state)
    {}

    __host__ __device__ accessor_t(const accessor_t& other)
      : state_(other.state_)
    {
      state_->n_copy_constructors++;
    }

    __host__ __device__ accessor_t& operator=(const accessor_t& other)
    {
      if (this != &other)
      {
        state_ = other.state_;
      }

      state_->n_copy_assignments++;
      return *this;
    }
  };

  [[nodiscard]] accessor_t get() const
  {
    return accessor_t{state_storage_.state_};
  }

  [[nodiscard]] std::size_t get_n_copy_constructions() const
  {
    return state_storage_.state_->n_copy_constructors;
  }

  [[nodiscard]] std::size_t get_n_copy_assignments() const
  {
    return state_storage_.state_->n_copy_assignments;
  }
};
