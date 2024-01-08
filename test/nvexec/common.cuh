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

#include <algorithm>
#include <stdexec/execution.hpp>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>

namespace nvexec {

  inline void throw_on_cuda_error(cudaError_t error, char const * file_name, int line) {
    // Clear the global CUDA error state which may have been set by the last
    // call. Otherwise, errors may "leak" to unrelated calls.
    cudaGetLastError();

    if (error != cudaSuccess) {
      throw std::runtime_error(
        std::string("CUDA Error: ") + file_name + ":" + std::to_string(line) + ": "
        + cudaGetErrorName(error) + ": " + cudaGetErrorString(error));
    }
  }

#define THROW_ON_CUDA_ERROR(...) \
  ::nvexec::throw_on_cuda_error(__VA_ARGS__, __FILE__, __LINE__); \
  /**/
} // namespace nvexec

namespace {

  template <int N = 1>
    requires(N > 0)
  class flags_storage_t {
    int* flags_{};

   public:
    class flags_t {
      int* flags_{};

      flags_t(int* flags)
        : flags_(flags) {
      }

     public:
      __device__ __host__ void set(int idx = 0) const {
        if (idx < N) {
          flags_[idx] += 1;
        }
      }

      friend flags_storage_t;
    };

    flags_storage_t(const flags_storage_t&) = delete;
    flags_storage_t(flags_storage_t&&) = delete;

    void operator()(const flags_storage_t&) = delete;
    void operator()(flags_storage_t&&) = delete;

    flags_t get() {
      return {flags_};
    }

    flags_storage_t() {
      THROW_ON_CUDA_ERROR(cudaMallocManaged(&flags_, sizeof(int) * N));
      THROW_ON_CUDA_ERROR(cudaMemset(flags_, 0, sizeof(int) * N));
    }

    ~flags_storage_t() {
      THROW_ON_CUDA_ERROR(cudaFree(flags_));
      flags_ = nullptr;
    }

    bool is_set_n_times(int n) {
      int host_flags[N];
      THROW_ON_CUDA_ERROR(cudaMemcpy(host_flags, flags_, sizeof(int) * N, cudaMemcpyDeviceToHost));

      return std::count(host_flags, host_flags + N, n) == N;
    }

    bool all_set_once() {
      return is_set_n_times(1);
    }

    bool all_unset() {
      return !all_set_once();
    }
  };

  namespace detail::a_sender {
    template <class SenderId, class ReceiverId>
    struct operation_state {
      using Sender = stdexec::__cvref_t<SenderId>;
      using Receiver = stdexec::__t<ReceiverId>;

      struct __t {
        using __id = operation_state;
        using inner_op_state_t = stdexec::connect_result_t<Sender, Receiver>;

        inner_op_state_t inner_op_;

        friend void tag_invoke(stdexec::start_t, __t& op) noexcept {
          stdexec::start(op.inner_op_);
        }

        __t(Sender&& sender, Receiver&& receiver)
          : inner_op_{stdexec::connect((Sender&&) sender, (Receiver&&) receiver)} {
        }
      };
    };

    template <class Sender, class Receiver>
    using _operation_state_t = //
      stdexec::__t<operation_state<stdexec::__cvref_id<Sender>, stdexec::__id<Receiver>>>;

    template <class ReceiverId, class Fun>
    struct receiver {
      using Receiver = stdexec::__t<ReceiverId>;
      class __t : stdexec::receiver_adaptor<__t, Receiver> {
        friend stdexec::receiver_adaptor<__t, Receiver>;

        static_assert(std::is_trivially_copyable_v<Receiver>);
        static_assert(std::is_trivially_copyable_v<Fun>);
        Fun f_;

        template <class... As>
        STDEXEC_ATTRIBUTE((host, device))
        void set_value(As&&... as) && noexcept
          requires stdexec::__callable<Fun, As&&...>
        {
          using result_t = std::invoke_result_t<Fun, As&&...>;

          if constexpr (std::is_same_v<void, result_t>) {
            f_((As&&) as...);
            stdexec::set_value(std::move(this->base()));
          } else {
            stdexec::set_value(std::move(this->base()), f_((As&&) as...));
          }
        }

      public:
        using __id = receiver;
        using receiver_concept = stdexec::receiver_t;

        explicit __t(Receiver rcvr, Fun fun)
          : stdexec::receiver_adaptor<__t, Receiver>((Receiver&&) rcvr)
          , f_((Fun&&) fun) {
        }
      };
    };

    template <class Receiver, class Fun>
    using _receiver_t = stdexec::__t<receiver<stdexec::__id<Receiver>, Fun>>;

    template <class SenderId, class Fun>
    struct sender {
      using Sender = stdexec::__t<SenderId>;

      struct __t {
        using __id = sender;
        using sender_concept = stdexec::sender_t;

        Sender sndr_;
        Fun fun_;

        template <class Self, class Receiver>
        using op_t = _operation_state_t<
          stdexec::__copy_cvref_t<Self, Sender>,
          _receiver_t<Receiver, Fun>>;

        template <class Self, class Env>
        using __completions_t = //
          stdexec::__try_make_completion_signatures<
            stdexec::__copy_cvref_t<Self, Sender>,
            Env,
            stdexec::completion_signatures<>,
            stdexec::__mbind_front_q<stdexec::__set_value_invoke_t, Fun>>;

        template <stdexec::__decays_to<__t> Self, stdexec::receiver Receiver>
          requires stdexec::receiver_of<Receiver, __completions_t<Self, stdexec::env_of_t<Receiver>>>
        friend auto tag_invoke(stdexec::connect_t, Self&& self, Receiver&& rcvr) //
          -> op_t<Self, Receiver> {
          return op_t<Self, Receiver>(
            ((Self&&) self).sndr_, _receiver_t<Receiver, Fun>((Receiver&&) rcvr, self.fun_));
        }

        template <stdexec::__decays_to<__t> Self, class Env>
        friend auto tag_invoke(stdexec::get_completion_signatures_t, Self&&, Env)
          -> __completions_t<Self, Env> {
          return {};
        }

        friend auto tag_invoke(stdexec::get_env_t, const __t& self) noexcept
            -> stdexec::env_of_t<const Sender&> {
          return stdexec::get_env(self.sndr_);
        }
      };
    };

    template <class Sender, class Fun>
    using _sender_t = stdexec::__t<sender<stdexec::__id<stdexec::__decay_t<Sender>>, Fun>>;
  } // namespace detail::a_sender

  namespace detail::a_receiverless_sender {
    template <class SenderId, class ReceiverId>
    struct operation_state {
      using Sender = stdexec::__cvref_t<SenderId>;
      using Receiver = stdexec::__t<ReceiverId>;

      struct __t {
        using __id = operation_state;
        using inner_op_state_t = stdexec::connect_result_t<Sender, Receiver>;

        inner_op_state_t inner_op_;

        friend void tag_invoke(stdexec::start_t, __t& op) noexcept {
          stdexec::start(op.inner_op_);
        }

        __t(Sender&& sender, Receiver&& receiver)
          : inner_op_{stdexec::connect((Sender&&) sender, (Receiver&&) receiver)} {
        }
      };
    };

    template <class Sender, class Receiver>
    using _operation_state_t = //
      stdexec::__t<operation_state<stdexec::__cvref_id<Sender>, stdexec::__id<Receiver>>>;

    template <class SenderId>
    struct sender {
      using Sender = stdexec::__t<SenderId>;

      struct __t {
        using __id = sender;
        using sender_concept = stdexec::sender_t;

        Sender sndr_;

        template <class Self, class Receiver>
        using op_t = _operation_state_t<
          stdexec::__copy_cvref_t<Self, Sender>,
          Receiver>;

        template <class Self, class Env>
        using completion_signatures = //
          stdexec::__try_make_completion_signatures<
            stdexec::__copy_cvref_t<Self, Sender>,
            Env,
            stdexec::completion_signatures<>>;

        template <stdexec::__decays_to<__t> Self, stdexec::receiver Receiver>
          requires stdexec::
            receiver_of<Receiver, completion_signatures<Self, stdexec::env_of_t<Receiver>>>
          friend auto
          tag_invoke(stdexec::connect_t, Self&& self, Receiver&& rcvr) -> op_t<Self, Receiver> {
          return op_t<Self, Receiver>(((Self&&) self).sndr_, (Receiver&&) rcvr);
        }

        template <stdexec::__decays_to<__t> Self, class Env>
        friend auto tag_invoke(stdexec::get_completion_signatures_t, Self&&, Env)
          -> completion_signatures<Self, Env> {
          return {};
        }

        friend auto tag_invoke(stdexec::get_env_t, const __t& self) //
          noexcept(stdexec::__nothrow_callable<stdexec::get_env_t, const Sender&>)
            -> stdexec::env_of_t<const Sender&> {
          return stdexec::get_env(self.sndr_);
        }
      };
    };

    template <class Sender>
    using _sender_t = stdexec::__t<sender<stdexec::__id<stdexec::__decay_t<Sender>>>>;
  } // namespace detail::a_receiverless_sender

  enum class a_sender_kind {
    then,
    receiverless
  };

  template <a_sender_kind kind>
  struct a_sender_helper_t;

  template <>
  struct a_sender_helper_t<a_sender_kind::then> {
    template <class _Sender, class _Fun>
    using sender_th = detail::a_sender::_sender_t<_Sender, _Fun>;

    template <stdexec::sender _Sender, class _Fun>
      requires stdexec::sender<sender_th<_Sender, _Fun>>
    sender_th<_Sender, _Fun> operator()(_Sender&& __sndr, _Fun __fun) const {
      return sender_th<_Sender, _Fun>{(_Sender&&) __sndr, (_Fun&&) __fun};
    }

    template <class _Fun>
    stdexec::__binder_back<a_sender_helper_t<a_sender_kind::then>, _Fun>
      operator()(_Fun __fun) const {
      return {{}, {}, {(_Fun&&) __fun}};
    };
  };

  template <>
  struct a_sender_helper_t<a_sender_kind::receiverless> {
    template <class _Sender>
    using receiverless_sender_th = detail::a_receiverless_sender::_sender_t<_Sender>;

    template <stdexec::sender _Sender>
      requires stdexec::sender<receiverless_sender_th<_Sender>>
    receiverless_sender_th<_Sender> operator()(_Sender&& __sndr) const {
      return receiverless_sender_th<_Sender>{(_Sender&&) __sndr};
    }

    stdexec::__binder_back<a_sender_helper_t<a_sender_kind::receiverless>> operator()() const {
      return {{}, {}, {}};
    }
  };

  struct a_sender_t
    : a_sender_helper_t<a_sender_kind::then>
    , a_sender_helper_t<a_sender_kind::receiverless> {
    using a_sender_helper_t<a_sender_kind::then>::operator();
    using a_sender_helper_t<a_sender_kind::receiverless>::operator();
  };

  constexpr a_sender_t a_sender;

  struct move_only_t {
    static constexpr int invalid() {
      return -42;
    }

    move_only_t() = delete;
    move_only_t(const move_only_t&) = delete;
    move_only_t& operator=(move_only_t&&) = delete;
    move_only_t& operator=(const move_only_t&) = delete;

    __host__ __device__ move_only_t(int data)
      : data_(data)
      , self_(this) {
    }

    __host__ __device__ move_only_t(move_only_t&& other)
      : data_(std::exchange(other.data_, invalid()))
      , self_(this) {
    }

    __host__ __device__ ~move_only_t() {
      if (this != self_) {
        // TODO Trap
        std::printf("Error: move_only_t::~move_only_t failed\n");
      }
      data_ = invalid();
    }

    __host__ __device__ bool contains(int val) {
      if (this != self_) {
        std::printf("Error: move_only_t::contains failed: %p\n", (void*) self_);
        return false;
      }

      return data_ == val;
    }

    int data_{invalid()};
    move_only_t* self_;
  };

  static_assert(!std::is_trivially_copyable_v<move_only_t>);
}
