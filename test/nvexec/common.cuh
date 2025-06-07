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

#include "../../include/stdexec/execution.hpp"
#include "../../include/nvexec/detail/throw_on_cuda_error.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

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

    auto get() -> flags_t {
      return {flags_};
    }

    flags_storage_t() {
      STDEXEC_TRY_CUDA_API(cudaMallocManaged(&flags_, sizeof(int) * N));
      STDEXEC_TRY_CUDA_API(cudaMemset(flags_, 0, sizeof(int) * N));
    }

    ~flags_storage_t() {
      STDEXEC_ASSERT_CUDA_API(cudaFree(flags_));
      flags_ = nullptr;
    }

    auto is_set_n_times(int n) -> bool {
      int host_flags[N]; // NOLINT
      STDEXEC_TRY_CUDA_API(cudaMemcpy(host_flags, flags_, sizeof(int) * N, cudaMemcpyDeviceToHost));

      return std::count(host_flags, host_flags + N, n) == N;
    }

    auto all_set_once() -> bool {
      return is_set_n_times(1);
    }

    auto all_unset() -> bool {
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

        void start() & noexcept {
          stdexec::start(inner_op_);
        }

        __t(Sender&& sender, Receiver&& receiver)
          : inner_op_{
              stdexec::connect(static_cast<Sender&&>(sender), static_cast<Receiver&&>(receiver))} {
        }
      };
    };

    template <class Sender, class Receiver>
    using _operation_state_t =
      stdexec::__t<operation_state<stdexec::__cvref_id<Sender>, stdexec::__id<Receiver>>>;

    template <class ReceiverId, class Fun>
    struct receiver {
      using Receiver = stdexec::__t<ReceiverId>;

      class __t : public stdexec::receiver_adaptor<__t, Receiver> {
        friend stdexec::receiver_adaptor<__t, Receiver>;

        static_assert(std::is_trivially_copyable_v<Receiver>);
        static_assert(std::is_trivially_copyable_v<Fun>);
        Fun f_;

       public:
        using __id = receiver;

        template <class... As>
          requires std::invocable<Fun, As...>
        STDEXEC_ATTRIBUTE(host, device)
        void set_value(As&&... as) && noexcept {
          using result_t = std::invoke_result_t<Fun, As...>;

          if constexpr (std::is_same_v<void, result_t>) {
            std::invoke(f_, static_cast<As&&>(as)...);
            stdexec::set_value(std::move(this->base()));
          } else {
            stdexec::set_value(std::move(this->base()), std::invoke(f_, static_cast<As&&>(as)...));
          }
        }

        explicit __t(Receiver rcvr, Fun fun)
          : stdexec::receiver_adaptor<__t, Receiver>(static_cast<Receiver&&>(rcvr))
          , f_(static_cast<Fun&&>(fun)) {
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
        using op_t =
          _operation_state_t<stdexec::__copy_cvref_t<Self, Sender>, _receiver_t<Receiver, Fun>>;

        template <class Self, class... Env>
        using __completions_t = stdexec::transform_completion_signatures<
          stdexec::__completion_signatures_of_t<stdexec::__copy_cvref_t<Self, Sender>, Env...>,
          stdexec::completion_signatures<>,
          stdexec::__mbind_front_q<stdexec::__set_value_invoke_t, Fun>::template __f
        >;

        template <stdexec::__decays_to<__t> Self, stdexec::receiver Receiver>
          requires stdexec::receiver_of<Receiver, __completions_t<Self, stdexec::env_of_t<Receiver>>>
        friend auto
          tag_invoke(stdexec::connect_t, Self&& self, Receiver&& rcvr) -> op_t<Self, Receiver> {
          return op_t<Self, Receiver>(
            static_cast<Self&&>(self).sndr_,
            _receiver_t<Receiver, Fun>(static_cast<Receiver&&>(rcvr), self.fun_));
        }

        template <stdexec::__decays_to<__t> Self, class... Env>
        static auto get_completion_signatures(Self&&, Env&&...) -> __completions_t<Self, Env...> {
          return {};
        }

        auto get_env() const noexcept -> stdexec::__fwd_env_t<stdexec::env_of_t<Sender>> {
          return stdexec::__env::__fwd_fn{}(stdexec::get_env(sndr_));
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

        void start() & noexcept {
          stdexec::start(inner_op_);
        }

        __t(Sender&& sender, Receiver&& receiver)
          : inner_op_{
              stdexec::connect(static_cast<Sender&&>(sender), static_cast<Receiver&&>(receiver))} {
        }
      };
    };

    template <class Sender, class Receiver>
    using _operation_state_t =
      stdexec::__t<operation_state<stdexec::__cvref_id<Sender>, stdexec::__id<Receiver>>>;

    template <class SenderId>
    struct sender {
      using Sender = stdexec::__t<SenderId>;

      struct __t {
        using __id = sender;
        using sender_concept = stdexec::sender_t;

        Sender sndr_;

        template <class Self, class Receiver>
        using op_t = _operation_state_t<stdexec::__copy_cvref_t<Self, Sender>, Receiver>;

        template <class Self, class... Env>
        using _completions_t =
          stdexec::__completion_signatures_of_t<stdexec::__copy_cvref_t<Self, Sender>, Env...>;

        template <stdexec::__decays_to<__t> Self, stdexec::receiver Receiver>
          requires stdexec::receiver_of<Receiver, _completions_t<Self, stdexec::env_of_t<Receiver>>>
        static auto connect(Self&& self, Receiver&& rcvr) -> op_t<Self, Receiver> {
          return op_t<Self, Receiver>(
            static_cast<Self&&>(self).sndr_, static_cast<Receiver&&>(rcvr));
        }

        template <stdexec::__decays_to<__t> Self, class... Env>
        static auto get_completion_signatures(Self&&, Env&&...) -> _completions_t<Self, Env...> {
          return {};
        }

        auto get_env() const noexcept -> stdexec::__fwd_env_t<stdexec::env_of_t<Sender>> {
          return stdexec::__env::__fwd_fn{}(stdexec::get_env(sndr_));
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
    auto operator()(_Sender&& __sndr, _Fun __fun) const -> sender_th<_Sender, _Fun> {
      return sender_th<_Sender, _Fun>{static_cast<_Sender&&>(__sndr), static_cast<_Fun&&>(__fun)};
    }

    template <class _Fun>
    auto operator()(_Fun __fun) const
      -> stdexec::__binder_back<a_sender_helper_t<a_sender_kind::then>, _Fun> {
      return {{static_cast<_Fun&&>(__fun)}, {}, {}};
    };
  };

  template <>
  struct a_sender_helper_t<a_sender_kind::receiverless> {
    template <class _Sender>
    using receiverless_sender_th = detail::a_receiverless_sender::_sender_t<_Sender>;

    template <stdexec::sender _Sender>
    auto operator()(_Sender&& __sndr) const -> receiverless_sender_th<_Sender> {
      return receiverless_sender_th<_Sender>{static_cast<_Sender&&>(__sndr)};
    }

    auto
      operator()() const -> stdexec::__binder_back<a_sender_helper_t<a_sender_kind::receiverless>> {
      return {{}, {}, {}};
    }
  };

  struct a_sender_t
    : a_sender_helper_t<a_sender_kind::then>
    , a_sender_helper_t<a_sender_kind::receiverless> {
    using a_sender_helper_t<a_sender_kind::then>::operator();
    using a_sender_helper_t<a_sender_kind::receiverless>::operator();
  };

  constexpr a_sender_t a_sender; // NOLINT (unused-const-variable)

  struct move_only_t {
    static constexpr auto invalid() -> int {
      return -42;
    }

    move_only_t() = delete;
    move_only_t(const move_only_t&) = delete;
    auto operator=(move_only_t&&) -> move_only_t& = delete;
    auto operator=(const move_only_t&) -> move_only_t& = delete;

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

    __host__ __device__ auto contains(int val) -> bool {
      if (this != self_) {
        std::printf("Error: move_only_t::contains failed: %p\n", static_cast<void*>(self_));
        return false;
      }

      return data_ == val;
    }

    int data_{invalid()};
    move_only_t* self_;
  };

  static_assert(!std::is_trivially_copyable_v<move_only_t>);
} // namespace
