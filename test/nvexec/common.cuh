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

  inline void throw_on_cuda_error(cudaError_t error, char const* file_name, int line) {
    // Clear the global CUDA error state which may have been set by the last
    // call. Otherwise, errors may "leak" to unrelated calls.
    cudaGetLastError();

    if (error != cudaSuccess) {
      throw std::runtime_error(std::string("CUDA Error: ")
                             + file_name
                             + ":"
                             + std::to_string(line)
                             + ": "
                             + cudaGetErrorName(error)
                             + ": "
                             + cudaGetErrorString(error));
    }
  }

  #define THROW_ON_CUDA_ERROR(...)                     \
    ::nvexec::throw_on_cuda_error(__VA_ARGS__, __FILE__, __LINE__); \
    /**/
}

template <int N = 1>
  requires (N > 0)
class flags_storage_t {
  int *flags_{};

public:
  class flags_t {
    int *flags_{};

    flags_t(int *flags)
      : flags_(flags) {
    }

  public:
    __device__ void set(int idx = 0) const {
      if (idx < N) {
        flags_[idx] += 1;
      }
    }

    friend flags_storage_t;
  };

  flags_storage_t(const flags_storage_t &) = delete;
  flags_storage_t(flags_storage_t &&) = delete;

  void operator()(const flags_storage_t &) = delete;
  void operator()(flags_storage_t &&) = delete;

  flags_t get() {
    return {flags_};
  }

  flags_storage_t() {
    THROW_ON_CUDA_ERROR(cudaMallocHost(&flags_, sizeof(int) * N));
    memset(flags_, 0, sizeof(int) * N);
  }

  ~flags_storage_t() {
    THROW_ON_CUDA_ERROR(cudaFreeHost(flags_));
    flags_ = nullptr;
  }

  bool is_set_n_times(int n) {
    return std::count(flags_, flags_ + N, n) == N;
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
    struct operation_state_t {
      using Sender = stdexec::__t<SenderId>;
      using Receiver = stdexec::__t<ReceiverId>;
      using inner_op_state_t = stdexec::connect_result_t<Sender, Receiver>;

      inner_op_state_t inner_op_;

      friend void tag_invoke(stdexec::start_t, operation_state_t& op) noexcept {
        stdexec::start(op.inner_op_);
      }

      operation_state_t(Sender&& sender, Receiver&& receiver)
        : inner_op_{stdexec::connect((Sender&&)sender, (Receiver&&)receiver)}
      {}
    };

  template <class ReceiverId, class Fun>
    class receiver_t : stdexec::receiver_adaptor<receiver_t<ReceiverId, Fun>, stdexec::__t<ReceiverId>> {
      using Receiver = stdexec::__t<ReceiverId>;
      friend stdexec::receiver_adaptor<receiver_t, Receiver>;

      Fun f_;

      template <class... As>
      void set_value(As&&... as) && noexcept
        requires stdexec::__callable<Fun, As&&...> {
        using result_t = std::invoke_result_t<Fun, As&&...>;

        if constexpr (std::is_same_v<void, result_t>) {
          f_((As&&)as...);
          stdexec::set_value(std::move(this->base()));
        } else {
          stdexec::set_value(std::move(this->base()), f_((As&&)as...));
        }
      }

     public:
      explicit receiver_t(Receiver rcvr, Fun fun)
        : stdexec::receiver_adaptor<receiver_t, Receiver>((Receiver&&) rcvr)
        , f_((Fun&&) fun)
      {}
    };

  template <class SenderId, class FunId>
    struct sender_t {
      using Sender = stdexec::__t<SenderId>;
      using Fun = stdexec::__t<FunId>;

      Sender sndr_;
      Fun fun_;

      template <class Receiver>
        using receiver_th = receiver_t<stdexec::__x<Receiver>, Fun>;

      template <class Self, class Receiver>
        using op_t = operation_state_t<
          stdexec::__x<stdexec::__member_t<Self, Sender>>,
          stdexec::__x<receiver_th<Receiver>>>;

      template <class Self, class Env>
        using completion_signatures =
          stdexec::__make_completion_signatures<
            stdexec::__member_t<Self, Sender>,
            Env,
            stdexec::completion_signatures<>,
            stdexec::__mbind_front_q<stdexec::__set_value_invoke_t, Fun>>;

      template <stdexec::__decays_to<sender_t> Self, stdexec::receiver Receiver>
        requires stdexec::receiver_of<Receiver, completion_signatures<Self, stdexec::env_of_t<Receiver>>>
      friend auto tag_invoke(stdexec::connect_t, Self&& self, Receiver&& rcvr)
        -> op_t<Self, Receiver> {
        return op_t<Self, Receiver>(((Self&&)self).sndr_, receiver_th<Receiver>((Receiver&&)rcvr, self.fun_));
      }

      template <stdexec::__decays_to<sender_t> Self, class Env>
      friend auto tag_invoke(stdexec::get_completion_signatures_t, Self&&, Env)
        -> stdexec::dependent_completion_signatures<Env>;

      template <stdexec::__decays_to<sender_t> Self, class Env>
      friend auto tag_invoke(stdexec::get_completion_signatures_t, Self&&, Env)
        -> completion_signatures<Self, Env> requires true;

      template <stdexec::tag_category<stdexec::forwarding_sender_query> Tag, class... As>
        requires stdexec::__callable<Tag, const Sender&, As...>
      friend auto tag_invoke(Tag tag, const sender_t& self, As&&... as)
        noexcept(stdexec::__nothrow_callable<Tag, const Sender&, As...>)
        -> stdexec::__call_result_if_t<stdexec::tag_category<Tag, stdexec::forwarding_sender_query>, Tag, const Sender&, As...> {
        return ((Tag&&) tag)(self.sndr_, (As&&) as...);
      }
    };
}

namespace detail::a_receiverless_sender {
  template <class SenderId, class ReceiverId>
    struct operation_state_t {
      using Sender = stdexec::__t<SenderId>;
      using Receiver = stdexec::__t<ReceiverId>;
      using inner_op_state_t = stdexec::connect_result_t<Sender, Receiver>;

      inner_op_state_t inner_op_;

      friend void tag_invoke(stdexec::start_t, operation_state_t& op) noexcept {
        stdexec::start(op.inner_op_);
      }

      operation_state_t(Sender&& sender, Receiver&& receiver)
        : inner_op_{stdexec::connect((Sender&&)sender, (Receiver&&)receiver)}
      {}
    };

  template <class SenderId>
    struct sender_t {
      using Sender = stdexec::__t<SenderId>;

      Sender sndr_;

      template <class Self, class Receiver>
        using op_t = operation_state_t<
          stdexec::__x<stdexec::__member_t<Self, Sender>>,
          stdexec::__x<Receiver>>;

      template <class Self, class Env>
        using completion_signatures =
          stdexec::__make_completion_signatures<
            stdexec::__member_t<Self, Sender>,
            Env,
            stdexec::completion_signatures<>>;

      template <stdexec::__decays_to<sender_t> Self, stdexec::receiver Receiver>
        requires stdexec::receiver_of<Receiver, completion_signatures<Self, stdexec::env_of_t<Receiver>>>
      friend auto tag_invoke(stdexec::connect_t, Self&& self, Receiver&& rcvr)
        -> op_t<Self, Receiver> {
        return op_t<Self, Receiver>(((Self&&)self).sndr_, (Receiver&&)rcvr);
      }

      template <stdexec::__decays_to<sender_t> Self, class Env>
      friend auto tag_invoke(stdexec::get_completion_signatures_t, Self&&, Env)
        -> stdexec::dependent_completion_signatures<Env>;

      template <stdexec::__decays_to<sender_t> Self, class Env>
      friend auto tag_invoke(stdexec::get_completion_signatures_t, Self&&, Env)
        -> completion_signatures<Self, Env> requires true;

      template <stdexec::tag_category<stdexec::forwarding_sender_query> Tag, class... As>
        requires stdexec::__callable<Tag, const Sender&, As...>
      friend auto tag_invoke(Tag tag, const sender_t& self, As&&... as)
        noexcept(stdexec::__nothrow_callable<Tag, const Sender&, As...>)
        -> stdexec::__call_result_if_t<stdexec::tag_category<Tag, stdexec::forwarding_sender_query>, Tag, const Sender&, As...> {
        return ((Tag&&) tag)(self.sndr_, (As&&) as...);
      }
    };
}

struct a_sender_t {
  template <class _Sender, class _Fun>
    using sender_th = detail::a_sender::sender_t<
      stdexec::__x<std::remove_cvref_t<_Sender>>,
      stdexec::__x<std::remove_cvref_t<_Fun>>>;

  template <class _Sender>
    using receiverless_sender_th = detail::a_receiverless_sender::sender_t<
      stdexec::__x<std::remove_cvref_t<_Sender>>>;

  template <stdexec::sender _Sender, class _Fun>
      requires stdexec::sender<sender_th<_Sender, _Fun>>
    sender_th<_Sender, _Fun> operator()(_Sender&& __sndr, _Fun __fun) const {
      return sender_th<_Sender, _Fun>{(_Sender&&) __sndr, (_Fun&&) __fun};
    }

  template <class _Fun>
    stdexec::__binder_back<a_sender_t, _Fun> operator()(_Fun __fun) const {
      return {{}, {}, {(_Fun&&) __fun}};
    }

  template <stdexec::sender _Sender>
      requires stdexec::sender<receiverless_sender_th<_Sender>>
    receiverless_sender_th<_Sender> operator()(_Sender&& __sndr) const {
      return receiverless_sender_th<_Sender>{(_Sender&&) __sndr};
    }

  stdexec::__binder_back<a_sender_t> operator()() const {
    return {{}, {}, {}};
  }
};

constexpr a_sender_t a_sender;

struct move_only_t {
  static constexpr int invalid = -42;

  move_only_t() = delete;
  move_only_t(int data) 
    : data_(data)
    , self_(this) {
  }

  move_only_t(move_only_t&& other) 
    : data_(other.data_)
    , self_(this) {
  }

  ~move_only_t() {
    if (this != self_) {
      // TODO Trap
      std::printf("Error: move_only_t::~move_only_t failed\n");
    }
  }

  bool contains(int val) {
    if (this != self_) {
      return false;
    }

    return data_ == val;
  }

  int data_{invalid};
  move_only_t* self_;
};

