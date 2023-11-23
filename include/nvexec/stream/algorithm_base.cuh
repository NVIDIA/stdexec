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
#include "../../stdexec/__detail/__ranges.hpp"
#include <type_traits>

#include <cuda/std/type_traits>

#include <cub/device/device_reduce.cuh>

#include "common.cuh"
#include "../detail/throw_on_cuda_error.cuh"

namespace nvexec::STDEXEC_STREAM_DETAIL_NS::__algo_range_init_fun {
  template <class Range, class InitT, class Fun>
  using binary_invoke_result_t = ::cuda::std::decay_t<
    ::cuda::std::invoke_result_t<Fun, stdexec::ranges::range_reference_t<Range>, InitT>>;

  template <class SenderId, class ReceiverId, class InitT, class Fun, class DerivedReceiver>
  struct receiver_t {
    struct __t : public stream_receiver_base {
      using Sender = stdexec::__t<SenderId>;
      using Receiver = stdexec::__t<ReceiverId>;

      template <class... Range>
      struct result_size_for {
        using __t = __msize_t< sizeof(typename DerivedReceiver::template result_t<Range...>)>;
      };

      template <class... Sizes>
      struct max_in_pack {
        static constexpr ::std::size_t value = ::std::max({::std::size_t{}, __v<Sizes>...});
      };

      struct max_result_size {
        template <class... _As>
        using result_size_for_t = stdexec::__t<result_size_for<_As...>>;

        static constexpr ::std::size_t value = //
          __v< __gather_completions_for<
            set_value_t,
            Sender,
            env_of_t<Receiver>,
            __q<result_size_for_t>,
            __q<max_in_pack>>>;
      };

      operation_state_base_t<ReceiverId>& op_state_;
      STDEXEC_ATTRIBUTE((no_unique_address)) InitT init_;
      STDEXEC_ATTRIBUTE((no_unique_address)) Fun fun_;

     public:
      using __id = receiver_t;

      constexpr static ::std::size_t memory_allocation_size = max_result_size::value;

      template <same_as<set_value_t> _Tag, class Range>
      friend void tag_invoke(_Tag, __t&& self, Range&& range) noexcept {
        DerivedReceiver::set_value_impl((__t&&) self, (Range&&) range);
      }

      template <__one_of<set_error_t, set_stopped_t> Tag, class... As>
      friend void tag_invoke(Tag, __t&& self, As&&... as) noexcept {
        self.op_state_.propagate_completion_signal(Tag(), (As&&) as...);
      }

      friend env_of_t<Receiver> tag_invoke(get_env_t, const __t& self) noexcept {
        return get_env(self.op_state_.rcvr_);
      }

      __t(InitT init, Fun fun, operation_state_base_t<ReceiverId>& op_state)
        : op_state_(op_state)
        , init_((InitT&&) init)
        , fun_((Fun&&) fun) {
      }
    };
  };

  template <class SenderId, class InitT, class Fun, class DerivedSender>
  struct sender_t {
    struct __t : stream_sender_base {
      using Sender = stdexec::__t<SenderId>;
      using __id = sender_t;

      template <class Receiver>
      using receiver_t = typename DerivedSender::template receiver_t<Receiver>;

      template <class Range>
      using _set_value_t = typename DerivedSender::template _set_value_t<Range>;

      Sender sndr_;
      STDEXEC_ATTRIBUTE((no_unique_address)) InitT init_;
      STDEXEC_ATTRIBUTE((no_unique_address)) Fun fun_;

      template <class Self, class Env>
      using completion_signatures = //
        __try_make_completion_signatures<
          __copy_cvref_t<Self, Sender>,
          Env,
          completion_signatures<set_error_t(cudaError_t)>,
          __q<_set_value_t >>;

      template <__decays_to<__t> Self, receiver Receiver>
        requires receiver_of<Receiver, completion_signatures<Self, env_of_t<Receiver>>>
      friend auto tag_invoke(connect_t, Self&& self, Receiver rcvr)
        -> stream_op_state_t< __copy_cvref_t<Self, Sender>, receiver_t<Receiver>, Receiver> {
        return stream_op_state<__copy_cvref_t<Self, Sender>>(
          ((Self&&) self).sndr_,
          (Receiver&&) rcvr,
          [&](operation_state_base_t<stdexec::__id<Receiver>>& stream_provider)
            -> receiver_t<Receiver> {
            return receiver_t<Receiver>(self.init_, self.fun_, stream_provider);
          });
      }

      template <__decays_to<__t> Self, class Env>
      friend auto tag_invoke(get_completion_signatures_t, Self&&, Env&&)
        -> completion_signatures<Self, Env> {
        return {};
      }

      friend auto tag_invoke(get_env_t, const __t& self) noexcept -> env_of_t<const Sender&> {
        return get_env(self.sndr_);
      }
    };
  };
}
