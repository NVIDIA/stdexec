/*
 * Copyright (c) 2023 Maikel Nadolski
 * Copyright (c) 2023 NVIDIA Corporation
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

#include "../sequence_senders.hpp"

namespace exec {
  namespace __single {
    using namespace stdexec;

    template <class _SenderId>
    struct __sender {
      using _Sender = stdexec::__t<_SenderId>;

      struct __t {
        using __id = __sender;
        using is_sender = sequence_tag;
        STDEXEC_NO_UNIQUE_ADDRESS _Sender __sndr_;

        template <class _Self, class _Receiver>
        using __next_t = __next_sender_of_t<__decay_t<_Receiver>&, __copy_cvref_t<_Self, _Sender>>;

        template <__decays_to<__t> _Self, receiver _Receiver>
          requires sequence_receiver_from<_Receiver, __copy_cvref_t<_Self, _Sender>>
                && sender_to<__next_t<_Self, _Receiver>, _Receiver>
        friend auto tag_invoke(sequence_connect_t, _Self&& __self, _Receiver&& __rcvr) noexcept(
          __nothrow_callable<set_next_t, __decay_t<_Receiver>&, __copy_cvref_t<_Self, _Sender>> //
            && __nothrow_connectable<__copy_cvref_t<_Self, _Sender>, _Receiver>)
          -> connect_result_t<__next_t<_Self, _Receiver>, _Receiver> {
          return stdexec::connect(
            exec::set_next(__rcvr, static_cast<_Self&&>(__self).__sndr_),
            static_cast<_Receiver&&>(__rcvr));
        }

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&& __self, const _Env& __env)
          -> completion_signatures_of_t<__copy_cvref_t<_Self, _Sender>, _Env>;
      };
    };

    struct single_t {
      template <class _Sender>
      using __sender_t = __t<__sender<__id<__decay_t<_Sender>>>>;

      template <class _Sender>
        requires tag_invocable<single_t, _Sender>
      auto operator()(_Sender&& __sender) const noexcept(nothrow_tag_invocable<single_t, _Sender>)
        -> tag_invoke_result_t<single_t, _Sender> {
        return tag_invoke(*this, static_cast<_Sender&&>(__sender));
      }

      template <class _Sender>
        requires(!tag_invocable<single_t, _Sender>)
      __sender_t<_Sender> operator()(_Sender&& __sender) const noexcept(__decay_copyable<_Sender>) {
        return {static_cast<_Sender&&>(__sender)};
      }

      __binder_back<single_t> operator()() const noexcept {
        return {{}, {}, {}};
      }
    };
  }

  using __single::single_t;
  inline constexpr single_t single{};
}