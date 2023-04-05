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

  namespace __ignore_all {
    using namespace stdexec;

    template <class _ItemReceiverId>
      requires receiver<stdexec::__t<_ItemReceiverId>>
    struct __item_receiver {
      using _ItemReceiver = stdexec::__t<_ItemReceiverId>;

      struct __t {
        using __id = __item_receiver;
        [[no_unique_address]] _ItemReceiver __rcvr_;

        // We eat all arguments
        template <class... _Args>
          requires __callable<set_value_t, _ItemReceiver&&>
        friend void tag_invoke(set_value_t, __t&& __self, _Args&&...) noexcept {
          stdexec::set_value(static_cast<_ItemReceiver&&>(__self.__rcvr_));
        }

        template <class _Error>
          requires __callable<set_error_t, _ItemReceiver&&, _Error&&>
        friend void tag_invoke(set_error_t, __t&& __self, _Error&& __error) noexcept {
          stdexec::set_error(
            static_cast<_ItemReceiver&&>(__self.__rcvr_), static_cast<_Error&&>(__error));
        }

        friend void tag_invoke(set_stopped_t, __t&& __self) noexcept
          requires __callable<set_stopped_t, _ItemReceiver&&>
        {
          stdexec::set_stopped(static_cast<_ItemReceiver&&>(__self.__rcvr_));
        }

        friend env_of_t<_ItemReceiver> tag_invoke(get_env_t, const __t& __self) noexcept {
          return stdexec::get_env(__self.__rcvr_);
        }
      };
    };

    template <class _ItemId>
    struct __item_sender {
      using _Item = stdexec::__t<_ItemId>;

      template <class _Rcvr>
      using __item_receiver_t = stdexec::__t<__item_receiver<__id<__decay_t<_Rcvr>>>>;

      struct __t {
        using __id = __item_sender;
        [[no_unique_address]] _Item __item;

        template <__decays_to<__t> _Self, receiver _ItemReceiver>
          requires sender_to<_Item, __item_receiver_t<_ItemReceiver>>
        friend auto tag_invoke(connect_t, _Self&& __self, _ItemReceiver&& __rcvr) {
          return stdexec::connect(
            static_cast<_Self&&>(__self).__item,
            __item_receiver_t<_ItemReceiver>{static_cast<_ItemReceiver&&>(__rcvr)});
        }

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&& __self, const _Env& __env)
          -> __make_completion_signatures<
            __copy_cvref_t<_Self, _Item>,
            _Env,
            completion_signatures<set_value_t()>,
            __mconst<completion_signatures<set_value_t()>>>;
      };
    };

    template <class _Sndr>
    using __item_sender_t = __t<__item_sender<__id<__decay_t<_Sndr>>>>;

    template <class _ReceiverId>
    struct __receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t {
        using __id = __receiver;
        [[no_unique_address]] _Receiver __rcvr_;

        template <sender _Item>
        friend __item_sender_t<_Item> tag_invoke(set_next_t, __t&, _Item&& __item) noexcept {
          // Eat all arguments
          return __item_sender_t<_Item>{static_cast<_Item&&>(__item)};
        }

        template <class _Error>
        friend void tag_invoke(set_error_t, __t&& __self, _Error&& __error) noexcept {
          set_error(static_cast<_Receiver&&>(__self.__rcvr_), static_cast<_Error&&>(__error));
        }

        template <__one_of<set_stopped_t, set_value_t> _Tag>
        friend void tag_invoke(_Tag complete, __t&& __self) noexcept {
          complete(static_cast<_Receiver&&>(__self.__rcvr_));
        }

        friend env_of_t<_Receiver> tag_invoke(get_env_t, const __t& __self) noexcept {
          return get_env(__self.__rcvr_);
        }
      };
    };

    template <class _Rcvr>
    using __receiver_t = __t<__receiver<__id<__decay_t<_Rcvr>>>>;

    template <class... _Args>
    using __drop_value_args = completion_signatures<set_value_t()>;

    template <class _Sender, class _Env>
    using __completion_sigs = make_completion_signatures<
      _Sender,
      _Env,
      completion_signatures<set_value_t()>,
      __drop_value_args>;

    template <class _SenderId>
    struct __sender {
      using _Sender = stdexec::__t<__decay_t<_SenderId>>;

      struct __t {
        using __id = __sender;
        [[no_unique_address]] _Sender __sndr_;

        template <__decays_to<__t> _Self, class _Receiver>
          requires receiver_of<
                     _Receiver,
                     __completion_sigs<__copy_cvref_t<_Self, _Sender>, env_of_t<_Receiver>>>
                && sequence_sender_to<__copy_cvref_t<_Self, _Sender>, __receiver_t<_Receiver>>
        friend sequence_connect_result_t< __copy_cvref_t<_Self, _Sender>, __receiver_t<_Receiver>>
          tag_invoke(connect_t, _Self&& __self, _Receiver&& __rcvr) {
          return sequence_connect(
            ((_Self&&) __self).__sndr_, __receiver_t<_Receiver>{static_cast<_Receiver&&>(__rcvr)});
        }

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, const _Env&)
          -> __completion_sigs<__copy_cvref_t<_Self, _Sender>, _Env>;
      };
    };

    struct ignore_all_t {
      template <class _Sender>
      constexpr auto operator()(_Sender&& __sndr) const {
        return __t<__sender<__id<__decay_t<_Sender>>>>{static_cast<_Sender&&>(__sndr)};
      }

      constexpr auto operator()() const noexcept -> __binder_back<ignore_all_t> {
        return {};
      }
    };
  } // namespace __ignore_all

  using __ignore_all::ignore_all_t;
  inline constexpr ignore_all_t ignore_all;

}