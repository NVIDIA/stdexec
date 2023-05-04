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
  namespace __transform_each {
    using namespace stdexec;

    template <class _Receiver, class _Fun>
    struct __operation_base {
      STDEXEC_NO_UNIQUE_ADDRESS _Receiver __rcvr_;
      STDEXEC_NO_UNIQUE_ADDRESS _Fun __fun_;
    };

    template <class _ReceiverId, class _Fun>
    struct __receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t {
        using __id = __receiver;
        __operation_base<_Receiver, _Fun>* __op_;

        template <class _Item>
          requires __callable<_Fun&, _Item&&>
                && __callable<set_next_t, _Receiver&, __call_result_t<_Fun&, _Item&&>>
        friend auto tag_invoke(set_next_t, __t& __self, _Item&& __item) noexcept {
          return exec::set_next(
            __self.__op_->__rcvr_, __self.__op_->__fun_(static_cast<_Item&&>(__item)));
        }

        template <__completion_tag _Tag, same_as<__t> _Self, class... _Args>
          requires __callable<_Tag, _Receiver&&, _Args&&...>
        friend void tag_invoke(_Tag __complete, _Self&& __self, _Args&&... __args) noexcept {
          __complete(
            static_cast<_Receiver&&>(__self.__op_->__rcvr_), static_cast<_Args&&>(__args)...);
        }

        friend env_of_t<_Receiver> tag_invoke(get_env_t, const __t& __self) noexcept {
          return stdexec::get_env(__self.__op_->__rcvr_);
        }
      };
    };
    template <class _Receiver, class _Fun>
    using __receiver_t = __t<__receiver<__id<__decay_t<_Receiver>>, _Fun>>;

    template <class _SenderId, class _ReceiverId, class _Fun>
    struct __operation {
      using _Sender = stdexec::__t<_SenderId>;
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t : __operation_base<_Receiver, _Fun> {
        using __id = __operation;
        sequence_connect_result_t<_Sender, __receiver_t<_Receiver, _Fun>> __op_;

        template <class _Sndr, class _Rcvr>
        __t(_Sndr&& __sndr, _Rcvr&& __rcvr, _Fun __fun)
          : __operation_base<
            _Receiver,
            _Fun>{static_cast<_Rcvr&&>(__rcvr), static_cast<_Fun&&>(__fun)}
          , __op_(
              exec::sequence_connect(static_cast<_Sndr&&>(__sndr), __receiver_t<_Receiver, _Fun>{this})) {
        }

       private:
        friend void tag_invoke(start_t, __t& __self) noexcept {
          stdexec::start(__self.__op_);
        }
      };
    };
    template <class _Sender, class _Receiver, class _Fun>
    using __operation_t = __t<__operation<__id<_Sender>, __id<__decay_t<_Receiver>>, _Fun>>;

    template <class... _Errors>
    using __error_sigs = completion_signatures<set_error_t(_Errors)...>;

    template <class _Sender, class _Env>
    using __some_sender = __copy_cvref_t<
      _Sender,
      __sequence_sndr::__unspecified_sender_of<__sequence_signatures_of_t<_Sender, _Env>>>;

    template <class _Sender, class _Env>
    using __completion_sigs =
      __if_c<
        sequence_sender_in<_Sender, _Env>,
        completion_signatures_of_t<_Sender, _Env>,
        completion_signatures<set_value_t(), set_stopped_t()>>;

    template <class _Sender, class _Env, class _Fun>
    using __sequence_sigs =
      completion_signatures_of_t<__call_result_t< _Fun, __some_sender<_Sender, _Env>>, _Env>;

    template <class _Sender, class _Fun>
    struct __sender {
      struct __t {
        using __id = __sender;
        using is_sender = sequence_tag;
        STDEXEC_NO_UNIQUE_ADDRESS _Sender __sndr_;
        STDEXEC_NO_UNIQUE_ADDRESS _Fun __fun_;

        template <__decays_to<_Sender> Sndr>
        __t(Sndr&& __sndr, _Fun __fun)
          : __sndr_{static_cast<Sndr&&>(__sndr)}
          , __fun_{static_cast<_Fun&&>(__fun)} {
        }

        template <__decays_to<__t> Self, class Rcvr>
          requires sequence_receiver_of<
                     Rcvr,
                     __sequence_sigs<__copy_cvref_t<Self, _Sender>, env_of_t<Rcvr>, _Fun>>
                && sequence_sender_to<
                     __copy_cvref_t<Self, _Sender>,
                     __receiver_t<__decay_t<Rcvr>, _Fun>>
        friend __operation_t<__copy_cvref_t<Self, _Sender>, Rcvr, _Fun>
          tag_invoke(sequence_connect_t, Self&& __self, Rcvr&& __rcvr) {
          return __operation_t<__copy_cvref_t<Self, _Sender>, Rcvr, _Fun>(
            static_cast<Self&&>(__self).__sndr_,
            static_cast<Rcvr&&>(__rcvr),
            static_cast<Self&&>(__self).__fun_);
        }

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, const _Env&)
          -> __completion_sigs<__copy_cvref_t<_Self, _Sender>, _Env>;

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_sequence_signatures_t, _Self&&, const _Env&)
          -> __sequence_sigs<__copy_cvref_t<_Self, _Sender>, _Env, _Fun>;
      };
    };

    struct transform_each_t {
      template <stdexec::sender _Sender, class _Fun>
      auto operator()(_Sender&& sndr, _Fun __fun) const {
        return __t<__sender<__decay_t<_Sender>, _Fun>>{
          static_cast<_Sender&&>(sndr), static_cast<_Fun&&>(__fun)};
      }

      template <class _Fun>
      constexpr auto operator()(_Fun __fun) const noexcept
        -> __binder_back<transform_each_t, _Fun> {
        return {{}, {}, {static_cast<_Fun&&>(__fun)}};
      }
    };
  } // namespace __transform_each

  using __transform_each::transform_each_t;
  inline constexpr transform_each_t transform_each{};

}