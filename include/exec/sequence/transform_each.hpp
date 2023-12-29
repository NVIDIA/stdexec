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

#include "../__detail/__basic_sequence.hpp"

namespace exec {
  namespace __transform_each {
    using namespace stdexec;

    template <class _Receiver, class _Adaptor>
    struct __operation_base {
      _Receiver __receiver_;
      _Adaptor __adaptor_;
    };

    template <class _ReceiverId, class _Adaptor>
    struct __receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t {
        using receiver_concept = stdexec::receiver_t;
        using __id = __receiver;
        __operation_base<_Receiver, _Adaptor>* __op_;

        template <same_as<set_next_t> _SetNext, same_as<__t> _Self, class _Item>
          requires __callable<_Adaptor&, _Item>
                && __callable<exec::set_next_t, _Receiver&, __call_result_t<_Adaptor&, _Item>>
        friend auto tag_invoke(_SetNext, _Self& __self, _Item&& __item) noexcept(
          __nothrow_callable<_SetNext, _Receiver&, __call_result_t<_Adaptor&, _Item>> //
            && __nothrow_callable<_Adaptor&, _Item>)
          -> next_sender_of_t<_Receiver, __call_result_t<_Adaptor&, _Item>> {
          return exec::set_next(
            __self.__op_->__receiver_, __self.__op_->__adaptor_(static_cast<_Item&&>(__item)));
        }

        template <same_as<set_value_t> _SetValue, same_as<__t> _Self>
        friend void tag_invoke(_SetValue, _Self&& __self) noexcept {
          stdexec::set_value(static_cast<_Receiver&&>(__self.__op_->__receiver_));
        }

        template <same_as<set_stopped_t> _SetStopped, same_as<__t> _Self>
          requires __callable<_SetStopped, _Receiver&&>
        friend void tag_invoke(_SetStopped, _Self&& __self) noexcept {
          stdexec::set_stopped(static_cast<_Receiver&&>(__self.__op_->__receiver_));
        }

        template <same_as<set_error_t> _SetError, same_as<__t> _Self, class _Error>
          requires __callable<_SetError, _Receiver&&, _Error>
        friend void tag_invoke(_SetError, _Self&& __self, _Error&& __error) noexcept {
          stdexec::set_error(
            static_cast<_Receiver&&>(__self.__op_->__receiver_), static_cast<_Error&&>(__error));
        }

        template <same_as<get_env_t> _GetEnv, __decays_to<__t> _Self>
        friend env_of_t<_Receiver> tag_invoke(_GetEnv, _Self&& __self) noexcept {
          return stdexec::get_env(__self.__op_->__receiver_);
        }
      };
    };

    template <class _Sender, class _ReceiverId, class _Adaptor>
    struct __operation {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t : __operation_base<_Receiver, _Adaptor> {
        using __id = __operation;
        subscribe_result_t<_Sender, stdexec::__t<__receiver<_ReceiverId, _Adaptor>>> __op_;

        __t(_Sender&& __sndr, _Receiver __rcvr, _Adaptor __adaptor)
          : __operation_base<
            _Receiver,
            _Adaptor>{static_cast<_Receiver&&>(__rcvr), static_cast<_Adaptor&&>(__adaptor)}
          , __op_{exec::subscribe(
              static_cast<_Sender&&>(__sndr),
              stdexec::__t<__receiver<_ReceiverId, _Adaptor>>{this})} {
        }

        friend void tag_invoke(start_t, __t& __self) noexcept {
          stdexec::start(__self.__op_);
        }
      };
    };

    template <class _Receiver>
    struct __subscribe_fn {
      _Receiver& __rcvr_;

      template <class _Adaptor, class _Sequence>
      auto operator()(__ignore, _Adaptor __adaptor, _Sequence&& __sequence) noexcept(
        __nothrow_decay_copyable<_Adaptor>&& __nothrow_decay_copyable<_Sequence>&&
          __nothrow_decay_copyable<_Receiver>)
        -> __t< __operation<_Sequence, __id<_Receiver>, _Adaptor>> {
        return {
          static_cast<_Sequence&&>(__sequence),
          static_cast<_Receiver&&>(__rcvr_),
          static_cast<_Adaptor&&>(__adaptor)};
      }
    };

    template <class _Adaptor>
    struct _NOT_CALLABLE_ADAPTOR_ { };

    template <class _Item>
    struct _WITH_ITEM_SENDER_ { };

    template <class _Adaptor, class _Item>
    auto __try_call(_Item*) -> stdexec::__mexception<
      _NOT_CALLABLE_ADAPTOR_<_Adaptor&>,
      _WITH_ITEM_SENDER_<stdexec::__name_of<_Item>>>;

    template <class _Adaptor, class _Item>
      requires stdexec::__callable<_Adaptor&, _Item>
    stdexec::__msuccess __try_call(_Item*);

    template <class _Adaptor, class... _Items>
    auto __try_calls(item_types<_Items...>*)
      -> decltype((stdexec::__msuccess() && ... && __try_call<_Adaptor>((_Items*) nullptr)));

    template <class _Adaptor, class _Items>
    concept __callabale_adaptor_for = requires(_Items* __items) {
      { __try_calls<stdexec::__decay_t<_Adaptor>>(__items) } -> stdexec::__ok;
    };

    struct transform_each_t {
      template <sender _Sequence, __sender_adaptor_closure _Adaptor>
      auto operator()(_Sequence&& __sndr, _Adaptor&& __adaptor) const
        noexcept(__nothrow_decay_copyable<_Sequence> //
                   && __nothrow_decay_copyable<_Adaptor>) {
        return make_sequence_expr<transform_each_t>(
          static_cast<_Adaptor&&>(__adaptor), static_cast<_Sequence&&>(__sndr));
      }

      template <class _Adaptor>
      constexpr auto operator()(_Adaptor __adaptor) const noexcept
        -> __binder_back<transform_each_t, _Adaptor> {
        return {{}, {}, {static_cast<_Adaptor&&>(__adaptor)}};
      }

      template <class _Self, class _Env>
      using __completion_sigs_t = __sequence_completion_signatures_of_t<__child_of<_Self>, _Env>;

      template <sender_expr_for<transform_each_t> _Self, class _Env>
      static __completion_sigs_t<_Self, _Env> get_completion_signatures(_Self&&, _Env&&) noexcept {
        return {};
      }

      template <class _Self, class _Env>
      using __item_types_t = stdexec::__mapply<
        stdexec::__transform<
          stdexec::__mbind_front_q<__call_result_t, __data_of<_Self>&>,
          stdexec::__munique<stdexec::__q<item_types>>>,
        item_types_of_t<__child_of<_Self>, _Env>>;

      template <sender_expr_for<transform_each_t> _Self, class _Env>
      static __item_types_t<_Self, _Env> get_item_types(_Self&&, _Env&&) noexcept {
        return {};
      }

      template <class _Self, class _Receiver>
      using __receiver_t = __t<__receiver<__id<_Receiver>, __data_of<_Self>>>;

      template <class _Self, class _Receiver>
      using __operation_t = __t< __operation<__child_of<_Self>, __id<_Receiver>, __data_of<_Self>>>;

      template <sender_expr_for<transform_each_t> _Self, receiver _Receiver>
        requires __callabale_adaptor_for<
                   __data_of<_Self>,
                   __item_types_t<_Self, env_of_t<_Receiver>>>
              && sequence_receiver_of<_Receiver, __item_types_t<_Self, env_of_t<_Receiver>>>
              && sequence_sender_to<__child_of<_Self>, __receiver_t<_Self, _Receiver>>
      static auto subscribe(_Self&& __self, _Receiver __rcvr) noexcept(
        __nothrow_callable<__sexpr_apply_t, _Self, __subscribe_fn<_Receiver>>)
        -> __call_result_t<__sexpr_apply_t, _Self, __subscribe_fn<_Receiver>> {
        return __sexpr_apply(static_cast<_Self&&>(__self), __subscribe_fn<_Receiver>{__rcvr});
      }

      template <sender_expr_for<transform_each_t> _Sexpr>
      static env_of_t<__child_of<_Sexpr>> get_env(const _Sexpr& __sexpr) noexcept {
        return __sexpr_apply(__sexpr, []<class _Child>(__ignore, __ignore, const _Child& __child) {
          return stdexec::get_env(__child);
        });
      }
    };
  }

  using __transform_each::transform_each_t;
  inline constexpr transform_each_t transform_each{};
}