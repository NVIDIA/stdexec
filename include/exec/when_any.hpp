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

#include <stdexec/execution.hpp>

namespace exec {
  namespace __when_any {
    using namespace stdexec;

    struct __on_stop_requested {
      in_place_stop_source& __stop_source_;

      void operator()() noexcept {
        __stop_source_.request_stop();
      }
    };

    template <class _BaseEnv>
    using __env_t = __make_env_t<_BaseEnv, __with<get_stop_token_t, in_place_stop_token>>;

    template <class _Ret, class... _Args>
    __decayed_tuple<_Ret, _Args...> __signature_to_tuple_(_Ret (*)(_Args...));

    template <class _Sig>
    using __signature_to_tuple_t = decltype(__signature_to_tuple_((_Sig*) nullptr));

    template <class... _Args>
    using __all_nothrow_decay_copyable = __mbool<(__nothrow_decay_copyable<_Args> && ...)>;

    template <class... _Args>
    using __all_nothrow_move_constructible =
      __minvoke<__mall_of<__q<std::is_nothrow_move_constructible>>, _Args...>;

    template <class _Env, class... _SenderIds>
    using __all_value_args_nothrow_decay_copyable = __mand<
      __mand< value_types_of_t<__t<_SenderIds>, _Env, __all_nothrow_decay_copyable, __mand>...>,
      __mand<value_types_of_t<
        __t<_SenderIds>,
        _Env,
        __decayed_tuple, // This tests only decayed Args which is ok because moving tags is noexcept
        __all_nothrow_move_constructible>...>>;

    template <class... Args>
    using __as_rvalues = set_value_t(__decay_t<Args>&&...);

    template <class... E>
    using __as_error = completion_signatures<set_error_t(E)...>;

    // Here we convert all set_value(Args...) to set_value(__decay_t<Args>&&...)
    // Note, we keep all error types as they are and unconditionally add set_stopped()
    template <class _Env, class... _SenderIds>
    using __completion_signatures_t = __concat_completion_signatures_t<
      __if<
        __all_value_args_nothrow_decay_copyable<_Env, _SenderIds...>,
        completion_signatures<set_stopped_t()>,
        completion_signatures<set_stopped_t(), set_error_t(std::exception_ptr)>>,
      value_types_of_t<__t<_SenderIds>, _Env, __as_rvalues, completion_signatures>...,
      error_types_of_t<__t<_SenderIds>, _Env, __as_error>...>;

    // transform Tag(Args...) to a tuple __decayed_tuple<Tag, Args...>
    template <class _Env, class... _SenderIds>
    using __result_type_t = __mapply<
      __transform<__q<__signature_to_tuple_t>, __munique<__q<std::variant>>>,
      __completion_signatures_t<_Env, _SenderIds...>>;

    template <class _Variant, class... _Ts>
    concept __result_constructible_from =
      std::is_constructible_v<__decayed_tuple<_Ts...>, _Ts...>
      && std::is_constructible_v<_Variant, __decayed_tuple<_Ts...>&&>;

    template <class _Variant, class... _Ts>
    concept __nothrow_result_constructible_from =
      __nothrow_constructible_from<__decayed_tuple<_Ts...>, _Ts...>
      && __nothrow_constructible_from<_Variant, __decayed_tuple<_Ts...>&&>;

    template <class _Receiver, class _ResultVariant>
    struct __op_base : __immovable {
      __op_base(_Receiver&& __receiver, int __n_senders)
        : __count_{__n_senders}
        , __receiver_{(_Receiver&&) __receiver} {
      }

      using __on_stop = //
        std::optional<typename stop_token_of_t< env_of_t<_Receiver>&>::template callback_type<
          __on_stop_requested>>;

      in_place_stop_source __stop_source_{};
      __on_stop __on_stop_{};

      // If this hits true, we store the result
      std::atomic<bool> __emplaced_{false};
      // If this hits zero, we forward any result to the receiver
      std::atomic<int> __count_{};

      _Receiver __receiver_;
      std::optional<_ResultVariant> __result_{};

      template <class _CPO, class... _Args>
      void notify(_CPO, _Args&&... __args) noexcept {
        bool __expect = false;
        if (__emplaced_.compare_exchange_strong(
              __expect, true, std::memory_order_relaxed, std::memory_order_relaxed)) {
          // This emplacement can happen only once
          if constexpr (__nothrow_result_constructible_from<_ResultVariant, _CPO, _Args...>) {
            __result_.emplace(std::tuple{_CPO{}, (_Args&&) __args...});
          } else {
            try {
              __result_.emplace(std::tuple{_CPO{}, (_Args&&) __args...});
            } catch (...) {
              __result_.emplace(std::tuple{set_error_t{}, std::current_exception()});
            }
          }
          // stop pending operations
          __stop_source_.request_stop();
        }
        // make __result_ emplacement visible when __count_ goes from one to zero
        // This relies on the fact that each sender will call notify() at most once
        if (__count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
          __on_stop_.reset();
          auto stop_token = get_stop_token(get_env(__receiver_));
          if (stop_token.stop_requested()) {
            set_stopped((_Receiver&&) __receiver_);
            return;
          }
          STDEXEC_ASSERT(__result_.has_value());
          std::visit(
            [this]<class _Tuple>(_Tuple&& __result) {
              std::apply(
                [this]<class _Cpo, class... _As>(_Cpo, _As&&... __args) noexcept {
                  _Cpo{}((_Receiver&&) __receiver_, (_As&&) __args...);
                },
                (_Tuple&&) __result);
            },
            (_ResultVariant&&) *__result_);
        }
      }
    };

    template <class _Receiver, class _ResultVariant>
    struct __receiver {
      class __t {
       public:
        using receiver_concept = stdexec::receiver_t;
        using __id = __receiver;

        explicit __t(__op_base<_Receiver, _ResultVariant>* __op) noexcept
          : __op_{__op} {
        }

       private:
        __op_base<_Receiver, _ResultVariant>* __op_;

        template <__completion_tag _CPO, class... _Args>
          requires __result_constructible_from<_ResultVariant, _CPO, _Args...>
        friend void tag_invoke(_CPO, __t&& __self, _Args&&... __args) noexcept {
          __self.__op_->notify(_CPO{}, (_Args&&) __args...);
        }

        friend __env_t<env_of_t<_Receiver>> tag_invoke(get_env_t, const __t& __self) noexcept {
          auto __token = __mkprop(__self.__op_->__stop_source_.get_token(), get_stop_token);
          return __make_env(get_env(__self.__op_->__receiver_), std::move(__token));
        }
      };
    };

    template <class _ReceiverId, class... _SenderIds>
    struct __op {
      using _Receiver = stdexec::__t<_ReceiverId>;

      using __result_t = __result_type_t<env_of_t<_Receiver>, _SenderIds...>;
      using __receiver_t = stdexec::__t<__receiver<_Receiver, __result_t>>;
      using __op_base_t = __op_base<_Receiver, __result_t>;

      class __t : __op_base_t {
       public:
        template <class _SenderTuple>
        __t(_SenderTuple&& __senders, _Receiver&& __rcvr) //
          noexcept(
            __nothrow_decay_copyable<_Receiver>
            && (__nothrow_connectable<stdexec::__t<_SenderIds>, __receiver_t> && ...))
          : __t{
            (_SenderTuple&&) __senders,
            (_Receiver&&) __rcvr,
            std::index_sequence_for<_SenderIds...>{}} {
        }

       private:
        template <class _SenderTuple, std::size_t... _Is>
        __t(_SenderTuple&& __senders, _Receiver&& __rcvr, std::index_sequence<_Is...>) //
          noexcept(
            __nothrow_decay_copyable<_Receiver>
            && (__nothrow_connectable<stdexec::__t<_SenderIds>, __receiver_t> && ...))
          : __op_base_t{(_Receiver&&) __rcvr, static_cast<int>(sizeof...(_SenderIds))}
          , __ops_{__conv{[&__senders, this] {
            return stdexec::connect(
              std::get<_Is>((_SenderTuple&&) __senders),
              __receiver_t{static_cast<__op_base_t*>(this)});
          }}...} {
        }

        std::tuple<connect_result_t<stdexec::__t<_SenderIds>, __receiver_t>...> __ops_;

        friend void tag_invoke(start_t, __t& __self) noexcept {
          __self.__on_stop_.emplace(
            get_stop_token(get_env(__self.__receiver_)),
            __on_stop_requested{__self.__stop_source_});
          if (__self.__stop_source_.stop_requested()) {
            set_stopped((_Receiver&&) __self.__receiver_);
          } else {
            std::apply([](auto&... __ops) { (start(__ops), ...); }, __self.__ops_);
          }
        }
      };
    };

    template <class... _SenderIds>
    struct __sender {

      template <class _Receiver>
      using __receiver_t =
        stdexec::__t< __receiver<_Receiver, __result_type_t<env_of_t<_Receiver>, _SenderIds...>>>;

      template <class _Receiver>
      using __op_t = stdexec::__t<__op<__id<__decay_t<_Receiver>>, _SenderIds...>>;

      class __t {
       public:
        using __id = __sender;
        using sender_concept = stdexec::sender_t;

        template <class... _Senders>
        explicit(sizeof...(_Senders) == 1)
          __t(_Senders&&... __senders) noexcept((__nothrow_decay_copyable<_Senders> && ...))
          : __senders_((_Senders&&) __senders...) {
        }

       private:
        template <__decays_to<__t> _Self, receiver _Receiver>
          requires(
            sender_to< __copy_cvref_t<_Self, stdexec::__t<_SenderIds>>, __receiver_t<_Receiver>>
            && ...)
        friend __op_t<_Receiver> tag_invoke(connect_t, _Self&& __self, _Receiver&& __rcvr) //
          noexcept(__nothrow_constructible_from<__op_t<_Receiver>, _Self&&, _Receiver&&>) {
          return __op_t<_Receiver>{((_Self&&) __self).__senders_, (_Receiver&&) __rcvr};
        }

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&& __self, _Env __env) noexcept
          -> __completion_signatures_t<_Env, _SenderIds...> {
          return {};
        }

        std::tuple<stdexec::__t<_SenderIds>...> __senders_;
      };
    };

    struct __when_any_t {
      template <class... _Senders>
      using __sender_t = __t<__sender<__id<__decay_t<_Senders>>...>>;

      template <sender... _Senders>
        requires(sizeof...(_Senders) > 0 && sender<__sender_t<_Senders...>>)
      __sender_t<_Senders...> operator()(_Senders&&... __senders) const
        noexcept((__nothrow_decay_copyable<_Senders> && ...)) {
        return __sender_t<_Senders...>((_Senders&&) __senders...);
      }
    };

    inline constexpr __when_any_t when_any{};
  } // namespace __when_any

  using __when_any::when_any;
}
