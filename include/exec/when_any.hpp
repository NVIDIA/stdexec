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
    __decayed_tuple<_Args...> __signature_to_tuple_(_Ret (*)(_Args...));

    template <class _Sig>
    using __signature_to_tuple_t = decltype(__signature_to_tuple_((_Sig*) nullptr));

    template <class _Tag>
    struct __ret_equals_to {
      template <class _Sig>
      using __f = std::is_same<_Tag, __tag_of_sig_t<_Sig>>;
    };

    template <class... _Args>
    using __all_nothrow_decay_copyable = __bool<(__nothrow_decay_copyable<_Args> && ...)>;

    template <class _Env, class... _SenderIds>
    using __all_value_args_nothrow_decay_copyable =
      __mand<value_types_of_t<__t<_SenderIds>, _Env, __all_nothrow_decay_copyable, __mand>...>;

    template <class... Args>
    using __as_rvalues = set_value_t(decay_t<Args>&&...);

    template <class... E>
    using __as_error = completion_signatures<set_error_t(E)...>;

    template <class _Env, class... _SenderIds>
    using __completion_signatures_t = __concat_completion_signatures_t<
        __if<
        __all_value_args_nothrow_decay_copyable<_Env, _SenderIds...>,
        completion_signatures<set_stopped_t()>,
        completion_signatures<set_stopped_t(), set_error_t(std::exception_ptr)>>,
      value_types_of_t<__t<_SenderIds>, _Env, __as_rvalues, completion_signatures>...,
      error_types_of_t<__t<_SenderIds>, _Env, __as_error>...>;

    template <class _Env, class... _SenderIds>
    using __result_type_t_ = __mapply<
        __transform<__q<__signature_to_tuple_t>, __q<std::variant>>,
      __mapply<
        __remove_if<__mnone_of<__ret_equals_to<set_value_t>>, __q<completion_signatures>>,
        __completion_signatures_t<_Env, _SenderIds...>>>;

    struct __exception_tag { };

    template <class _Env, class... _SenderIds>
    using __result_type_t = __if<
      __all_value_args_nothrow_decay_copyable<_Env, _SenderIds...>,
      __result_type_t_<_Env, _SenderIds...>,
      __minvoke<
        __push_back<__q<std::variant>>,
        __result_type_t_<_Env, _SenderIds...>,
        std::tuple<__exception_tag, std::exception_ptr>>>;

    template <class _Variant, class... _Ts>
    concept __result_constructible_from =
      std::is_constructible_v<__decayed_tuple<_Ts...>, _Ts...>
      && std::is_constructible_v<_Variant, __decayed_tuple<_Ts...>&&>;

    template <class _Variant, class... _Ts>
    concept __nothrow_result_constructible_from =
      std::is_nothrow_constructible_v<__decayed_tuple<_Ts...>, _Ts...>
      && std::is_nothrow_constructible_v<_Variant, __decayed_tuple<_Ts...>&&>;

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

      template <__one_of<set_value_t, set_error_t, set_stopped_t> _CPO, class... _Args>
      void notify(_CPO, _Args&&... __args) noexcept {
        // make __result_ emplacement visible when __count_ goes from one to zero
        // This relies on the fact that each sender will call notify() at most once
        if (__count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
          __on_stop_.reset();
          auto stop_token = get_stop_token(get_env(__receiver_));
          if (stop_token.stop_requested()) {
            set_stopped((_Receiver&&) __receiver_);
          } else if (!__result_.has_value()) {
            _CPO{}((_Receiver&&) __receiver_, (_Args&&) __args...);
          } else {
          std::visit(
            [this]<class _Tuple>(_Tuple&& __result) {
              std::apply(
                  [this]<class... _As>(_As&&... __args) noexcept {
                    if constexpr (sizeof...(_As) >= 1) {
                      if constexpr (__decays_to<__mfront<_As...>, __exception_tag>) {
                        [this]<class _B, class... _Bs>(_B&&, _Bs&&... __bs) {
                          set_error((_Receiver&&) __receiver_, (_Bs&&) __bs...);
                        }((_As&&) __args...);
                      } else {
                        set_value((_Receiver&&) __receiver_, (_As&&) __args...);
                      }
                    } else {
                      set_value((_Receiver&&) __receiver_, (_As&&) __args...);
                    }
                },
                (_Tuple&&) __result);
            },
            (_ResultVariant&&) *__result_);
          }
        }
      }

      template <class... _Args>
      void emplace_and_notify(_Args&&... __args) noexcept {
        bool __expect = false;
        if (__emplaced_.compare_exchange_strong(
              __expect, true, std::memory_order_relaxed, std::memory_order_relaxed)) {
          // This emplacement can happen only once
          if constexpr (__nothrow_result_constructible_from<_ResultVariant, _Args...>) {
            __result_.emplace(std::tuple{(_Args&&) __args...});
          } else {
            try {
              __result_.emplace(std::tuple{(_Args&&) __args...});
            } catch (...) {
              __result_.emplace(std::tuple{__exception_tag{}, std::current_exception()});
            }
          }
          // stop pending operations
          __stop_source_.request_stop();
        }
        notify(set_value, (_Args&&) __args...);
      }
    };

    template <class _Receiver, class _ResultVariant>
    struct __receiver {
      class __t {
       public:
        using __id = __receiver;

        explicit __t(__op_base<_Receiver, _ResultVariant>* __op) noexcept
          : __op_{__op} {
        }

       private:
        __op_base<_Receiver, _ResultVariant>* __op_;

        template <__one_of<set_error_t, set_stopped_t> _CPO, class... _Args>
        friend void tag_invoke(_CPO, __t&& __self, _Args&&... __args) noexcept {
          __self.__op_->notify(_CPO{}, (_Args&&) __args...);
        }

        template <class... _Args>
          requires __result_constructible_from<_ResultVariant, _Args...>
        friend void tag_invoke(set_value_t, __t&& __self, _Args&&... __args) noexcept {
          __self.__op_->emplace_and_notify((_Args&&) __args...);
        }

        friend __env_t<env_of_t<_Receiver>> tag_invoke(get_env_t, const __t& __self) noexcept {
          using __with_token = __with<get_stop_token_t, in_place_stop_token>;
          auto __token = __with_token{__self.__op_->__stop_source_.get_token()};
          return __make_env(get_env(__self.__op_->__receiver_), (__with_token&&) __token);
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
            return connect(
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
      using __op_t = stdexec::__t<__op<__id<decay_t<_Receiver>>, _SenderIds...>>;

      class __t {
       public:
        using __id = __sender;
        using is_sender = void;

        template <class... _Senders>
        explicit(sizeof...(_Senders) == 1)
          __t(_Senders&&... __senders) noexcept((__nothrow_decay_copyable<_Senders> && ...))
          : __senders_((_Senders&&) __senders...) {
        }

       private:
        using senders_tuple_t = std::tuple<stdexec::__t<_SenderIds>...>;

        template <__decays_to<__t> _Self, receiver _Receiver>
          requires(
            sender_to<__copy_cvref_t<_Self, stdexec::__t<_SenderIds>>, __receiver_t<_Receiver>>
            && ...)
        friend __op_t<_Receiver> tag_invoke(connect_t, _Self&& __self, _Receiver&& __rcvr) //
          noexcept(std::is_nothrow_constructible_v<
                   __op_t<_Receiver>,
                   __copy_cvref_t<_Self, senders_tuple_t>,
                   _Receiver&&>) {
          return __op_t<_Receiver>{((_Self&&) __self).__senders_, (_Receiver&&) __rcvr};
        }

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&& __self, _Env __env) noexcept
          -> dependent_completion_signatures<_Env>;

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&& __self, _Env __env) noexcept
          -> __completion_signatures_t<_Env, _SenderIds...>
          requires true;

        std::tuple<stdexec::__t<_SenderIds>...> __senders_;
      };
    };

    struct __when_any_value_t {
      template <class... _Senders>
      using __sender_t = __t<__sender<__id<decay_t<_Senders>>...>>;

      template <sender... _Senders>
        requires(sizeof...(_Senders) > 0 && sender<__sender_t<_Senders...>>)
      __sender_t<_Senders...> operator()(_Senders&&... __senders) const
        noexcept((__nothrow_decay_copyable<_Senders> && ...)) {
        return __sender_t<_Senders...>((_Senders&&) __senders...);
      }
    };

    inline constexpr __when_any_value_t when_any_value{};
  } // namespace __when_any

  using __when_any::when_any_value;
}
