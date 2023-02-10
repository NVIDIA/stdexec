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

namespace exec
{
  namespace __when_any_ {
    using namespace stdexec;
    template <class _BaseEnv>
      using __env_t =
          __make_env_t<_BaseEnv, __with<get_stop_token_t, in_place_stop_token>>;

    template <class _Env, class _Sender, class... _Senders>
      make_completion_signatures<
          _Sender, _Env,
          __minvoke<__mconcat<__q<completion_signatures>>,
                    completion_signatures_of_t<_Senders, _Env>...>>
      __completion_signatures_();

    template <class _Env, class... _Senders>
      using __completion_signatures_t = decltype(__completion_signatures_<_Env, _Senders...>());

    template <class _Ret, class... _Args>
      std::tuple<_Ret, _Args...> __signature_to_tuple_(_Ret(*)(_Args...));

    template <class _Sig>
      using __signature_to_tuple =
        decltype(__signature_to_tuple_((_Sig*) nullptr));

    template <class _Sig>
    using __signature_to_tuple_t = __t<__signature_to_tuple<_Sig>>;

    template <class _Env, class... _Senders>
    using __result_type_t =
        __mapply<__transform<__q<__signature_to_tuple_t>,
                            __mbind_front_q<std::variant, std::monostate>>,
                __completion_signatures_t<_Env, _Senders...>>;

    template <class _Receiver, class _Op> struct __receiver {
      class __t {
       public:
        using __id = __receiver;

        explicit __t(_Op* __op) noexcept : __op_{__op} {}

       private:
        _Op* __op_;

        template <class _CPO, class... _Args>
          void notify(_CPO, _Args&&... __args) noexcept {
            __op_->notify(_CPO{}, ((_Args &&) __args)...);
          }

        template <__one_of<set_value_t, set_error_t, set_stopped_t> _CPO,
                  class... _Args>
        friend void tag_invoke(_CPO, __t&& __self, _Args&&... __args) noexcept {
          __self.notify(_CPO{}, (_Args &&) __args...);
        }

        friend __env_t<env_of_t<_Receiver>> tag_invoke(get_env_t, const __t& __self) noexcept {
          using __with_token = __with<get_stop_token_t, in_place_stop_token>;
          auto __token = __with_token{self.__op_->__stop_source_.get_token()};
          return __make_env(stdexec::get_env(self.__op_->__receiver_), (__with_token &&) __token);
        }
      };
    };

    template <class _Receiver, class... _Senders> struct __op_id {
      struct __t {
        __t(__t&&) = delete;

        __t(std::tuple<_Senders...>&& __senders, _Receiver&& __rcvr)
            : __t{(std::tuple<_Senders...> &&) __senders, (_Receiver &&) __rcvr,
                  std::make_index_sequence<sizeof...(_Senders)>{}} {}

        template <std::size_t... _Is>
        __t(std::tuple<_Senders...>&& __senders, _Receiver&& __rcvr,
            std::index_sequence<_Is...>)
            : __receiver_{(_Receiver &&) __rcvr}, __ops_{__conv{[&__senders, this] {
                return connect((_Senders &&)(std::get<_Is>(__senders)),
                              __receiver<_Receiver, __t>{this});
              }}...} {}

        struct __on_stop_requested {
          in_place_stop_source& __stop_source_;
          void operator()() noexcept { __stop_source_.request_stop(); }
        };

        // If this hits true, we store the reuslt
        std::atomic<bool> __emplaced_{false};
        // If this hits zero, we forward any reuslt to the receiver
        std::atomic<int> __count_{sizeof...(_Senders)};

        in_place_stop_source __stop_source_;
        using __on_stop = std::optional<typename stop_token_of_t<
            env_of_t<_Receiver>&>::template callback_type<__on_stop_requested>>;
        __on_stop __on_stop_{};
        _Receiver __receiver_;
        std::tuple<connect_result_t<_Senders, __receiver<_Receiver, __t>>...>
            __ops_;

        using __result_type = __result_type_t<env_of_t<_Receiver>, _Senders...>;
        __result_type __result_;

        template <class _CPO, class... _Args>
        void notify(_CPO, _Args&&... __args) noexcept {
          bool __expect = false;
          if (__emplaced_.compare_exchange_strong(__expect, true, std::memory_order_relaxed,
                                                  std::memory_order_relaxed)) {
            // This emplacement can happen only once
            __result_.template emplace<std::tuple<_CPO, std::decay_t<_Args>...>>(
                _CPO{}, ((_Args &&) __args)...);
            // stop pending operations
            __stop_source_.request_stop();
          }
          // make __result_ emplacement visible when __count_ goes from one to zero
          // This relies on the fact that each sender will call notify() at most once
          if (__count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            __on_stop_.reset();
            std::visit(
                [this]<class _Tuple>(_Tuple&& __result) {
                  if constexpr (std::same_as<std::decay_t<_Tuple>, std::monostate>) {
                    set_stopped((_Receiver &&) __receiver_);
                  } else {
                    auto stop_token = get_stop_token(get_env(__receiver_));
                    if (stop_token.stop_requested()) {
                      set_stopped((_Receiver &&) __receiver_);
                      return;
                    }
                    std::apply(
                        [this]<class _C, class... _As>(_C, _As&&... __args) noexcept {
                          _C{}((_Receiver &&) __receiver_, (_As &&) __args...);
                        },
                        (_Tuple &&) __result);
                  }
                },
                (__result_type &&) __result_);
          }
        }

        friend void tag_invoke(start_t, __t& __self) noexcept {
          __self.__on_stop_.emplace(get_stop_token(get_env(__self.__receiver_)),
                                    __on_stop_requested{__self.__stop_source_});
          if (__self.__stop_source_.stop_requested()) {
            stdexec::set_stopped((_Receiver &&) __self.__receiver_);
          } else {
            std::apply([](auto&... __ops) { (start(__ops), ...); }, __self.__ops_);
          }
        }
      };
    };
    template <class _Receiver, class... _Senders>
    using __op = __t<__op_id<_Receiver, _Senders...>>;

    template <class _Sender, class... _Senders> struct __sender_id {
      struct __t {
        using __id = __sender_id;

        std::tuple<_Sender, _Senders...> __senders_;

        template <class _R>
        friend __op<std::remove_cvref_t<_R>, _Sender, _Senders...> 
          tag_invoke(connect_t, __t&& __self, _R&& __rcvr) noexcept {
            return {(std::tuple<_Sender, _Senders...> &&) __self.__senders_, (_R &&) __rcvr};
          }

        friend __empty_env tag_invoke(get_env_t, const __t& __self) noexcept {
          return {};
        }

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&& __self, _Env __env) noexcept
          -> dependent_completion_signatures<_Env>;

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&& __self, _Env __env) noexcept
          -> __completion_signatures_t<_Env, _Sender, _Senders...> requires true;
      };
    };

    template <class... _Senders> using __sender = __t<__sender_id<_Senders...>>;

    struct __when_any_t {
      template <class... _Senders>
        requires ((stdexec::sender<_Senders> && ...) && sizeof...(_Senders) > 0)
      auto operator()(_Senders&&... __senders) const
          noexcept((__nothrow_decay_copyable<_Senders> && ...)) {
        return __sender<std::decay_t<_Senders>...>{
            std::tuple<std::decay_t<_Senders>...>{((_Senders &&) __senders)...}};
      }
    };

    inline constexpr __when_any_t __when_any{};
  } // namespace __when_any_
  using __when_any_::__when_any;
}