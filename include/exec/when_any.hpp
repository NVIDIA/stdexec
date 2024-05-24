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

#include "../stdexec/concepts.hpp"
#include "../stdexec/execution.hpp"
#include "../stdexec/stop_token.hpp"

#include <type_traits>
#include <exception>

namespace exec {
  namespace __when_any {
    using namespace stdexec;

    struct __on_stop_requested {
      inplace_stop_source& __stop_source_;

      void operator()() noexcept {
        __stop_source_.request_stop();
      }
    };

    template <class _BaseEnv>
    using __env_t = __env::__join_t<__env::__with<inplace_stop_token, get_stop_token_t>, _BaseEnv>;

    template <class... _Ts>
    using __nothrow_decay_copyable_and_move_constructible_t = __mbool<(
      (__nothrow_decay_copyable<_Ts> && __nothrow_move_constructible<__decay_t<_Ts>>) &&...)>;

    template <class _Env, class... _CvrefSenders>
    using __all_value_args_nothrow_decay_copyable = //
      __mand_t<value_types_of_t<
        _CvrefSenders,
        _Env,
        __nothrow_decay_copyable_and_move_constructible_t,
        __mand_t>...>;

    template <class... Args>
    using __as_rvalues = set_value_t (*)(__decay_t<Args>...);

    template <class... E>
    using __as_error = set_error_t (*)(E...);

    template <class... _CvrefSenders, class _Env>
    auto __completions_fn(_Env&&) //
      -> __concat_completion_signatures<
        __eptr_completion_if_t<__all_value_args_nothrow_decay_copyable<_Env, _CvrefSenders...>>,
        completion_signatures<set_stopped_t()>,
        __transform_completion_signatures<
          __completion_signatures_of_t<_CvrefSenders, _Env>,
          __as_rvalues,
          __as_error,
          set_stopped_t (*)(),
          __completion_signature_ptrs>...>;

    // Here we convert all set_value(Args...) to set_value(__decay_t<Args>...). Note, we keep all
    // error types as they are and unconditionally add set_stopped(). The indirection through the
    // __completions_fn is to avoid a pack expansion bug in nvc++.
    template <class _Env, class... _CvrefSenders>
    using __completions_t = //
      decltype(__completions_fn<_CvrefSenders...>(__declval<_Env>()));

    template <class _Env, class... _CvrefSenders>
    using __result_type_t = //
      __for_each_completion_signature<
        __completions_t<_Env, _CvrefSenders...>,
        __decayed_tuple,
        __munique<__q<std::variant>>::__f>;

    template <class _Variant, class... _Ts>
    concept __nothrow_result_constructible_from =
      __nothrow_constructible_from<__decayed_tuple<_Ts...>, _Ts...>
      && __nothrow_constructible_from<_Variant, __decayed_tuple<_Ts...>>;

    template <class _Receiver>
    auto __make_visitor_fn(_Receiver& __rcvr) noexcept {
      return [&__rcvr]<class _Tuple>(_Tuple&& __result) noexcept {
        std::apply(
          [&__rcvr]<class _Tag, class... _As>(_Tag, _As&&... __args) noexcept {
            _Tag{}(static_cast<_Receiver&&>(__rcvr), static_cast<_As&&>(__args)...);
          },
          static_cast<_Tuple&&>(__result));
      };
    }

    template <class _Receiver, class _ResultVariant>
    struct __op_base : __immovable {
      __op_base(_Receiver&& __rcvr, std::size_t __n_senders)
        : __count_{__n_senders}
        , __rcvr_{static_cast<_Receiver&&>(__rcvr)} {
      }

      using __on_stop =
        stop_callback_for_t<stop_token_of_t<env_of_t<_Receiver>&>, __on_stop_requested>;

      inplace_stop_source __stop_source_{};
      std::optional<__on_stop> __on_stop_{};

      // If this hits true, we store the result
      std::atomic<bool> __emplaced_{false};
      // If this hits zero, we forward any result to the receiver
      std::atomic<std::size_t> __count_{};

      _Receiver __rcvr_;
      std::optional<_ResultVariant> __result_{};

      template <class _Tag, class... _Args>
      void notify(_Tag, _Args&&... __args) noexcept {
        bool __expect = false;
        if (__emplaced_.compare_exchange_strong(
              __expect, true, std::memory_order_relaxed, std::memory_order_relaxed)) {
          // This emplacement can happen only once
          if constexpr (__nothrow_result_constructible_from<_ResultVariant, _Tag, _Args...>) {
            __result_.emplace(std::tuple{_Tag{}, static_cast<_Args&&>(__args)...});
          } else {
            try {
              __result_.emplace(std::tuple{_Tag{}, static_cast<_Args&&>(__args)...});
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
          auto stop_token = get_stop_token(get_env(__rcvr_));
          if (stop_token.stop_requested()) {
            stdexec::set_stopped(static_cast<_Receiver&&>(__rcvr_));
            return;
          }
          STDEXEC_ASSERT(__result_.has_value());
          std::visit(
            __when_any::__make_visitor_fn(__rcvr_), static_cast<_ResultVariant&&>(*__result_));
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

        auto get_env() const noexcept -> __env_t<env_of_t<_Receiver>> {
          auto __token = __env::__with(__op_->__stop_source_.get_token(), get_stop_token);
          return __env::__join(std::move(__token), stdexec::get_env(__op_->__rcvr_));
        }

        template <class... _Args>
        void set_value(_Args&&... __args) noexcept {
          __op_->notify(set_value_t(), static_cast<_Args&&>(__args)...);
        }

        template <class _Error>
        void set_error(_Error&& __err) noexcept {
          __op_->notify(set_error_t(), static_cast<_Error&&>(__err));
        }

        void set_stopped() noexcept {
          __op_->notify(set_stopped_t());
        }

       private:
        __op_base<_Receiver, _ResultVariant>* __op_;
      };
    };

    template <class _ReceiverId, class... _CvrefSenderIds>
    struct __op {
      using _Receiver = stdexec::__t<_ReceiverId>;

      using __result_t = __result_type_t<env_of_t<_Receiver>, __cvref_t<_CvrefSenderIds>...>;
      using __receiver_t = stdexec::__t<__receiver<_Receiver, __result_t>>;
      using __op_base_t = __op_base<_Receiver, __result_t>;

      static constexpr bool __nothrow_construct = //
        __nothrow_decay_copyable<_Receiver>
        && (__nothrow_connectable<__cvref_t<_CvrefSenderIds>, __receiver_t> && ...);

      class __t : __op_base_t {
       public:
        template <class _SenderTuple>
        __t(_SenderTuple&& __senders, _Receiver&& __rcvr) noexcept(__nothrow_construct)
          : __t{
            static_cast<_SenderTuple&&>(__senders),
            static_cast<_Receiver&&>(__rcvr),
            std::index_sequence_for<_CvrefSenderIds...>{}} {
        }

        void start() & noexcept {
          this->__on_stop_.emplace(
            get_stop_token(get_env(this->__rcvr_)), __on_stop_requested{this->__stop_source_});
          if (this->__stop_source_.stop_requested()) {
            stdexec::set_stopped(static_cast<_Receiver&&>(this->__rcvr_));
          } else {
            std::apply([](auto&... __ops) { (stdexec::start(__ops), ...); }, __ops_);
          }
        }

       private:
        template <class _SenderTuple, std::size_t... _Is>
        __t(_SenderTuple&& __senders, _Receiver&& __rcvr, std::index_sequence<_Is...>) //
          noexcept(__nothrow_construct)
          : __op_base_t{static_cast<_Receiver&&>(__rcvr), sizeof...(_CvrefSenderIds)}
          , __ops_{__conv{[&__senders, this] {
            return stdexec::connect(
              std::get<_Is>(static_cast<_SenderTuple&&>(__senders)),
              __receiver_t{static_cast<__op_base_t*>(this)});
          }}...} {
        }

        std::tuple<connect_result_t<stdexec::__cvref_t<_CvrefSenderIds>, __receiver_t>...> __ops_;
      };
    };

    template <class... _SenderIds>
    struct __sender {
      template <class _Self, class _Env>
      using __result_t = __result_type_t<_Env, __copy_cvref_t<_Self, stdexec::__t<_SenderIds>>...>;

      template <class _Self, class _Receiver>
      using __receiver_t =
        stdexec::__t<__receiver<_Receiver, __result_t<_Self, env_of_t<_Receiver>>>>;

      template <class _Self, class _Receiver>
      using __op_t = stdexec::__t<__op<__id<_Receiver>, __copy_cvref_t<_Self, _SenderIds>...>>;

      class __t {
       public:
        using __id = __sender;
        using sender_concept = stdexec::sender_t;

        template <__not_decays_to<__t>... _Senders>
        explicit(sizeof...(_Senders) == 1)
          __t(_Senders&&... __senders) noexcept((__nothrow_decay_copyable<_Senders> && ...))
          : __senders_(static_cast<_Senders&&>(__senders)...) {
        }

        template <__decays_to<__t> _Self, receiver _Receiver>
        STDEXEC_MEMFN_DECL(
          auto connect)(this _Self&& __self, _Receiver __rcvr) //
          noexcept(__nothrow_constructible_from<__op_t<_Self, _Receiver>, _Self, _Receiver>)
            -> __op_t<_Self, _Receiver> {
          return __op_t<_Self, _Receiver>{
            static_cast<_Self&&>(__self).__senders_, static_cast<_Receiver&&>(__rcvr)};
        }

        template <__decays_to<__t> _Self, class _Env>
        static auto get_completion_signatures(_Self&&, _Env&&) noexcept
          -> __completions_t<_Env, __copy_cvref_t<_Self, stdexec::__t<_SenderIds>>...> {
          return {};
        }

       private:
        std::tuple<stdexec::__t<_SenderIds>...> __senders_;
      };
    };

    struct __when_any_t {
      template <class... _Senders>
      using __sender_t = __t<__sender<__id<__decay_t<_Senders>>...>>;

      template <sender... _Senders>
        requires(sizeof...(_Senders) > 0 && sender<__sender_t<_Senders...>>)
      auto operator()(_Senders&&... __senders) const
        noexcept((__nothrow_decay_copyable<_Senders> && ...)) -> __sender_t<_Senders...> {
        return __sender_t<_Senders...>(static_cast<_Senders&&>(__senders)...);
      }
    };

    inline constexpr __when_any_t when_any{};
  } // namespace __when_any

  using __when_any::when_any;
} // namespace exec
