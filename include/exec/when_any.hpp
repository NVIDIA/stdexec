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

#include "../stdexec/execution.hpp"
#include "../stdexec/stop_token.hpp"

#include <exception>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

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
    using __env_t = __join_env_t<prop<get_stop_token_t, inplace_stop_token>, _BaseEnv>;

    template <class... _Ts>
    using __nothrow_decay_copyable_and_move_constructible_t = __mbool<(
      (__nothrow_decay_copyable<_Ts> && __nothrow_move_constructible<__decay_t<_Ts>>) && ...)>;

    template <class... Args>
    using __as_rvalues = set_value_t (*)(__decay_t<Args>...);

    template <class... E>
    using __as_error = set_error_t (*)(E...);

    // Here we convert all set_value(Args...) to set_value(__decay_t<Args>...). Note, we keep all
    // error types as they are and unconditionally add set_stopped(). The indirection through the
    // __completions_fn is to avoid a pack expansion bug in nvc++.
    template <class... _Env>
    struct __completions_fn {
      template <class... _CvrefSenders>
      using __all_value_args_nothrow_decay_copyable = __mand_t<__value_types_t<
        __completion_signatures_of_t<_CvrefSenders, _Env...>,
        __qq<__nothrow_decay_copyable_and_move_constructible_t>,
        __qq<__mand_t>
      >...>;

      template <class... _CvrefSenders>
      using __f = __mtry_q<__concat_completion_signatures>::__f<
        __eptr_completion_if_t<__all_value_args_nothrow_decay_copyable<_CvrefSenders...>>,
        completion_signatures<set_stopped_t()>,
        __transform_completion_signatures<
          __completion_signatures_of_t<_CvrefSenders, _Env...>,
          __as_rvalues,
          __as_error,
          set_stopped_t (*)(),
          __completion_signature_ptrs
        >...
      >;
    };

    template <class _Env, class... _CvrefSenders>
    using __result_type_t = __for_each_completion_signature<
      __minvoke<__completions_fn<_Env>, _CvrefSenders...>,
      __decayed_tuple,
      __uniqued_variant_for
    >;

    template <class _Receiver>
    auto __make_visitor_fn(_Receiver& __rcvr) noexcept {
      return [&__rcvr]<class _Tuple>(_Tuple&& __result) noexcept {
        __result.apply(
          [&__rcvr]<class... _As>(auto __tag, _As&... __args) noexcept {
            __tag(static_cast<_Receiver&&>(__rcvr), static_cast<_As&&>(__args)...);
          },
          __result);
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
      _ResultVariant __result_{};

      template <class _Tag, class... _Args>
      void notify(_Tag, _Args&&... __args) noexcept {
        using __result_t = __decayed_tuple<_Tag, _Args...>;
        bool __expect = false;
        if (__emplaced_.compare_exchange_strong(
              __expect, true, std::memory_order_relaxed, std::memory_order_relaxed)) {
          // This emplacement can happen only once
          if constexpr ((__nothrow_decay_copyable<_Args> && ...)) {
            __result_.template emplace<__result_t>(_Tag{}, static_cast<_Args&&>(__args)...);
          } else {
            STDEXEC_TRY {
              __result_.template emplace<__result_t>(_Tag{}, static_cast<_Args&&>(__args)...);
            }
            STDEXEC_CATCH_ALL {
              using __error_t = __tuple_for<set_error_t, std::exception_ptr>;
              __result_.template emplace<__error_t>(set_error_t{}, std::current_exception());
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
          STDEXEC_ASSERT(!__result_.is_valueless());
          __result_.visit(
            __when_any::__make_visitor_fn(__rcvr_), static_cast<_ResultVariant&&>(__result_));
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
          auto __token = prop{get_stop_token, __op_->__stop_source_.get_token()};
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

      static constexpr bool __nothrow_construct =
        __nothrow_move_constructible<_Receiver>
        && (__nothrow_connectable<__cvref_t<_CvrefSenderIds>, __receiver_t> && ...);

      class __t : __op_base_t {
        using __opstate_tuple =
          __tuple_for<connect_result_t<stdexec::__cvref_t<_CvrefSenderIds>, __receiver_t>...>;
       public:
        template <class _SenderTuple>
        __t(_SenderTuple&& __senders, _Receiver&& __rcvr) noexcept(__nothrow_construct)
          : __op_base_t{static_cast<_Receiver&&>(__rcvr), sizeof...(_CvrefSenderIds)}
          , __ops_{__senders.apply(
              [this]<class... _Senders>(_Senders&&... __sndrs) noexcept(
                __nothrow_construct) -> __opstate_tuple {
                return __opstate_tuple{
                  stdexec::connect(static_cast<_Senders&&>(__sndrs), __receiver_t{this})...};
              },
              static_cast<_SenderTuple&&>(__senders))} {
        }

        void start() & noexcept {
          this->__on_stop_.emplace(
            get_stop_token(get_env(this->__rcvr_)), __on_stop_requested{this->__stop_source_});
          if (this->__stop_source_.stop_requested()) {
            stdexec::set_stopped(static_cast<_Receiver&&>(this->__rcvr_));
          } else {
            __ops_.for_each(stdexec::start, __ops_);
          }
        }

       private:
        __opstate_tuple __ops_;
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

      template <class _Self, class... _Env>
      using __completions_t = __minvoke<
        __when_any::__completions_fn<_Env...>,
        __copy_cvref_t<_Self, stdexec::__t<_SenderIds>>...
      >;

      class __t {
       public:
        using __id = __sender;
        using sender_concept = stdexec::sender_t;
        using __senders_tuple = __tuple_for<stdexec::__t<_SenderIds>...>;

        template <__not_decays_to<__t>... _Senders>
        explicit(sizeof...(_Senders) == 1) __t(_Senders&&... __senders)
          noexcept((__nothrow_decay_copyable<_Senders> && ...))
          : __senders_{static_cast<_Senders&&>(__senders)...} {
        }

        template <__decays_to<__t> _Self, receiver _Receiver>
        static auto
          connect(_Self&& __self, _Receiver __rcvr) noexcept(__nothrow_constructible_from<
                                                             __op_t<_Self, _Receiver>,
                                                             __copy_cvref_t<_Self, __senders_tuple>,
                                                             _Receiver
          >) -> __op_t<_Self, _Receiver> {
          return __op_t<_Self, _Receiver>{
            static_cast<_Self&&>(__self).__senders_, static_cast<_Receiver&&>(__rcvr)};
        }

        template <__decays_to<__t> _Self, class... _Env>
        static auto get_completion_signatures(_Self&&, _Env&&...) noexcept
          -> __completions_t<_Self, _Env...> {
          return {};
        }

       private:
        __senders_tuple __senders_;
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

STDEXEC_PRAGMA_POP()
