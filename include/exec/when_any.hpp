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
    using namespace STDEXEC;

    template <class _Env>
    using __env_t = __join_env_t<prop<get_stop_token_t, inplace_stop_token>, _Env>;

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
      template <class... _CvSenders>
      using __all_value_args_nothrow_decay_copyable = __minvoke_q<
        __mand_t,
        __value_types_t<
          __completion_signatures_of_t<_CvSenders, __env_t<_Env>...>,
          __qq<__nothrow_decay_copyable_and_move_constructible_t>,
          __qq<__mand_t>
        >...
      >;

      template <class... _CvSenders>
      using __f = __mtry_q<__concat_completion_signatures_t>::__f<
        __eptr_completion_unless_t<__all_value_args_nothrow_decay_copyable<_CvSenders...>>,
        completion_signatures<set_stopped_t()>,
        __transform_completion_signatures_t<
          __completion_signatures_of_t<_CvSenders, __env_t<_Env>...>,
          __as_rvalues,
          __as_error,
          set_stopped_t (*)(),
          __completion_signature_ptrs_t
        >...
      >;
    };

    template <class _Env, class... _CvSenders>
    using __result_type_t = __for_each_completion_signature_t<
      __minvoke<__completions_fn<_Env>, _CvSenders...>,
      __decayed_tuple,
      __uniqued_variant
    >;

    template <class _Receiver>
    auto __make_visitor_fn(_Receiver& __rcvr) noexcept {
      return [&__rcvr]<class _Tuple>(_Tuple&& __result) noexcept {
        STDEXEC::__apply(
          [&__rcvr]<class... _As>(auto __tag, _As&... __args) noexcept {
            __tag(static_cast<_Receiver&&>(__rcvr), static_cast<_As&&>(__args)...);
          },
          __result);
      };
    }

    template <class _Receiver, class _ResultVariant>
    struct __opstate_base : __immovable {
      __opstate_base(_Receiver&& __rcvr, std::size_t __n_senders)
        : __count_{__n_senders}
        , __rcvr_{static_cast<_Receiver&&>(__rcvr)} {
      }

      using __on_stop =
        stop_callback_for_t<stop_token_of_t<env_of_t<_Receiver>&>, __forward_stop_request>;

      inplace_stop_source __stop_source_{};
      std::optional<__on_stop> __on_stop_{};

      // If this hits true, we store the result
      __std::atomic<bool> __emplaced_{false};
      // If this hits zero, we forward any result to the receiver
      __std::atomic<std::size_t> __count_{};

      _Receiver __rcvr_;
      _ResultVariant __result_{__no_init};

      template <class _Tag, class... _Args>
      void notify(_Tag, _Args&&... __args) noexcept {
        using __result_t = __decayed_tuple<_Tag, _Args...>;
        bool __expect = false;
        if (__emplaced_.compare_exchange_strong(
              __expect, true, __std::memory_order_relaxed, __std::memory_order_relaxed)) {
          // This emplacement can happen only once
          if constexpr ((__nothrow_decay_copyable<_Args> && ...)) {
            __result_.template emplace<__result_t>(_Tag{}, static_cast<_Args&&>(__args)...);
          } else {
            STDEXEC_TRY {
              __result_.template emplace<__result_t>(_Tag{}, static_cast<_Args&&>(__args)...);
            }
            STDEXEC_CATCH_ALL {
              using __error_t = __tuple<set_error_t, std::exception_ptr>;
              __result_.template emplace<__error_t>(set_error_t{}, std::current_exception());
            }
          }
          // stop pending operations
          __stop_source_.request_stop();
        }
        // make __result_ emplacement visible when __count_ goes from one to zero
        // This relies on the fact that each sender will call notify() at most once
        if (__count_.fetch_sub(1, __std::memory_order_acq_rel) == 1) {
          __on_stop_.reset();
          auto stop_token = get_stop_token(get_env(__rcvr_));
          if (stop_token.stop_requested()) {
            STDEXEC::set_stopped(static_cast<_Receiver&&>(__rcvr_));
            return;
          }
          STDEXEC_ASSERT(!__result_.__is_valueless());
          STDEXEC::__visit(
            __when_any::__make_visitor_fn(__rcvr_), static_cast<_ResultVariant&&>(__result_));
        }
      }
    };

    template <class _Receiver, class _ResultVariant>
    struct __receiver {
     public:
      using receiver_concept = STDEXEC::receiver_t;

      explicit __receiver(__opstate_base<_Receiver, _ResultVariant>* __op) noexcept
        : __op_{__op} {
      }

      auto get_env() const noexcept -> __env_t<env_of_t<_Receiver>> {
        auto __token = prop{get_stop_token, __op_->__stop_source_.get_token()};
        return __env::__join(std::move(__token), STDEXEC::get_env(__op_->__rcvr_));
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
      __opstate_base<_Receiver, _ResultVariant>* __op_;
    };

    template <class _Receiver, class... _CvSenders>
    struct __opstate
      : __opstate_base<_Receiver, __result_type_t<env_of_t<_Receiver>, _CvSenders...>> {
      using __result_t = __result_type_t<env_of_t<_Receiver>, _CvSenders...>;
      using __receiver_t = __receiver<_Receiver, __result_t>;
      using __op_base_t = __opstate_base<_Receiver, __result_t>;
      using __opstates_t = __tuple<connect_result_t<_CvSenders, __receiver_t>...>;

      static constexpr bool __nothrow_construct =
        (__nothrow_connectable<_CvSenders, __receiver_t> && ...);

     public:
      explicit __opstate(_Receiver&& __rcvr, _CvSenders&&... __sndrs) noexcept(__nothrow_construct)
        : __op_base_t{static_cast<_Receiver&&>(__rcvr), sizeof...(_CvSenders)}
        , __ops_{STDEXEC::connect(static_cast<_CvSenders&&>(__sndrs), __receiver_t{this})...} {
      }

      void start() & noexcept {
        this->__on_stop_.emplace(
          get_stop_token(get_env(this->__rcvr_)), __forward_stop_request{this->__stop_source_});
        if (this->__stop_source_.stop_requested()) {
          STDEXEC::set_stopped(static_cast<_Receiver&&>(this->__rcvr_));
        } else {
          STDEXEC::__apply(STDEXEC::__for_each{STDEXEC::start}, __ops_);
        }
      }

     private:
      __opstates_t __ops_;
    };

    template <class... _Senders>
    struct __sender {
      template <class _Self, class _Env>
      using __result_t = __result_type_t<_Env, __copy_cvref_t<_Self, _Senders>...>;

      template <class _Self, class _Receiver>
      using __receiver_t = __receiver<_Receiver, __result_t<_Self, env_of_t<_Receiver>>>;

      template <class _Self, class _Receiver>
      using __opstate_t = __opstate<_Receiver, __copy_cvref_t<_Self, _Senders>...>;

      template <class _Self, class... _Env>
      using __completions_t =
        __minvoke<__completions_fn<_Env...>, __copy_cvref_t<_Self, _Senders>...>;
     public:
      using sender_concept = STDEXEC::sender_t;

      template <__not_decays_to<__sender>... _CvSenders>
      explicit __sender(_CvSenders&&... __sndrs)
        noexcept((__nothrow_decay_copyable<_CvSenders> && ...))
        : __sndrs_{static_cast<_CvSenders&&>(__sndrs)...} {
      }

      template <__decay_copyable _Self, receiver _Receiver>
      STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this _Self&& __self, _Receiver __rcvr) noexcept( //
        STDEXEC::__nothrow_constructible_from<
          __opstate_t<_Self, _Receiver>,
          _Receiver,
          __copy_cvref_t<_Self, _Senders>...
        >) -> __opstate_t<_Self, _Receiver> {
        return STDEXEC::__apply(
          STDEXEC::__construct<__opstate_t<_Self, _Receiver>>{},
          static_cast<_Self&&>(__self).__sndrs_,
          static_cast<_Receiver&&>(__rcvr));
      }
      STDEXEC_EXPLICIT_THIS_END(connect)

      template <__decay_copyable _Self, class... _Env>
      static consteval auto get_completion_signatures() {
        return __completions_t<_Self, _Env...>{};
      }

      template <class _Self, class... _Env>
      static consteval auto get_completion_signatures() {
        return STDEXEC::__throw_compile_time_error<
          _SENDER_TYPE_IS_NOT_DECAY_COPYABLE_,
          _WITH_PRETTY_SENDERS_<_Senders>...
        >();
      }

     private:
      __tuple<_Senders...> __sndrs_;
    };

    struct __when_any_t {
      template <class... _CvSenders>
      using __sender_t = __sender<__decay_t<_CvSenders>...>;

      auto operator()() const noexcept = delete;

      template <sender... _CvSenders>
      constexpr auto operator()(_CvSenders&&... __sndrs) const
        noexcept((__nothrow_decay_copyable<_CvSenders> && ...)) -> __sender_t<_CvSenders...> {
        return __sender_t<_CvSenders...>(static_cast<_CvSenders&&>(__sndrs)...);
      }
    };

    inline constexpr __when_any_t when_any{};
  } // namespace __when_any

  using __when_any::when_any;
} // namespace exec

STDEXEC_PRAGMA_POP()
