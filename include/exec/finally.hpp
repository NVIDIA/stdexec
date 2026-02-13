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

#include "../stdexec/__detail/__execution_fwd.hpp"

// include these after execution_fwd.hpp
#include "../stdexec/__detail/__basic_sender.hpp"
#include "../stdexec/__detail/__optional.hpp"
#include "../stdexec/__detail/__senders.hpp"
#include "../stdexec/__detail/__tuple.hpp"
#include "../stdexec/__detail/__utility.hpp"
#include "../stdexec/__detail/__variant.hpp"

#include "completion_signatures.hpp"

namespace exec {
  namespace __final {
    template <class _Initial, class _Final>
    struct __sender;
  } // namespace __final

  struct _THE_FINAL_SENDER_MUST_BE_A_SENDER_OF_VOID_ { };
  struct _INVALID_ARGUMENT_TO_THE_FINALLY_ALGORITHM_ { };

  struct finally_t {
    template <STDEXEC::sender _Initial, STDEXEC::sender _Final>
    auto operator()(_Initial&& __initial, _Final&& __final) const //
      -> STDEXEC::__well_formed_sender auto {
      return STDEXEC::__make_sexpr<finally_t>(
        {}, static_cast<_Initial&&>(__initial), static_cast<_Final&&>(__final));
    }

    template <STDEXEC::sender _Final>
    STDEXEC_ATTRIBUTE(always_inline)
    auto operator()(_Final&& __final) const {
      return __closure(*this, static_cast<_Final&&>(__final));
    }

    template <class _Sender>
    static auto transform_sender(STDEXEC::set_value_t, _Sender&& __sndr, STDEXEC::__ignore) {
      auto& [__tag, __ign, __initial, __final] = __sndr;
      return __final ::__sender{
        STDEXEC::__forward_like<_Sender>(__initial), //
        STDEXEC::__forward_like<_Sender>(__final)};
    }
  };

  inline constexpr finally_t finally{};

  namespace __final {
    using namespace STDEXEC;

    template <bool _FinalSenderHasValueCompletions>
    struct __result_variant_fn {
      template <class _InitialSender, class _Receiver>
      using __f = __for_each_completion_signature_t<
        completion_signatures_of_t<_InitialSender, env_of_t<_Receiver>>,
        __decayed_tuple,
        __variant
      >;
    };

    template <>
    struct __result_variant_fn<false> {
      template <class, class>
      using __f = __variant<>;
    };

    // If the final sender has no value completions, then we don't need to store the
    // initial sender's values because they won't be propagated.
    template <class _CvInitialSender, class _CvFinalSender, class _Receiver>
    using __result_variant_t = __mcall2<
      __result_variant_fn<__sends<set_value_t, _CvFinalSender, __fwd_env_t<env_of_t<_Receiver>>>>,
      _CvInitialSender,
      _Receiver
    >;

    template <class _ResultType, class _Receiver>
    struct __opstate_base {
      _Receiver __rcvr_{};
      _ResultType __result_{__no_init}; // __variant<__tuple<set_tag, args...>, ...>
    };

    struct __applier {
      template <class _Receiver, class _Tag, class... _Args>
      void operator()(_Receiver& __rcvr, _Tag, _Args&&... __args) noexcept {
        _Tag()(static_cast<_Receiver&&>(__rcvr), static_cast<_Args&&>(__args)...);
      }
    };

    struct __visitor {
      template <class _Receiver, class _Tuple>
      void operator()(_Receiver& __rcvr, _Tuple&& __tuple) noexcept {
        STDEXEC::__apply(__applier{}, static_cast<_Tuple&&>(__tuple), __rcvr);
      }
    };

    template <class _ResultType, class _Receiver>
    struct __final_receiver {
      using receiver_concept = receiver_t;

      void set_value() noexcept {
        STDEXEC::__visit(__visitor{}, std::move(__opstate_->__result_), __opstate_->__rcvr_);
      }

      template <class _Error>
      void set_error(_Error&& __error) noexcept {
        STDEXEC::set_error(
          static_cast<_Receiver&&>(__opstate_->__rcvr_), static_cast<_Error&&>(__error));
      }

      void set_stopped() noexcept {
        STDEXEC::set_stopped(static_cast<_Receiver&&>(__opstate_->__rcvr_));
      }

      auto get_env() const noexcept -> __fwd_env_t<env_of_t<_Receiver>> {
        return __fwd_env(STDEXEC::get_env(__opstate_->__rcvr_));
      }

      __opstate_base<_ResultType, _Receiver>* __opstate_;
    };

    template <class _CvFinalSender, class _ResultType, class _Receiver>
    struct __final_opstate : __opstate_base<_ResultType, _Receiver> {
      using __cleanup_callback_t = void(__final_opstate*) noexcept;
      using __final_receiver_t = __final_receiver<_ResultType, _Receiver>;
      using __final_opstate_t = connect_result_t<_CvFinalSender, __final_receiver_t>;

      constexpr explicit __final_opstate(
        __cleanup_callback_t* __cleanup_callback,
        _CvFinalSender&& __final,
        _Receiver&& __rcvr) noexcept
        : __opstate_base<_ResultType, _Receiver>{static_cast<_Receiver&&>(__rcvr)}
        , __cleanup_callback_{__cleanup_callback}
        , __final_opstate_(
            STDEXEC::connect(static_cast<_CvFinalSender&&>(__final), __final_receiver_t{this})) {
      }

      template <class... _Args>
      void __store_result_and_start_next_op([[maybe_unused]] _Args&&... __args) noexcept {
        if constexpr (!__sends<set_value_t, _CvFinalSender, __fwd_env_t<env_of_t<_Receiver>>>) {
          // If the final sender has no set_value completions, then we don't need to store the
          // initial sender's values because they won't be propagated.
          (*__cleanup_callback_)(this);
          STDEXEC::start(this->__final_opstate_);
        } else {
          STDEXEC_TRY {
            using __tuple_t = __decayed_tuple<_Args...>;
            this->__result_.template emplace<__tuple_t>(static_cast<_Args&&>(__args)...);
            (*__cleanup_callback_)(this);
            STDEXEC::start(this->__final_opstate_);
          }
          STDEXEC_CATCH_ALL {
            if constexpr (!__nothrow_decay_copyable<_Args...>) {
              (*__cleanup_callback_)(this);
              STDEXEC::set_error(static_cast<_Receiver&&>(this->__rcvr_), std::current_exception());
            }
          }
        }
      }

      __cleanup_callback_t* __cleanup_callback_;
      __final_opstate_t __final_opstate_;
    };

    template <class _CvFinalSender, class _ResultType, class _Receiver>
    struct __initial_receiver;

    template <class _CvInitialSender, class _CvFinalSender, class _Receiver>
    struct __opstate
      : __final_opstate<
          _CvFinalSender,
          __result_variant_t<_CvInitialSender, _CvFinalSender, _Receiver>,
          _Receiver
        > {
      using __initial_results_t = __result_variant_t<_CvInitialSender, _CvFinalSender, _Receiver>;
      using __base_t = __final_opstate<_CvFinalSender, __initial_results_t, _Receiver>;

      explicit __opstate(
        _CvInitialSender&& __initial,
        _CvFinalSender&& __final,
        _Receiver __receiver)
        : __base_t(
            &__cleanup_initial_opstate,
            static_cast<_CvFinalSender&&>(__final),
            static_cast<_Receiver&&>(__receiver)) {
        __initial_opstate_.__emplace_from(
          STDEXEC::connect, static_cast<_CvInitialSender&&>(__initial), __initial_receiver_t{this});
      }

      void start() & noexcept {
        STDEXEC::start(*__initial_opstate_);
      }

     private:
      using __initial_receiver_t =
        __initial_receiver<_CvFinalSender, __initial_results_t, _Receiver>;
      using __initial_opstate_t = connect_result_t<_CvInitialSender, __initial_receiver_t>;

      static void __cleanup_initial_opstate(__base_t* __base) noexcept {
        auto* __self = static_cast<__opstate*>(__base);
        __self->__initial_opstate_.reset();
      }

      __optional<__initial_opstate_t> __initial_opstate_{};
    };

    template <class _CvFinalSender, class _ResultType, class _Receiver>
    struct __initial_receiver {
      using receiver_concept = receiver_t;

      template <class... _As>
      void set_value(_As&&... __as) noexcept {
        __opstate_
          ->__store_result_and_start_next_op(STDEXEC::set_value, static_cast<_As&&>(__as)...);
      }

      template <class _Error>
      void set_error(_Error&& __error) noexcept {
        __opstate_
          ->__store_result_and_start_next_op(STDEXEC::set_error, static_cast<_Error&&>(__error));
      }

      void set_stopped() noexcept {
        __opstate_->__store_result_and_start_next_op(STDEXEC::set_stopped);
      }

      auto get_env() const noexcept -> env_of_t<_Receiver> {
        return STDEXEC::get_env(__opstate_->__rcvr_);
      }

      __final_opstate<_CvFinalSender, _ResultType, _Receiver>* __opstate_;
    };

    template <class _CvInitialSender, class _CvFinalSender, class... _Env>
    consteval auto __get_completion_signatures() {
      STDEXEC_COMPLSIGS_LET(
        __initial_completions, STDEXEC::get_completion_signatures<_CvInitialSender, _Env...>()) {
        using __initial_completions_t = decltype(__initial_completions);
        auto __final_completions =
          STDEXEC::get_completion_signatures<_CvFinalSender, __fwd_env_t<_Env>...>();

        if constexpr (!__sends<set_value_t, _CvFinalSender, __fwd_env_t<_Env>...>) {
          // If the finally sender doesn't have set_value completions, then we
          // don't need to worry about the initial sender's value types not being
          // nothrow decay-copyable, because they won't be propagated to the
          // receiver.
          return exec::transform_completion_signatures(
            __initial_completions, ignore_completion(), {}, {}, __final_completions);
        } else if constexpr (!sender_of<_CvFinalSender, set_value_t(), __fwd_env_t<_Env>...>) {
          // If the finally sender has value completions other than set_value_t(), then
          // throw a compilation error.
          return exec::throw_compile_time_error<
            _WHAT_(_INVALID_ARGUMENT_TO_THE_FINALLY_ALGORITHM_),
            _WHERE_(_IN_ALGORITHM_, finally_t),
            _WHY_(_THE_FINAL_SENDER_MUST_BE_A_SENDER_OF_VOID_)
          >();
        } else {
          using __is_nothrow_t = __nothrow_decay_copyable_results_t<__initial_completions_t>;
          // The finally sender's completion signatures are ...
          return exec::concat_completion_signatures(
            // ... the initial sender's completions with value types decayed ...
            exec::transform_completion_signatures(
              __initial_completions, exec::decay_arguments<set_value_t>()),
            // ... and the final sender's error and stopped completions ...
            exec::transform_completion_signatures(__final_completions, exec::ignore_completion()),
            // ... and possibly a set_error(exception_ptr) completion.
            __eptr_completion_unless_t<__is_nothrow_t>());
        }
      }
    }

    template <class _InitialSender, class _FinalSender>
    struct __sender {
      using sender_concept = sender_t;

      template <class _Self, class _Receiver>
      using __opstate_t = __opstate<
        __copy_cvref_t<_Self, _InitialSender>,
        __copy_cvref_t<_Self, _FinalSender>,
        _Receiver
      >;

      template <__decays_to<__sender> _Self, class _Receiver>
      STDEXEC_EXPLICIT_THIS_BEGIN(
        auto connect)(this _Self&& __self, _Receiver&& __receiver) noexcept
        -> __opstate_t<_Self, _Receiver> {
        return __opstate_t<_Self, _Receiver>{
          static_cast<_Self&&>(__self).__initial_sndr_,
          static_cast<_Self&&>(__self).__final_sndr_,
          static_cast<_Receiver&&>(__receiver)};
      }
      STDEXEC_EXPLICIT_THIS_END(connect)

      template <__decays_to<__sender> _Self, class... _Env>
      static consteval auto get_completion_signatures() {
        return __final ::__get_completion_signatures<
          __copy_cvref_t<_Self, _InitialSender>,
          __copy_cvref_t<_Self, _FinalSender>,
          _Env...
        >();
      }

      _InitialSender __initial_sndr_;
      _FinalSender __final_sndr_;
    };

    template <class _InitialSender, class _FinalSender>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
      __sender(_InitialSender, _FinalSender) -> __sender<_InitialSender, _FinalSender>;
  } // namespace __final
} // namespace exec

namespace STDEXEC {
  template <>
  struct __sexpr_impl<exec::finally_t> : __sexpr_defaults {
    template <class _Sender, class... _Env>
    static consteval auto __get_completion_signatures() {
      return exec::__final ::__get_completion_signatures<
        __nth_child_of_c<0, _Sender>,
        __nth_child_of_c<1, _Sender>,
        _Env...
      >();
    }
  };
} // namespace STDEXEC
