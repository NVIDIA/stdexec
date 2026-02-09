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

// include these after execution.hpp
#include "../stdexec/__detail/__manual_lifetime.hpp"

#include "completion_signatures.hpp"

namespace exec {
  namespace __final {
    using namespace STDEXEC;

    template <class _Sigs>
    using __result_variant_t = __for_each_completion_signature_t<_Sigs, __decayed_tuple, __variant>;

    template <class _ResultType, class _Receiver>
    struct __opstate_base {
      _Receiver __rcvr_{};
      __manual_lifetime<_ResultType> __result_{};
    };

    struct __applier {
      // We intentially take the arguments by value so they will still be valid
      // after we clear the result storage.
      template <class _OpState, class _Tag, class... _Args>
      void operator()(_OpState* __op, _Tag __tag, _Args... __args) noexcept {
        __op->__result_.__destroy();
        __tag(std::move(__op->__rcvr_), static_cast<_Args&&>(__args)...);
      }
    };

    struct __visitor {
      template <class _OpState, class _Tuple>
      void operator()(_OpState* __op, _Tuple&& __tuple) noexcept(__nothrow_decay_copyable<_Tuple>) {
        STDEXEC::__apply(__applier{}, static_cast<_Tuple&&>(__tuple), __op);
      }
    };

    template <class _ResultType, class _Receiver>
    struct __final_receiver {
      using receiver_concept = receiver_t;

      explicit __final_receiver(__opstate_base<_ResultType, _Receiver>* __op) noexcept
        : __opstate_{__op} {
      }

      auto get_env() const noexcept -> env_of_t<_Receiver> {
        return STDEXEC::get_env(__opstate_->__rcvr_);
      }

      void set_value() noexcept {
        STDEXEC_TRY {
          auto& __result = __opstate_->__result_.__get();
          STDEXEC::__visit(__visitor{}, static_cast<_ResultType&&>(__result), __opstate_);
        }
        STDEXEC_CATCH_ALL {
          if constexpr (!__mapply_q<__nothrow_decay_copyable_t, _ResultType>::value) {
            STDEXEC::set_error(
              static_cast<_Receiver&&>(__opstate_->__rcvr_), std::current_exception());
          }
        }
      }

      template <class _Error>
      void set_error(_Error&& __error) noexcept {
        __opstate_->__result_.__destroy();
        STDEXEC::set_error(
          static_cast<_Receiver&&>(__opstate_->__rcvr_), static_cast<_Error&&>(__error));
      }

      void set_stopped() noexcept {
        __opstate_->__result_.__destroy();
        STDEXEC::set_stopped(static_cast<_Receiver&&>(__opstate_->__rcvr_));
      }

     private:
      __opstate_base<_ResultType, _Receiver>* __opstate_;
    };

    template <class _InitialSender, class _FinalSender, class _Receiver>
    struct __initial_receiver;

    template <class _InitialSender, class _FinalSender, class _Receiver>
    struct __opstate
      : __opstate_base<
          __result_variant_t<completion_signatures_of_t<_InitialSender, env_of_t<_Receiver>>>,
          _Receiver
        > {
      using __signatures = completion_signatures_of_t<_InitialSender, env_of_t<_Receiver>>;
      using __base_t = __opstate_base<__result_variant_t<__signatures>, _Receiver>;
      using __final_receiver_t = __final_receiver<__result_variant_t<__signatures>, _Receiver>;
      using __final_opstate_t = connect_result_t<_FinalSender, __final_receiver_t>;

      template <class... _Args>
      void __store_result_and_start_next_op(_Args&&... __args) {
        this->__result_.__construct(STDEXEC::__no_init)
          .template emplace<__decayed_tuple<_Args...>>(static_cast<_Args&&>(__args)...);
        STDEXEC_ASSERT(__current_opstate_.index() == 0);
        auto __final = static_cast<_FinalSender&&>(__var::__get<0>(__current_opstate_).__sndr_);
        __final_opstate_t& __final_op = __current_opstate_.template __emplace_from<1>(
          STDEXEC::connect, static_cast<_FinalSender&&>(__final), __final_receiver_t{this});
        STDEXEC::start(__final_op);
      }

      explicit __opstate(_InitialSender&& __initial, _FinalSender __final, _Receiver __receiver)
        : __base_t{{static_cast<_Receiver&&>(__receiver)}}
        , __current_opstate_(STDEXEC::__no_init) {
        __current_opstate_.template emplace<0>(
          static_cast<_InitialSender&&>(__initial),
          static_cast<_FinalSender&&>(__final),
          __initial_receiver_t{this});
      }

      void start() & noexcept {
        STDEXEC_ASSERT(__current_opstate_.index() == 0);
        STDEXEC::start(__var::__get<0>(__current_opstate_).__initial_opstate_);
      }

     private:
      using __initial_receiver_t = __initial_receiver<_InitialSender, _FinalSender, _Receiver>;

      struct __initial_op_t {
        explicit __initial_op_t(
          _InitialSender&& __sndr,
          _FinalSender&& __final,
          __initial_receiver_t __rcvr)
          : __sndr_{static_cast<_FinalSender&&>(__final)}
          , __initial_opstate_{STDEXEC::connect(
              static_cast<_InitialSender&&>(__sndr),
              static_cast<__initial_receiver_t&&>(__rcvr))} {
        }

        _FinalSender __sndr_;
        connect_result_t<_InitialSender, __initial_receiver_t> __initial_opstate_;
      };

      __variant<__initial_op_t, __final_opstate_t> __current_opstate_;
    };

    template <class _InitialSender, class _FinalSender, class _Receiver>
    struct __initial_receiver {
      using receiver_concept = receiver_t;
      using __base_op_t = __opstate<_InitialSender, _FinalSender, _Receiver>;

      explicit __initial_receiver(__base_op_t* __op) noexcept
        : __opstate_(__op) {
      }

      template <class... _As>
      void set_value(_As&&... __as) noexcept {
        STDEXEC_TRY {
          __opstate_
            ->__store_result_and_start_next_op(STDEXEC::set_value, static_cast<_As&&>(__as)...);
        }
        STDEXEC_CATCH_ALL {
          STDEXEC::set_error(
            static_cast<_Receiver&&>(__opstate_->__rcvr_), std::current_exception());
        }
      }

      template <class _Error>
      void set_error(_Error&& __error) noexcept {
        STDEXEC_TRY {
          __opstate_
            ->__store_result_and_start_next_op(STDEXEC::set_error, static_cast<_Error&&>(__error));
        }
        STDEXEC_CATCH_ALL {
          STDEXEC::set_error(
            static_cast<_Receiver&&>(__opstate_->__rcvr_), std::current_exception());
        }
      }

      void set_stopped() noexcept {
        STDEXEC_TRY {
          __opstate_->__store_result_and_start_next_op(STDEXEC::set_stopped);
        }
        STDEXEC_CATCH_ALL {
          STDEXEC::set_error(
            static_cast<_Receiver&&>(__opstate_->__rcvr_), std::current_exception());
        }
      }

      auto get_env() const noexcept -> env_of_t<_Receiver> {
        return STDEXEC::get_env(__opstate_->__rcvr_);
      }

     private:
      __base_op_t* __opstate_;
    };

    template <class _InitialSender, class _FinalSender>
    struct __sender {
      using sender_concept = sender_t;

      template <class _Self, class _Receiver>
      using __opstate_t = __opstate<__copy_cvref_t<_Self, _InitialSender>, _FinalSender, _Receiver>;

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
        requires __decay_copyable<__copy_cvref_t<_Self, _FinalSender>>
      static consteval auto get_completion_signatures() {
        STDEXEC_COMPLSIGS_LET(
          __initial_completions,
          exec::get_child_completion_signatures<_Self, _InitialSender, _Env...>()) {
          STDEXEC_COMPLSIGS_LET(
            __final_completions,
            STDEXEC::get_completion_signatures<_FinalSender, __fwd_env_t<_Env>...>()) {
            // The finally sender's completion signatures are ...
            return exec::concat_completion_signatures(
              // ... the initial sender's completions with value types decayed ...
              exec::transform_completion_signatures(
                __initial_completions, exec::decay_arguments<set_value_t>()),
              // ... and the finally sender's error and stopped completions ...
              exec::transform_completion_signatures(__final_completions, exec::ignore_completion()),
              // ... and a set_error(exception_ptr) (TODO: only needed if the
              // values of the initial sender are not nothrow decay copyable).
              completion_signatures<set_error_t(std::exception_ptr)>());
          }
        }
      }

      template <__decays_to<__sender> _Self, class... _Env>
      static consteval auto get_completion_signatures() {
        return exec::throw_compile_time_error<
          _SENDER_TYPE_IS_NOT_DECAY_COPYABLE_,
          _WITH_PRETTY_SENDER_<_FinalSender>
        >();
      }

      _InitialSender __initial_sndr_;
      _FinalSender __final_sndr_;
    };

    struct finally_t {
      template <sender _Initial, sender _Final>
      auto operator()(_Initial&& __initial, _Final&& __final) const {
        return __make_sexpr<finally_t>(
          {}, static_cast<_Initial&&>(__initial), static_cast<_Final&&>(__final));
      }

      template <sender _Final>
      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()(_Final&& __final) const {
        return __closure(*this, static_cast<_Final&&>(__final));
      }

      template <class _Sender>
      static auto transform_sender(set_value_t, _Sender&& __sndr, __ignore) {
        return __apply(
          []<class _Initial, class _Final>(
            __ignore, __ignore, _Initial&& __initial, _Final&& __final) {
            return __sender<_Initial, _Final>{
              static_cast<_Initial&&>(__initial), static_cast<_Final&&>(__final)};
          },
          static_cast<_Sender&&>(__sndr));
      }
    };
  } // namespace __final

  using __final::finally_t;
  inline constexpr __final::finally_t finally{};
} // namespace exec

namespace STDEXEC {
  template <>
  struct __sexpr_impl<exec::finally_t> : __sexpr_defaults {
    template <class _Sender, class... _Env>
    static consteval auto get_completion_signatures() {
      using __sndr_t =
        __detail::__transform_sender_result_t<exec::finally_t, set_value_t, _Sender, env<>>;
      return STDEXEC::get_completion_signatures<__sndr_t, _Env...>();
    }
  };
} // namespace STDEXEC
