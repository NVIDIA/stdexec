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
    using __result_variant = __for_each_completion_signature_t<_Sigs, __decayed_tuple, __variant_for>;

    template <class _ResultType, class _ReceiverId>
    struct __final_operation_base {
      using _Receiver = __t<_ReceiverId>;

      _Receiver __receiver_{};
      __manual_lifetime<_ResultType> __result_{};
    };

    struct __applier {
      // We intentially take the arguments by value so they will still be valid
      // after we clear the result storage.
      template <class _OpState, class _Tag, class... _Args>
      void operator()(_OpState* __op, _Tag __tag, _Args... __args) noexcept {
        __op->__result_.__destroy();
        __tag(std::move(__op->__receiver_), static_cast<_Args&&>(__args)...);
      }
    };

    struct __visitor {
      template <class _OpState, class _Tuple>
      void operator()(_OpState* __op, _Tuple&& __tuple) noexcept {
        STDEXEC::__apply(__applier{}, static_cast<_Tuple&&>(__tuple), __op);
      }
    };

    template <class _ResultType, class _ReceiverId>
    struct __final_receiver {
      using _Receiver = STDEXEC::__t<_ReceiverId>;

      class __t {
       public:
        using __id = __final_receiver;
        using receiver_concept = receiver_t;

        explicit __t(__final_operation_base<_ResultType, _ReceiverId>* __op) noexcept
          : __op_{__op} {
        }

        auto get_env() const noexcept -> env_of_t<_Receiver> {
          return STDEXEC::get_env(__op_->__receiver_);
        }

        void set_value() noexcept {
          STDEXEC_TRY {
            auto& __result = __op_->__result_.__get();
            STDEXEC::__visit(__visitor{}, static_cast<_ResultType&&>(__result), __op_);
          }
          STDEXEC_CATCH_ALL {
            if constexpr (!__mapply_q<__nothrow_decay_copyable_t, _ResultType>::value) {
              STDEXEC::set_error(
                static_cast<_Receiver&&>(__op_->__receiver_), std::current_exception());
            }
          }
        }

        template <class _Error>
        void set_error(_Error&& __error) noexcept {
          __op_->__result_.__destroy();
          STDEXEC::set_error(
            static_cast<_Receiver&&>(__op_->__receiver_), static_cast<_Error&&>(__error));
        }

        void set_stopped() noexcept {
          __op_->__result_.__destroy();
          STDEXEC::set_stopped(static_cast<_Receiver&&>(__op_->__receiver_));
        }

       private:
        __final_operation_base<_ResultType, _ReceiverId>* __op_;
      };
    };

    template <class _InitialSenderId, class _FinalSenderId, class _ReceiverId>
    struct __operation_state {
      using _InitialSender = __cvref_t<_InitialSenderId>;
      using _FinalSender = STDEXEC::__t<_FinalSenderId>;
      using _Receiver = STDEXEC::__t<_ReceiverId>;
      using __signatures = completion_signatures_of_t<_InitialSender, env_of_t<_Receiver>>;
      using __base_t = __final_operation_base<__result_variant<__signatures>, _ReceiverId>;
      using __final_receiver_t =
        STDEXEC::__t<__final_receiver<__result_variant<__signatures>, _ReceiverId>>;
      using __final_op_t = connect_result_t<_FinalSender, __final_receiver_t>;
      struct __t;
    };

    template <class _InitialSenderId, class _FinalSenderId, class _ReceiverId>
    struct __initial_receiver {
      using _Receiver = __cvref_t<_ReceiverId>;
      using _FinalSender = STDEXEC::__t<_FinalSenderId>;

      using __base_op_t =
        STDEXEC::__t<__operation_state<_InitialSenderId, _FinalSenderId, _ReceiverId>>;

      class __t {
       public:
        using __id = __initial_receiver;
        using receiver_concept = receiver_t;

        explicit __t(__base_op_t* __op) noexcept
          : __op_(__op) {
        }

        template <class... _As>
        void set_value(_As&&... __as) noexcept {
          STDEXEC_TRY {
            __op_
              ->__store_result_and_start_next_op(STDEXEC::set_value, static_cast<_As&&>(__as)...);
          }
          STDEXEC_CATCH_ALL {
            STDEXEC::set_error(
              static_cast<_Receiver&&>(__op_->__receiver_), std::current_exception());
          }
        }

        template <class _Error>
        void set_error(_Error&& __err) noexcept {
          STDEXEC_TRY {
            __op_
              ->__store_result_and_start_next_op(STDEXEC::set_error, static_cast<_Error&&>(__err));
          }
          STDEXEC_CATCH_ALL {
            STDEXEC::set_error(
              static_cast<_Receiver&&>(__op_->__receiver_), std::current_exception());
          }
        }

        void set_stopped() noexcept {
          STDEXEC_TRY {
            __op_->__store_result_and_start_next_op(STDEXEC::set_stopped);
          }
          STDEXEC_CATCH_ALL {
            STDEXEC::set_error(
              static_cast<_Receiver&&>(__op_->__receiver_), std::current_exception());
          }
        }

        auto get_env() const noexcept -> env_of_t<_Receiver> {
          return STDEXEC::get_env(__op_->__receiver_);
        }

       private:
        __base_op_t* __op_;
      };
    };

    template <class _InitialSenderId, class _FinalSenderId, class _ReceiverId>
    struct __operation_state<_InitialSenderId, _FinalSenderId, _ReceiverId>::__t : __base_t {
      using __id = __operation_state;

      template <class... _Args>
      void __store_result_and_start_next_op(_Args&&... __args) {
        this->__result_.__construct()
          .template emplace<__decayed_tuple<_Args...>>(static_cast<_Args&&>(__args)...);
        STDEXEC_ASSERT(__op_.index() == 0);
        auto __final = static_cast<_FinalSender&&>(__op_.template get<0>().__sndr_);
        __final_op_t& __final_op = __op_.template __emplace_from<1>(
          STDEXEC::connect, static_cast<_FinalSender&&>(__final), __final_receiver_t{this});
        STDEXEC::start(__final_op);
      }

      explicit __t(_InitialSender&& __initial, _FinalSender __final, _Receiver __receiver)
        : __base_t{{static_cast<_Receiver&&>(__receiver)}}
        , __op_() {
        __op_.template emplace<0>(
          static_cast<_InitialSender&&>(__initial),
          static_cast<_FinalSender&&>(__final),
          __initial_receiver_t{this});
      }

      void start() & noexcept {
        STDEXEC_ASSERT(__op_.index() == 0);
        STDEXEC::start(__op_.template get<0>().__initial_operation_);
      }

     private:
      using __initial_receiver_t =
        STDEXEC::__t<__initial_receiver<_InitialSenderId, _FinalSenderId, _ReceiverId>>;

      struct __initial_op_t {
        explicit __initial_op_t(
          _InitialSender&& __sndr,
          _FinalSender&& __final,
          __initial_receiver_t __rcvr)
          : __sndr_{static_cast<_FinalSender&&>(__final)}
          , __initial_operation_{STDEXEC::connect(
              static_cast<_InitialSender&&>(__sndr),
              static_cast<__initial_receiver_t&&>(__rcvr))} {
        }

        _FinalSender __sndr_;
        connect_result_t<_InitialSender, __initial_receiver_t> __initial_operation_;
      };

      __variant_for<__initial_op_t, __final_op_t> __op_;
    };

    template <class _InitialSenderId, class _FinalSenderId>
    struct __sender {
      using _InitialSender = STDEXEC::__t<_InitialSenderId>;
      using _FinalSender = STDEXEC::__t<_FinalSenderId>;

      template <class _Self, class _Receiver>
      using __op_t = STDEXEC::__t<
        __operation_state<__cvref_id<_Self, _InitialSender>, __id<_FinalSender>, __id<_Receiver>>
      >;

      struct __t {
        using __id = __sender;
        using sender_concept = sender_t;

        template <__decays_to<__t> _Self, class _Rec>
        STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this _Self&& __self, _Rec&& __receiver) noexcept
          -> __op_t<_Self, _Rec> {
          return __op_t<_Self, _Rec>{
            static_cast<_Self&&>(__self).__initial_sndr_,
            static_cast<_Self&&>(__self).__final_sndr_,
            static_cast<_Rec&&>(__receiver)};
        }
        STDEXEC_EXPLICIT_THIS_END(connect)

        template <__decays_to<__t> _Self, class... _Env>
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
                exec::transform_completion_signatures(
                  __final_completions, exec::ignore_completion()),
                // ... and a set_error(exception_ptr) (TODO: only needed if the
                // values of the initial sender are not nothrow decay copyable).
                completion_signatures<set_error_t(std::exception_ptr)>());
            }
          }
        }

        template <__decays_to<__t> _Self, class... _Env>
        static consteval auto get_completion_signatures() {
          return exec::throw_compile_time_error<
            _SENDER_TYPE_IS_NOT_COPYABLE_,
            _WITH_PRETTY_SENDER_<_FinalSender>
          >();
        }

        _InitialSender __initial_sndr_;
        _FinalSender __final_sndr_;
      };
    };

    template <class _InitialSender, class _FinalSender>
    using __sender_t = __t<__sender<__id<_InitialSender>, __id<_FinalSender>>>;

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
            return __sender_t<_Initial, _Final>{
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
