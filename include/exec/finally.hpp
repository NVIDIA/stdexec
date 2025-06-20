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
#include "../stdexec/__detail/__manual_lifetime.hpp"

namespace exec {
  namespace __final {
    using namespace stdexec;

    template <class _Sigs>
    using __result_variant =
      __for_each_completion_signature<_Sigs, __decayed_std_tuple, __std_variant>;

    template <class _ResultType, class _ReceiverId>
    struct __final_operation_base {
      using _Receiver = __t<_ReceiverId>;

      _Receiver __receiver_{};
      stdexec::__manual_lifetime<_ResultType> __result_{};
    };

    template <class... _Args>
    using __as_rvalues = completion_signatures<set_value_t(__decay_t<_Args>...)>;

    template <class _InitialSender, class _FinalSender, class... _Env>
    using __completion_signatures_t = transform_completion_signatures<
      __completion_signatures_of_t<_InitialSender, _Env...>,
      transform_completion_signatures<
        __completion_signatures_of_t<_FinalSender, _Env...>,
        completion_signatures<set_error_t(std::exception_ptr)>,
        __mconst<completion_signatures<>>::__f
      >, // swallow the final sender's value completions
      __as_rvalues
    >;

    template <class _Receiver>
    struct __applier {
      _Receiver __receiver_;

      template <class _Tag, class... _Args>
      void operator()(_Tag __tag, _Args&&... __args) noexcept {
        __tag(static_cast<_Receiver&&>(__receiver_), static_cast<_Args&&>(__args)...);
      }
    };

    template <class _Receiver>
    struct __visitor {
      _Receiver __receiver_;

      template <class _Tuple>
      void operator()(_Tuple&& __tuple) noexcept {
        std::apply(
          __applier<_Receiver>{static_cast<_Receiver&&>(__receiver_)},
          static_cast<_Tuple&&>(__tuple));
      }
    };

    template <class _ResultType, class _ReceiverId>
    struct __final_receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;

      class __t {
       public:
        using __id = __final_receiver;
        using receiver_concept = stdexec::receiver_t;

        explicit __t(__final_operation_base<_ResultType, _ReceiverId>* __op) noexcept
          : __op_{__op} {
        }

        auto get_env() const noexcept -> env_of_t<_Receiver> {
          return stdexec::get_env(__op_->__receiver_);
        }

        void set_value() noexcept {
          if constexpr (std::is_nothrow_move_constructible_v<_ResultType>) {
            _ResultType __result = static_cast<_ResultType&&>(__op_->__result_.__get());
            __op_->__result_.__destroy();
            std::visit(
              __visitor<_Receiver>{static_cast<_Receiver&&>(__op_->__receiver_)},
              static_cast<_ResultType&&>(__result));
          } else {
            STDEXEC_TRY {
              _ResultType __result = static_cast<_ResultType&&>(__op_->__result_.__get());
              __op_->__result_.__destroy();
              std::visit(
                __visitor<_Receiver>{static_cast<_Receiver&&>(__op_->__receiver_)},
                static_cast<_ResultType&&>(__result));
            }
            STDEXEC_CATCH_ALL {
              stdexec::set_error(
                static_cast<_Receiver&&>(__op_->__receiver_), std::current_exception());
            }
          }
        }

        template <class _Error>
        void set_error(_Error&& __error) noexcept {
          __op_->__result_.__destroy();
          stdexec::set_error(
            static_cast<_Receiver&&>(__op_->__receiver_), static_cast<_Error&&>(__error));
        }

        void set_stopped() noexcept {
          __op_->__result_.__destroy();
          stdexec::set_stopped(static_cast<_Receiver&&>(__op_->__receiver_));
        }

       private:
        __final_operation_base<_ResultType, _ReceiverId>* __op_;
      };
    };

    template <class _InitialSenderId, class _FinalSenderId, class _ReceiverId>
    struct __operation_state {
      using _InitialSender = __cvref_t<_InitialSenderId>;
      using _FinalSender = stdexec::__t<_FinalSenderId>;
      using _Receiver = stdexec::__t<_ReceiverId>;
      using __signatures = completion_signatures_of_t<_InitialSender, env_of_t<_Receiver>>;
      using __base_t = __final_operation_base<__result_variant<__signatures>, _ReceiverId>;
      using __final_receiver_t =
        stdexec::__t<__final_receiver<__result_variant<__signatures>, _ReceiverId>>;
      using __final_op_t = connect_result_t<_FinalSender, __final_receiver_t>;
      class __t;
    };

    template <class _InitialSenderId, class _FinalSenderId, class _ReceiverId>
    struct __initial_receiver {
      using _Receiver = __cvref_t<_ReceiverId>;
      using _FinalSender = stdexec::__t<_FinalSenderId>;

      using __base_op_t =
        stdexec::__t<__operation_state<_InitialSenderId, _FinalSenderId, _ReceiverId>>;

      class __t {
       public:
        using __id = __initial_receiver;
        using receiver_concept = stdexec::receiver_t;

        explicit __t(__base_op_t* __op) noexcept
          : __op_(__op) {
        }

        template <class... _As>
        void set_value(_As&&... __as) noexcept {
          STDEXEC_TRY {
            __op_
              ->__store_result_and_start_next_op(stdexec::set_value, static_cast<_As&&>(__as)...);
          }
          STDEXEC_CATCH_ALL {
            stdexec::set_error(
              static_cast<_Receiver&&>(__op_->__receiver_), std::current_exception());
          }
        }

        template <class _Error>
        void set_error(_Error&& __err) noexcept {
          STDEXEC_TRY {
            __op_
              ->__store_result_and_start_next_op(stdexec::set_error, static_cast<_Error&&>(__err));
          }
          STDEXEC_CATCH_ALL {
            stdexec::set_error(
              static_cast<_Receiver&&>(__op_->__receiver_), std::current_exception());
          }
        }

        void set_stopped() noexcept {
          STDEXEC_TRY {
            __op_->__store_result_and_start_next_op(stdexec::set_stopped);
          }
          STDEXEC_CATCH_ALL {
            stdexec::set_error(
              static_cast<_Receiver&&>(__op_->__receiver_), std::current_exception());
          }
        }

        auto get_env() const noexcept -> env_of_t<_Receiver> {
          return stdexec::get_env(__op_->__receiver_);
        }

       private:
        __base_op_t* __op_;
      };
    };

    template <class _InitialSenderId, class _FinalSenderId, class _ReceiverId>
    class __operation_state<_InitialSenderId, _FinalSenderId, _ReceiverId>::__t : public __base_t {
      using __initial_receiver_t =
        stdexec::__t<__initial_receiver<_InitialSenderId, _FinalSenderId, _ReceiverId>>;

      struct __initial_op_t {
        _FinalSender __sndr_;
        connect_result_t<_InitialSender, __initial_receiver_t> __initial_operation_;
      };

      std::variant<__initial_op_t, __final_op_t> __op_;

     public:
      using __id = __operation_state;

      template <class... _Args>
        requires constructible_from<__result_variant<__signatures>, __decayed_std_tuple<_Args...>>
      void __store_result_and_start_next_op(_Args&&... __args) {
        this->__result_.__construct(
          std::in_place_type<__decayed_std_tuple<_Args...>>, static_cast<_Args&&>(__args)...);
        STDEXEC_ASSERT(__op_.index() == 0);
        auto __final = static_cast<_FinalSender&&>(std::get_if<0>(&__op_)->__sndr_);
        __final_op_t& __final_op = __op_.template emplace<1>(__emplace_from{[&] {
          return stdexec::connect(static_cast<_FinalSender&&>(__final), __final_receiver_t{this});
        }});
        stdexec::start(__final_op);
      }

      __t(_InitialSender&& __initial, _FinalSender __final, _Receiver __receiver)
        : __base_t{{static_cast<_Receiver&&>(__receiver)}}
        , __op_(std::in_place_index<0>, __emplace_from{[&] {
                  return __initial_op_t{
                    static_cast<_FinalSender&&>(__final),
                    stdexec::connect(
                      static_cast<_InitialSender&&>(__initial), __initial_receiver_t{this})};
                }}) {
      }

      void start() & noexcept {
        STDEXEC_ASSERT(__op_.index() == 0);
        stdexec::start(std::get_if<0>(&__op_)->__initial_operation_);
      }
    };

    template <class _InitialSenderId, class _FinalSenderId>
    struct __sender {
      using _InitialSender = stdexec::__t<_InitialSenderId>;
      using _FinalSender = stdexec::__t<_FinalSenderId>;

      template <class _Self, class _Receiver>
      using __op_t = stdexec::__t<
        __operation_state<__cvref_id<_Self, _InitialSender>, __id<_FinalSender>, __id<_Receiver>>
      >;

      class __t {
        _InitialSender __initial_sndr_;
        _FinalSender __final_sndr_;

       public:
        using __id = __sender;
        using sender_concept = stdexec::sender_t;

        template <__decays_to<_InitialSender> _Initial, __decays_to<_FinalSender> _Final>
        __t(_Initial&& __initial, _Final&& __final)
          noexcept(__nothrow_decay_copyable<_Initial> && __nothrow_decay_copyable<_Final>)
          : __initial_sndr_{static_cast<_Initial&&>(__initial)}
          , __final_sndr_{static_cast<_Final&&>(__final)} {
        }

        template <__decays_to<__t> _Self, class _Rec>
        static auto connect(_Self&& __self, _Rec&& __receiver) noexcept -> __op_t<_Self, _Rec> {
          return {
            static_cast<_Self&&>(__self).__initial_sndr_,
            static_cast<_Self&&>(__self).__final_sndr_,
            static_cast<_Rec&&>(__receiver)};
        }

        template <__decays_to<__t> _Self, class... _Env>
        static auto get_completion_signatures(_Self&&, _Env&&...) noexcept
          -> __completion_signatures_t<__copy_cvref_t<_Self, _InitialSender>, _FinalSender, _Env...> {
          return {};
        }

        template <__decays_to<__t> _Self, class... _Env>
          requires(!__decay_copyable<__copy_cvref_t<_Self, _FinalSender>>)
        static auto get_completion_signatures(_Self&&, _Env&&...) noexcept {
          return _ERROR_<_SENDER_TYPE_IS_NOT_COPYABLE_, _WITH_SENDER_<_FinalSender>>{};
        }
      };
    };

    struct finally_t {
      template <sender _Initial, sender _Final>
        requires __has_common_domain<_Initial, _Final>
      auto operator()(_Initial&& __initial, _Final&& __final) const {
        using _Domain = __common_domain_t<_Initial, _Final>;
        return stdexec::transform_sender(
          _Domain(),
          __make_sexpr<finally_t>(
            {}, static_cast<_Initial&&>(__initial), static_cast<_Final&&>(__final)));
      }

      template <sender _Final>
      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()(_Final&& __final) const -> __binder_back<finally_t, __decay_t<_Final>> {
        return {{static_cast<_Final&&>(__final)}, {}, {}};
      }

      template <class _Sender>
      static auto transform_sender(_Sender&& __sndr, __ignore) {
        return __sexpr_apply(
          static_cast<_Sender&&>(__sndr),
          []<class _Initial, class _Final>(
            __ignore, __ignore, _Initial&& __initial, _Final&& __final) {
            using __result_sndr_t =
              __t<__sender<__id<__decay_t<_Initial>>, __id<__decay_t<_Final>>>>;
            return __result_sndr_t{
              static_cast<_Initial&&>(__initial), static_cast<_Final&&>(__final)};
          });
      }
    };
  } // namespace __final

  using __final::finally_t;
  inline constexpr __final::finally_t finally{};
} // namespace exec

namespace stdexec {
  template <>
  struct __sexpr_impl<exec::finally_t> : __sexpr_defaults {
    static constexpr auto get_completion_signatures = []<class _Sender>(_Sender&&) noexcept
      -> __completion_signatures_of_t<transform_sender_result_t<default_domain, _Sender, env<>>> {
    };
  };
} // namespace stdexec