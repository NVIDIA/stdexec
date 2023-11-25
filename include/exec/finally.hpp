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

#include "__detail/__manual_lifetime.hpp"
#include "materialize.hpp"

namespace exec {
  namespace __finally_ {
    using namespace stdexec;

    template <class _Arg, class... _Args>
    using __nonempty_decayed_tuple = __decayed_tuple<_Arg, _Args...>;

    template <class _Sigs>
    using __value_types_ = __gather_signal<
      set_value_t,
      _Sigs,
      __mbind_front_q<__decayed_tuple, set_value_t>,
      __q<__types>>;

    template <class _Sigs>
    using __error_types_ = __gather_signal<
      set_error_t,
      _Sigs,
      __mbind_front_q<__decayed_tuple, set_error_t>,
      __q<__types>>;

    template <class _Sigs>
    using __stopped_types_ = __gather_signal<
      set_stopped_t,
      _Sigs,
      __mbind_front_q<__decayed_tuple, set_stopped_t>,
      __q<__types>>;

    template <class _Sigs>
    using __result_variant = __minvoke<
      __mconcat<__q<__variant>>,
      __value_types_<_Sigs>,
      __error_types_<_Sigs>,
      __stopped_types_<_Sigs>>;

    template <class _ResultType, class _ReceiverId>
    struct __final_operation_base {
      using _Receiver = __t<_ReceiverId>;

      _Receiver __receiver_{};
      __manual_lifetime<_ResultType> __result_{};
    };

    template <class... _Args>
    using __as_rvalues = completion_signatures<set_value_t(__decay_t<_Args>&&...)>;

    template <class _InitialSender, class _FinalSender, class _Env>
    using __completion_signatures_t = make_completion_signatures<
      _InitialSender,
      _Env,
      completion_signatures<set_error_t(std::exception_ptr)>,
      __as_rvalues>;

    template <class _Receiver>
    struct __applier {
      _Receiver __receiver_;

      template <class _Tag, class... _Args>
      void operator()(_Tag __tag, _Args&&... __args) const noexcept {
        __tag((_Receiver&&) __receiver_, (_Args&&) __args...);
      }
    };

    template <class _Receiver>
    struct __visitor {
      _Receiver __receiver_;

      template <class _Tuple>
      void operator()(_Tuple&& __tuple) const noexcept {
        std::apply(__applier<_Receiver>{(_Receiver&&) __receiver_}, (_Tuple&&) __tuple);
      }
    };

    template <class _ResultType, class _ReceiverId>
    struct __final_receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;

      class __t {
       public:
        using receiver_concept = stdexec::receiver_t;

        explicit __t(__final_operation_base<_ResultType, _ReceiverId>* __op) noexcept
          : __op_{__op} {
        }

       private:
        __final_operation_base<_ResultType, _ReceiverId>* __op_;

        template <same_as<set_value_t> _Tag, __decays_to<__t> _Self>
        friend void tag_invoke(_Tag, _Self&& __self) noexcept {
          if constexpr (std::is_nothrow_move_constructible_v<_ResultType>) {
            _ResultType __result = (_ResultType&&) __self.__op_->__result_;
            __self.__op_->__result_.__destroy();
            std::visit(
              __visitor<_Receiver>{(_Receiver&&) __self.__op_->__receiver_},
              (_ResultType&&) __result);
          } else {
            try {
              _ResultType __result = (_ResultType&&) __self.__op_->__result_;
              __self.__op_->__result_.__destroy();
              std::visit(
                __visitor<_Receiver>{(_Receiver&&) __self.__op_->__receiver_},
                (_ResultType&&) __result);
            } catch (...) {
              stdexec::set_error((_Receiver&&) __self.__op_->__receiver_, std::current_exception());
            }
          }
        }

        template <__one_of<set_error_t, set_stopped_t> _Tag, __decays_to<__t> _Self, class... _Error>
          requires __callable<_Tag, _Receiver&&, _Error...>
        friend void tag_invoke(_Tag __tag, _Self&& __self, _Error&&... __error) noexcept {
          __self.__op_->__result_.__destroy();
          __tag((_Receiver&&) __self.__op_->__receiver_, (_Error&&) __error...);
        }

        template <std::same_as<__t> _Self>
        friend env_of_t<_Receiver> tag_invoke(get_env_t, const _Self& __self) noexcept {
          return get_env(__self.__op_->__receiver_);
        }
      };
    };

    template <class _InitialSenderId, class _FinalSenderId, class _ReceiverId>
    struct __operation_state {
      using _InitialSender = stdexec::__t<_InitialSenderId>;
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
      using _Receiver = stdexec::__t<_ReceiverId>;
      using _FinalSender = stdexec::__t<_FinalSenderId>;

      using __base_op_t =
        stdexec::__t<__operation_state<_InitialSenderId, _FinalSenderId, _ReceiverId>>;

      class __t {
       public:
        using receiver_concept = stdexec::receiver_t;

        explicit __t(__base_op_t* __op) noexcept
          : __op_(__op) {
        }

       private:
        __base_op_t* __op_;

        template <class _Tag, __decays_to<__t> _Self, class... _Args>
          requires __callable<_Tag, _Receiver&&, _Args...>
        friend void tag_invoke(_Tag __tag, _Self&& __self, _Args&&... __args) noexcept {
          try {
            __self.__op_->__store_result_and_start_next_op(__tag, (_Args&&) __args...);
          } catch (...) {
            set_error((_Receiver&&) __self.__op_->__receiver_, std::current_exception());
          }
        }

        template <std::same_as<__t> _Self>
        friend env_of_t<_Receiver> tag_invoke(get_env_t, const _Self& __self) noexcept {
          return get_env(__self.__op_->__receiver_);
        }
      };
    };

    template <class _InitialSenderId, class _FinalSenderId, class _ReceiverId>
    class __operation_state<_InitialSenderId, _FinalSenderId, _ReceiverId>::__t : public __base_t {
      using __initial_receiver_t =
        stdexec::__t<__initial_receiver<_InitialSenderId, _FinalSenderId, _ReceiverId>>;

      struct __initial_op_t {
        _FinalSender __sender_;
        connect_result_t<_InitialSender, __initial_receiver_t> __initial_operation_;
      };

      std::variant<__initial_op_t, __final_op_t> __op_;

      template <std::same_as<__t> _Self>
      friend void tag_invoke(start_t, _Self& __self) noexcept {
        STDEXEC_ASSERT(__self.__op_.index() == 0);
        start(std::get_if<0>(&__self.__op_)->__initial_operation_);
      }

     public:
      template <class... _Args>
        requires std::is_constructible_v<__result_variant<__signatures>, __decayed_tuple<_Args...>>
      void __store_result_and_start_next_op(_Args&&... __args) {
        this->__result_.__construct(
          std::in_place_type<__decayed_tuple<_Args...>>, (_Args&&) __args...);
        STDEXEC_ASSERT(__op_.index() == 0);
        _FinalSender __final_sender = (_FinalSender&&) std::get_if<0>(&__op_)->__sender_;
        __final_op_t& __final_op = __op_.template emplace<1>(__conv{[&] {
          return stdexec::connect((_FinalSender&&) __final_sender, __final_receiver_t{this});
        }});
        start(__final_op);
      }

      __t(_InitialSender&& __initial_sender, _FinalSender&& __final_sender, _Receiver __receiver)
        : __base_t{{(_Receiver&&) __receiver}}
        , __op_(std::in_place_index<0>, __conv{[&] {
                  return __initial_op_t{
                    (_FinalSender&&) __final_sender,
                    stdexec::connect(
                      (_InitialSender&&) __initial_sender, __initial_receiver_t{this})};
                }}) {
      }
    };

    template <class _InitialSenderId, class _FinalSenderId>
    struct __sender {
      using _InitialSender = stdexec::__t<_InitialSenderId>;
      using _FinalSender = stdexec::__t<_FinalSenderId>;

      template <class _Self, class _Receiver>
      using __op_t = stdexec::__t<__operation_state<
        __x<__copy_cvref_t<_Self, _InitialSender>>,
        __x<__copy_cvref_t<_Self, _FinalSender>>,
        __x<_Receiver>>>;

      class __t {
        _InitialSender __initial_sender_;
        _FinalSender __final_sender_;

        template <__decays_to<__t> _Self, class _Rec>
          requires receiver_of<
            _Rec,
            __completion_signatures_t<_InitialSender, _FinalSender, env_of_t<_Rec>>>
        friend __op_t< _Self, _Rec>
          tag_invoke(connect_t, _Self&& __self, _Rec&& __receiver) noexcept {
          return {
            ((_Self&&) __self).__initial_sender_,
            ((_Self&&) __self).__final_sender_,
            (_Rec&&) __receiver};
        }

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env&&) noexcept
          -> __completion_signatures_t<
            __copy_cvref_t<_Self, _InitialSender>,
            __copy_cvref_t<_Self, _FinalSender>,
            _Env> {
          return {};
        }

       public:
        using sender_concept = stdexec::sender_t;

        template <__decays_to<_InitialSender> _Is, __decays_to<_FinalSender> _Fs>
        __t(_Is&& __initial_sender, _Fs&& __final_sender) noexcept(
          __nothrow_decay_copyable<_Is>&& __nothrow_decay_copyable<_Fs>)
          : __initial_sender_{(_Is&&) __initial_sender}
          , __final_sender_{(_Fs&&) __final_sender} {
        }
      };
    };

    struct __finally_t {
      template <sender _Is, sender _Fs>
      __t<__sender<__id<__decay_t<_Is>>, __id<__decay_t<_Fs>>>>
        operator()(_Is&& __initial_sender, _Fs&& __final_sender) const
        noexcept(__nothrow_decay_copyable<_Is>&& __nothrow_decay_copyable<_Fs>) {
        return {(_Is&&) __initial_sender, (_Fs&&) __final_sender};
      }
    };
  }

  inline constexpr __finally_ ::__finally_t finally{};
}
