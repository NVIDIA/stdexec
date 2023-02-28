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
#include "any_sender_of.hpp"

namespace exec {
  namespace __finally {
    using namespace stdexec;

    template <class _Arg, class... _Args>
    using __nonempty_decayed_tuple = __decayed_tuple<_Arg, _Args...>;

    template <class _Sigs>
    using __result_type_ =
      __gather_signal<set_value_t, _Sigs, __q<__nonempty_decayed_tuple>, __q<__variant>>;

    template <class _Sigs>
    using __result_type =
      __if<std::is_same<__result_type_<_Sigs>, __not_a_variant>, __ignore, __result_type_<_Sigs>>;

    template <class _ResultType, class _ReceiverId>
    struct __final_operation_base {
      using _Receiver = __t<_ReceiverId>;

      _Receiver __receiver_{};
      __manual_lifetime<_ResultType> __result_{};
    };

    namespace __rec {
      template <class _Sig>
      struct __rcvr_vfun;

      template <class _Sigs>
      struct __vtable {
        struct __t;
      };

      template <class _Sigs, class _Env>
      struct __ref;

      template <class _Tag, class... _As>
      struct __rcvr_vfun<_Tag(_As...)> {
        void (*__fn_)(void*, _As...) noexcept;
      };

      template <class _Rcvr>
      struct __rcvr_vfun_fn {
        template <class _Tag, class... _As>
        constexpr void (*operator()(_Tag (*)(_As...)) const noexcept)(void*, _As...) noexcept {
          return +[](void* __rcvr, _As... __as) noexcept -> void {
            _Tag{}((_Rcvr&&) *(_Rcvr*) __rcvr, (_As&&) __as...);
          };
        }
      };

      template <class... _Sigs>
      struct __vtable<completion_signatures<_Sigs...>> {
        struct __t : __rcvr_vfun<_Sigs>... {
          template <class _Rcvr>
            requires receiver_of<_Rcvr, completion_signatures<_Sigs...>>
          friend const __t*
            tag_invoke(__any::__create_vtable_t, __mtype<__t>, __mtype<_Rcvr>) noexcept {
            static const __t __vtable_{{__rcvr_vfun_fn<_Rcvr>{}((_Sigs*) nullptr)}...};
            return &__vtable_;
          }
        };
      };

      template <class... _Sigs, class _Env>
      struct __ref<completion_signatures<_Sigs...>, _Env> {
        using __vtable_t = stdexec::__t<__vtable<completion_signatures<_Sigs...>>>;

        class __t {
          void* __receiver_;
          _Env __env_;
          const __vtable_t* __vtable_;

          template <std::same_as<__t> _Self>
          friend _Env tag_invoke(get_env_t, const _Self& __self) noexcept {
            return __self.__env_;
          }

          template <__completion_tag _Tag, __decays_to<__t> _Self, class... _As>
            requires __one_of<_Tag(_As...), _Sigs...>
          friend void tag_invoke(_Tag, _Self&& __self, _As&&... __as) noexcept {
            (*static_cast<const __rcvr_vfun<_Tag(_As...)>*>(__self.__vtable_)->__fn_)(
              __self.__receiver_, (_As&&) __as...);
          }

         public:
          template <__none_of<__t, const __t> _Rcvr>
            requires receiver_of<_Rcvr, completion_signatures<_Sigs...>>
          __t(_Rcvr& __rcvr) noexcept
            : __receiver_{std::addressof(__rcvr)}
            , __env_{get_env(__rcvr)}
            , __vtable_{__any::__create_vtable(__mtype<__vtable_t>{}, __mtype<_Rcvr>{})} {
          }
        };
      };
    } // __rec

    template <class _Sigs, class __ReceiverId>
    using __receiver_ref = __t<__rec::__ref<_Sigs, env_of_t<__t<__ReceiverId>>>>;

    template <class... _Args>
    using __all_nothrow_decay_copyable = __bool<(__nothrow_decay_copyable<_Args> && ...)>;

    template <class _Sender, class _Receiver>
    static const bool __is_nothrow_connectable_v =
      noexcept(connect(std::declval<_Sender>(), std::declval<_Receiver>()));

    template <class _Sender, class _Receiver, class... _Args>
    using __nothrow_store_and_connect = __mand<
      __all_nothrow_decay_copyable<_Args...>,
      __bool<__nothrow_connectable<_Sender, _Receiver>>>;

    template <class _Sender, class _Receiver, class _Sigs>
    using __nothrow_store_and_connect_sigs = __mand<
      __gather_signal<set_value_t, _Sigs, __q<__all_nothrow_decay_copyable>, __q<__mand>>,
      __bool<__nothrow_connectable<_Sender, _Receiver>>>;

    template <class _Sender, class _Receiver, class... _Args>
    inline constexpr bool __nothrow_store_and_connect_v =
      __v<__nothrow_store_and_connect<_Sender, _Receiver, _Args...>>;

    template <class... _Args>
    using __as_rvalues = completion_signatures<set_value_t(decay_t<_Args> && ...)>;

    template <class... _Errors>
    using __error_sigs = completion_signatures<set_error_t(_Errors)...>;

    template <class _Sigs, class _FinalSender, class _Env>
    using __additional_error_types = __concat_completion_signatures_t<
      error_types_of_t<_FinalSender, _Env, __error_sigs>,
      __if<
        __nothrow_store_and_connect_sigs<
          _FinalSender,
          __t<__rec::__ref<completion_signatures_of_t<_FinalSender>, _Env>>,
          _Sigs>,
        completion_signatures<>,
        completion_signatures<set_error_t(std::exception_ptr)>>>;

    template <class _InitialSender, class _FinalSender, class _Env>
    using __completion_signatures_t = make_completion_signatures<
      _InitialSender,
      _Env,
      __additional_error_types<completion_signatures_of_t<_InitialSender>, _FinalSender, _Env>,
      __as_rvalues>;

    template <class _ResultType, class _ReceiverId>
    struct __final_receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;

      class __t {
       public:
        explicit __t(__final_operation_base<_ResultType, _ReceiverId>* __op) noexcept
          : __op_{__op} {
        }

       private:
        __final_operation_base<_ResultType, _ReceiverId>* __op_;

        template <__decays_to<__t> _Self>
        friend void tag_invoke(set_value_t, _Self&& __self) noexcept {
          if constexpr (!std::same_as<_ResultType, __ignore>) {
            std::visit(
              [&]<class _Tuple>(_Tuple&& __tuple) noexcept {
                std::apply(
                  [&]<class... _Args>(_Args&&... __args) noexcept {
                    set_value((_Receiver&&) __self.__op_->__receiver_, (_Args&&) __args...);
                  },
                  (_Tuple&&) __tuple);
              },
              (_ResultType&&) __self.__op_->__result_);
          } else {
            set_value((_Receiver&&) __self.__op_->__receiver_);
          }
          __self.__op_->__result_.__destruct();
        }

        template <__one_of<set_error_t, set_stopped_t> _Tag, __decays_to<__t> _Self, class... _Error>
          requires __callable<_Tag, _Receiver&&, _Error...>
        friend void tag_invoke(_Tag __tag, _Self&& __self, _Error&&... __error) noexcept {
          __self.__op_->__result_.__destruct();
          __tag((_Receiver&&) __self.__op_->__receiver_, (_Error&&) __error...);
        }

        template <std::same_as<__t> _Self>
        friend env_of_t<_Receiver> tag_invoke(get_env_t, const _Self& __self) noexcept {
          return get_env(__self.__op_->__receiver_);
        }
      };
    };

    template <class _FinalSenderId, class _Sigs, class _ReceiverId>
    struct __initial_operation_base : __final_operation_base<__result_type<_Sigs>, _ReceiverId> {
      using _FinalSender = __t<_FinalSenderId>;
      using __final_receiver_ref_t =
        __receiver_ref<completion_signatures_of_t<_FinalSender>, _ReceiverId>;
      using __final_receiver_t = __t<__final_receiver<__result_type<_Sigs>, _ReceiverId>>;
      using __final_operation_t = connect_result_t<_FinalSender, __final_receiver_ref_t>;

      _FinalSender __final_sender_;
      __final_receiver_t __final_receiver_{this};
      __manual_lifetime<__final_operation_t> __final_operation_{};

      template <class... _Args>
        requires std::is_constructible_v<__result_type<_Sigs>, __decayed_tuple<_Args...>>
      void __store_result_and_start_next_op(_Args&&... __args) noexcept(
        __nothrow_store_and_connect_v<_FinalSender, __final_receiver_ref_t, _Args...>) {
        this->__result_.__construct(std::tuple{(_Args&&) __args...});
        __final_operation_.__construct(__conv{[&] {
          return connect(
            (_FinalSender&&) __final_sender_, __final_receiver_ref_t{__final_receiver_});
        }});
        start(__final_operation_.__get());
      }
    };

    template <class _FinalSenderId, class _Sigs, class _ReceiverId>
    struct __initial_receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;
      using _FinalSender = stdexec::__t<_FinalSenderId>;

      using __final_receiver_ref_t =
        __receiver_ref<completion_signatures_of_t<_FinalSender>, _ReceiverId>;

      class __t {
       public:
        explicit __t(__initial_operation_base<_FinalSenderId, _Sigs, _ReceiverId>* __op) noexcept
          : __op_(__op) {
        }

       private:
        __initial_operation_base<_FinalSenderId, _Sigs, _ReceiverId>* __op_;

        template <__decays_to<__t> _Self, class... _Args>
          requires __callable<set_value_t, _Receiver&&, _Args...>
        friend void tag_invoke(set_value_t, _Self&& __self, _Args&&... __args) noexcept {
          if constexpr (
            __nothrow_store_and_connect_v<_FinalSender, __final_receiver_ref_t, _Args...>) {
            __self.__op_->__store_result_and_start_next_op((_Args&&) __args...);
          } else {
            try {
              __self.__op_->__store_result_and_start_next_op((_Args&&) __args...);
            } catch (...) {
              set_error((_Receiver&&) __self.__op_->__receiver_, std::current_exception());
            }
          }
        }

        template <__one_of<set_error_t, set_stopped_t> _Tag, __decays_to<__t> _Self, class... _Error>
          requires __callable<_Tag, _Receiver&&, _Error...>
        friend void tag_invoke(_Tag __tag, _Self&& __self, _Error&&... __error) noexcept {
          __tag((_Receiver&&) __self.__op_->__receiver_, (_Error&&) __error...);
        }

        template <std::same_as<__t> _Self>
        friend env_of_t<_Receiver> tag_invoke(get_env_t, const _Self& __self) noexcept {
          return get_env(__self.__op_->__receiver_);
        }
      };
    };

    template <class _InitialSenderId, class _FinalSenderId, class _ReceiverId>
    struct operation_state {
      using _InitialSender = stdexec::__t<_InitialSenderId>;
      using _FinalSender = stdexec::__t<_FinalSenderId>;
      using _Receiver = stdexec::__t<_ReceiverId>;
      using _Sigs = completion_signatures_of_t<_InitialSender>;
      using __initial_receiver_t =
        stdexec::__t<__initial_receiver<_FinalSenderId, _Sigs, _ReceiverId>>;

      using __base_t = __initial_operation_base<_FinalSenderId, _Sigs, _ReceiverId>;

      class __t : __base_t {
        connect_result_t<_InitialSender, __initial_receiver_t> __initial_operation_;

        template <std::same_as<__t> _Self>
        friend void tag_invoke(start_t, _Self& __self) noexcept {
          start(__self.__initial_operation_);
        }

       public:
        __t(_InitialSender&& __initial_sender, _FinalSender&& __final_sender, _Receiver __receiver)
          : __base_t{{(_Receiver&&) __receiver}, (_FinalSender&&) __final_sender}
          , __initial_operation_(
              connect((_InitialSender&&) __initial_sender, __initial_receiver_t{this})) {
        }
      };
    };

    template <class _InitialSenderId, class _FinalSenderId>
    struct __sender {
      using _InitialSender = stdexec::__t<_InitialSenderId>;
      using _FinalSender = stdexec::__t<_FinalSenderId>;

      template <class _Self, class _Receiver>
      using __op_t = stdexec::__t<operation_state<
        __x<__copy_cvref_t<_Self, _InitialSender>>,
        __x<__copy_cvref_t<_Self, _FinalSender>>,
        __x<_Receiver>>>;

      class __t {
        _InitialSender __initial_sender_;
        _FinalSender __final_sender_;

        template <__decays_to<__t> _Self, class _R>
          requires receiver_of<
            _R,
            __completion_signatures_t<_InitialSender, _FinalSender, env_of_t<_R>>>
        friend __op_t< _Self, _R> tag_invoke(connect_t, _Self&& __self, _R&& __receiver) noexcept {
          return {
            ((_Self&&) __self).__initial_sender_,
            ((_Self&&) __self).__final_sender_,
            (_R&&) __receiver};
        }

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env) noexcept
          -> dependent_completion_signatures<_Env>;

        template <__decays_to<__t> _Self, __none_of<no_env> _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env) noexcept
          -> __completion_signatures_t<
            __copy_cvref_t<_Self, _InitialSender>,
            __copy_cvref_t<_Self, _FinalSender>,
            _Env>;

       public:
        using is_sender = void;

        template <__decays_to<_InitialSender> _I, __decays_to<_FinalSender> _F>
        __t(_I&& __initial_sender, _F&& __final_sender) noexcept(
          __nothrow_decay_copyable<_I>&& __nothrow_decay_copyable<_F>)
          : __initial_sender_{(_I&&) __initial_sender}
          , __final_sender_{(_F&&) __final_sender} {
        }
      };
    };

    struct __finally_t {
      template <sender _I, sender _F>
      __t<__sender<__id<decay_t<_I>>, __id<decay_t<_F>>>>
        operator()(_I&& __initial_sender, _F&& __final_sender) const
        noexcept(__nothrow_decay_copyable<_I>&& __nothrow_decay_copyable<_F>) {
        return {(_I&&) __initial_sender, (_F&&) __final_sender};
      }
    };
  }

  inline constexpr __finally ::__finally_t finally{};
}