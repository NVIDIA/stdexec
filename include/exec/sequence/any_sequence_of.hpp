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

#include "../any_sender_of.hpp"

namespace exec {
  namespace __any {
    namespace __next {
      template <__valid_completion_signatures _Sigs>
      struct __rcvr_next_vfun {
        using __return_sigs = completion_signatures<set_value_t(), set_stopped_t()>;
        using __void_sender = typename any_receiver_ref<__return_sigs>::template any_sender<>;
        using __item_sender = typename any_receiver_ref<_Sigs>::template any_sender<>;
        __void_sender (*__fn_)(void*, __item_sender&&);
      };

      template <class _Rcvr>
      struct __rcvr_next_vfun_fn {
        using __return_sigs = completion_signatures<set_value_t(), set_stopped_t()>;
        using __void_sender = typename any_receiver_ref<__return_sigs>::template any_sender<>;

        template <class _Sigs>
        using __item_sender = typename any_receiver_ref<_Sigs>::template any_sender<>;

        template <__valid_completion_signatures _Sigs>
        constexpr __void_sender (*operator()(_Sigs*) const)(void*, __item_sender<_Sigs>&&) {
          return +[](void* __r, __item_sender<_Sigs>&& __sndr) noexcept -> __void_sender {
            return __void_sender{
              set_next(*static_cast<_Rcvr*>(__r), static_cast<__item_sender<_Sigs>&&>(__sndr))};
          };
        }
      };

      template <class _NextSigs, class _Sigs, class... _Queries>
      struct __next_vtable;

      template <class _NextSigs, class... _Sigs, class... _Queries>
      struct __next_vtable<_NextSigs, completion_signatures<_Sigs...>, _Queries...> {
        using __item_sender = typename any_receiver_ref<_NextSigs>::template any_sender<>;
        using __item_types = item_types<__item_sender>;

        struct __t
          : public __rcvr_next_vfun<_NextSigs>
          , public __rec::__rcvr_vfun<_Sigs>...
          , public __query_vfun<_Queries>... {
          using __id = __next_vtable;
          using __query_vfun<_Queries>::operator()...;

          template <class _Rcvr>
            requires sequence_receiver_of<_Rcvr, __item_types>
                  && (__callable<__query_vfun_fn<_Rcvr>, _Queries> && ...)
          friend const __t* tag_invoke(__create_vtable_t, __mtype<__t>, __mtype<_Rcvr>) noexcept {
            static const __t __vtable_{
              {__rcvr_next_vfun_fn<_Rcvr>{}((_NextSigs*) nullptr)},
              {__rec::__rcvr_vfun_fn<_Rcvr>{}((_Sigs*) nullptr)}...,
              {__query_vfun_fn<_Rcvr>{}((_Queries) nullptr)}...};
            return &__vtable_;
          }
        };
      };

      template <class _Sigs, class... _Queries>
      struct __env {
        using __compl_sigs = __to_sequence_completions_t<_Sigs>;

        using __vtable_t = stdexec::__t<__next_vtable<_Sigs, __compl_sigs, _Queries...>>;

        struct __t {
          using __id = __env;
          const __vtable_t* __vtable_;
          void* __rcvr_;

          template <class _Tag, same_as<__t> _Self, class... _As>
            requires __callable<const __vtable_t&, _Tag, void*, _As...>
          friend auto tag_invoke(_Tag, const _Self& __self, _As&&... __as) noexcept(
            __nothrow_callable<const __vtable_t&, _Tag, void*, _As...>)
            -> __call_result_t<const __vtable_t&, _Tag, void*, _As...> {
            return (*__self.__vtable_)(_Tag{}, __self.__rcvr_, (_As&&) __as...);
          }
        };
      };

      template <class _Sigs, class... _Queries>
      struct __receiver_ref;

      template <class... _Sigs, class... _Queries>
      struct __receiver_ref<completion_signatures<_Sigs...>, _Queries...> {
        struct __t {
          using __return_sigs = completion_signatures<set_value_t(), set_stopped_t()>;
          using __void_sender = typename any_receiver_ref<__return_sigs>::template any_sender<>;
          using __next_sigs = completion_signatures<_Sigs...>;
          using __compl_sigs = __to_sequence_completions_t<__next_sigs>;
          using __item_sender = typename any_receiver_ref<__next_sigs>::template any_sender<>;
          using __item_types = item_types<__item_sender>;

          using __vtable_t = stdexec::__t<__next_vtable<__next_sigs, __compl_sigs, _Queries...>>;

          template <class Sig>
          using __vfun = __rec::__rcvr_vfun<Sig>;

          using __env_t = stdexec::__t<__env<__next_sigs, _Queries...>>;
          __env_t __env_;

          using receiver_concept = stdexec::receiver_t;

          template <__none_of<__t, const __t, __env_t, const __env_t> _Rcvr>
            requires sequence_receiver_of<_Rcvr, __item_types>
                  && (__callable<__query_vfun_fn<_Rcvr>, _Queries> && ...)
          __t(_Rcvr& __rcvr) noexcept
            : __env_{__create_vtable(__mtype<__vtable_t>{}, __mtype<_Rcvr>{}), &__rcvr} {
          }

          template <same_as<set_next_t> _SetNext, same_as<__t> _Self, class _Sender>
            requires constructible_from<__item_sender, _Sender>
          friend __void_sender tag_invoke(_SetNext, _Self& __self, _Sender&& __sndr) {
            return (
              *static_cast<const __rcvr_next_vfun<__next_sigs>*>(__self.__env_.__vtable_)->__fn_)(
              __self.__env_.__rcvr_, static_cast<_Sender&&>(__sndr));
          }

          template <same_as<set_value_t> _SetValue, same_as<__t> _Self>
          // set_value_t() is always valid for a sequence
          friend void tag_invoke(_SetValue, _Self&& __self) noexcept {
            (*static_cast<const __vfun<_SetValue()>*>(__self.__env_.__vtable_)->__fn_)(
              __self.__env_.__rcvr_);
          }

          template <same_as<set_error_t> _SetError, same_as<__t> _Self, class Error>
            requires __v< __mapply<__contains<set_error_t(Error)>, __compl_sigs>>
          friend void tag_invoke(_SetError, _Self&& __self, Error&& __error) noexcept {
            (*static_cast<const __vfun<set_error_t(Error)>*>(__self.__env_.__vtable_)->__fn_)(
              __self.__env_.__rcvr_, static_cast<Error&&>(__error));
          }

          template <same_as<set_stopped_t> _SetStopped, same_as<__t> _Self>
            requires __v< __mapply<__contains<set_stopped_t()>, __compl_sigs>>
          friend void tag_invoke(_SetStopped, _Self&& __self) noexcept

          {
            (*static_cast<const __vfun<set_stopped_t()>*>(__self.__env_.__vtable_)->__fn_)(
              __self.__env_.__rcvr_);
          }

          template <same_as<get_env_t> _GetEnv, __decays_to<__t> _Self>
          friend const __env_t& tag_invoke(_GetEnv, _Self&& __self) noexcept {
            return __self.__env_;
          }
        };
      };
    }

    template <class _Sigs, class _Queries>
    using __next_receiver_ref =
      __mapply<__mbind_front<__q<__next::__receiver_ref>, _Sigs>, _Queries>;

    template <class _Sigs, class _SenderQueries, class _ReceiverQueries>
    struct __sender_vtable {
      using __query_vtable_t = __query_vtable<_SenderQueries>;
      using __receiver_ref_t = stdexec::__t<__next_receiver_ref<_Sigs, _ReceiverQueries>>;

      struct __t : public __query_vtable_t {
        const __query_vtable_t& queries() const noexcept {
          return *this;
        }

        __immovable_operation_storage (*subscribe_)(void*, __receiver_ref_t);

        template <class _Sender>
          requires sequence_sender_to<_Sender, __receiver_ref_t>
        friend const __t* tag_invoke(__create_vtable_t, __mtype<__t>, __mtype<_Sender>) noexcept {
          static const __t __vtable_{
            {*__create_vtable(__mtype<__query_vtable_t>{}, __mtype<_Sender>{})},
            [](void* __object_pointer, __receiver_ref_t __receiver)
              -> __immovable_operation_storage {
              _Sender& __sender = *static_cast<_Sender*>(__object_pointer);
              using __op_state_t = subscribe_result_t<_Sender, __receiver_ref_t>;
              return __immovable_operation_storage{
                std::in_place_type<__op_state_t>, __conv{[&] {
                  return ::exec::subscribe(
                    static_cast<_Sender&&>(__sender), static_cast<__receiver_ref_t&&>(__receiver));
                }}};
            }};
          return &__vtable_;
        }
      };
    };

    template <class _Sigs, class _SenderQueries, class _ReceiverQueries>
    struct __sender_env {
      using __query_vtable_t = __query_vtable<_SenderQueries>;
      using __vtable_t = stdexec::__t<__sender_vtable<_Sigs, _SenderQueries, _ReceiverQueries>>;

      struct __t {
       public:
        __t(const __vtable_t* __vtable, void* __sender) noexcept
          : __vtable_{__vtable}
          , __sender_{__sender} {
        }
       private:
        const __vtable_t* __vtable_;
        void* __sender_;

        template <class _Tag, same_as<__t> _Self, class... _As>
          requires __callable<const __query_vtable_t&, _Tag, void*, _As...>
        friend auto tag_invoke(_Tag, const _Self& __self, _As&&... __as) noexcept(
          __nothrow_callable<const __query_vtable_t&, _Tag, void*, _As...>)
          -> __call_result_t<const __query_vtable_t&, _Tag, void*, _As...> {
          return __self.__vtable_->queries()(_Tag{}, __self.__sender_, (_As&&) __as...);
        }
      };
    };

    template <class _Sigs, class _SenderQueries = __types<>, class _ReceiverQueries = __types<>>
    struct __sequence_sender {
      using __receiver_ref_t = stdexec::__t<__next_receiver_ref<_Sigs, _ReceiverQueries>>;
      using __vtable_t = stdexec::__t<__sender_vtable<_Sigs, _SenderQueries, _ReceiverQueries>>;

      using __compl_sigs = __to_sequence_completions_t<_Sigs>;
      using __item_sender = typename any_receiver_ref<_Sigs>::template any_sender<>;

      class __t {
       public:
        using __id = __sequence_sender;
        using completion_signatures = __compl_sigs;
        using item_types = exec::item_types<__item_sender>;
        using sender_concept = sequence_sender_t;

        __t(const __t&) = delete;
        __t& operator=(const __t&) = delete;

        __t(__t&&) = default;
        __t& operator=(__t&&) = default;

        template <__not_decays_to<__t> _Sender>
          requires sequence_sender_to<_Sender, __receiver_ref_t>
        __t(_Sender&& __sndr)
          : __storage_{(_Sender&&) __sndr} {
        }

        __immovable_operation_storage __connect(__receiver_ref_t __receiver) {
          return __storage_.__get_vtable()->subscribe_(
            __storage_.__get_object_pointer(), __receiver);
        }

        __unique_storage_t<__vtable_t> __storage_;

        template <same_as<__t> _Self, class _Rcvr>
        friend stdexec::__t<__operation<stdexec::__id<_Rcvr>, true>>
          tag_invoke(subscribe_t, _Self&& __self, _Rcvr __rcvr) {
          return {static_cast<_Self&&>(__self), static_cast<_Rcvr&&>(__rcvr)};
        }

        using __env_t = stdexec::__t<__sender_env<_Sigs, _SenderQueries, _ReceiverQueries>>;

        template <same_as<get_env_t> _GetEnv, __decays_to<__t> _Self>
        friend __env_t tag_invoke(_GetEnv, _Self&& __self) noexcept {
          return {__self.__storage_.__get_vtable(), __self.__storage_.__get_object_pointer()};
        }
      };
    };
  }

  template <class _Completions, auto... _ReceiverQueries>
  class any_sequence_receiver_ref {
    using __receiver_base =
      stdexec::__t<__any::__next_receiver_ref<_Completions, queries<_ReceiverQueries...>>>;
    using __env_t = stdexec::env_of_t<__receiver_base>;
    __receiver_base __receiver_;
   public:
    using __id = any_sequence_receiver_ref;
    using __t = any_sequence_receiver_ref;
    using receiver_concept = stdexec::receiver_t;

    template <std::same_as<stdexec::get_env_t> _GetEnv, std::same_as<__t> _Self>
      requires stdexec::__callable<stdexec::get_env_t, const __receiver_base&>
    friend __env_t tag_invoke(_GetEnv, const _Self& __self) noexcept {
      return stdexec::get_env(__self.__receiver_);
    }

    template <
      std::same_as<exec::set_next_t> _SetNext,
      std::same_as<__t> _Self,
      stdexec::sender _Sender>
      requires stdexec::__callable<set_next_t, _Self&, _Sender>
    friend auto tag_invoke(_SetNext, _Self& __self, _Sender&& __sender) {
      return exec::set_next(__self.__receiver_, static_cast<_Sender&&>(__sender));
    }

    template <std::same_as<stdexec::set_value_t> _SetValue, std::same_as<__t> _Self>
      requires stdexec::__callable<stdexec::set_value_t, __receiver_base&&>
    friend void tag_invoke(_SetValue, _Self&& __self) noexcept {
      stdexec::set_value(static_cast<__receiver_base&&>(__self.__receiver_));
    }

    template <std::same_as<stdexec::set_error_t> _SetError, std::same_as<__t> _Self, class _Error>
      requires stdexec::__callable<stdexec::set_error_t, __receiver_base&&, _Error>
    friend void tag_invoke(_SetError, _Self&& __self, _Error&& __error) noexcept {
      stdexec::set_error(
        static_cast<__receiver_base&&>(__self.__receiver_), static_cast<_Error&&>(__error));
    }

    template <std::same_as<stdexec::set_stopped_t> _SetStopped, std::same_as<__t> _Self>
      requires stdexec::__callable<stdexec::set_stopped_t, __receiver_base&&>
    friend void tag_invoke(_SetStopped, _Self&& __self) noexcept {
      stdexec::set_stopped(static_cast<__receiver_base&&>(__self.__receiver_));
    }

    template <stdexec::__not_decays_to<__t> _Receiver>
      requires sequence_receiver_of<_Receiver, _Completions>
    any_sequence_receiver_ref(_Receiver& __receiver) noexcept
      : __receiver_(__receiver) {
    }

    template <auto... _SenderQueries>
    class any_sender {
      using __sender_base = stdexec::__t< __any::__sequence_sender<
        _Completions,
        queries<_SenderQueries...>,
        queries<_ReceiverQueries...>>>;
      __sender_base __sender_;

     public:
      using __id = any_sender;
      using __t = any_sender;
      using sender_concept = sequence_sender_t;
      using completion_signatures = typename __sender_base::completion_signatures;
      using item_types = typename __sender_base::item_types;

      template <stdexec::__not_decays_to<any_sender> _Sender>
        requires stdexec::sender_in<_Sender, __env_t>
              && sequence_sender_to<_Sender, __receiver_base>
      any_sender(_Sender&& __sender) noexcept(
        stdexec::__nothrow_constructible_from<__sender_base, _Sender>)
        : __sender_(static_cast<_Sender&&>(__sender)) {
      }

      template <stdexec::same_as<__t> _Self, sequence_receiver_of<item_types> _Rcvr>
      friend subscribe_result_t<__sender_base, _Rcvr>
        tag_invoke(exec::subscribe_t, _Self&& __self, _Rcvr __rcvr) {
        return exec::subscribe(
          static_cast<__sender_base&&>(__self.__sender_), static_cast<_Rcvr&&>(__rcvr));
      }

      template <stdexec::same_as<stdexec::get_env_t> _GetEnv, stdexec::__decays_to<__t> _Self>
      friend stdexec::env_of_t<__sender_base> tag_invoke(_GetEnv, _Self&& __self) noexcept {
        return stdexec::get_env(__self.__sender_);
      }
    };
  };
}