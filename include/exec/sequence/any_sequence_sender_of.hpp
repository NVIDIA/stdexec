/* Copyright (c) 2023 Maikel Nadolski
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

#include "../sequence_senders.hpp"

namespace exec {
  namespace __any {
    namespace __next {
      using namespace __rec;

      template <__is_completion_signatures _Sigs>
      struct __rcvr_next_vfun {
        using __return_sigs = completion_signatures<set_value_t(), set_stopped_t()>;
        using __void_sender = any_receiver_ref<__return_sigs>::template any_sender<>;
        using __item_sender = any_receiver_ref<_Sigs>::template any_sender<>;
        __void_sender (*__fn_)(void*, __item_sender&&) noexcept;
      };

      template <class _Rcvr>
      struct __rcvr_next_vfun_fn {
        using __return_sigs = completion_signatures<set_value_t(), set_stopped_t()>;
        using __void_sender = any_receiver_ref<__return_sigs>::template any_sender<>;

        template <class _Sigs>
        using __item_sender = any_receiver_ref<_Sigs>::template any_sender<>;

        template <__is_completion_signatures _Sigs>
        constexpr __void_sender (
          *operator()(_Sigs*) const noexcept)(void*, __item_sender<_Sigs>&&) noexcept {
          return +[](void* __r, __item_sender<_Sigs>&& __sndr) noexcept -> __void_sender {
            return __void_sender{exec::set_next(
              *static_cast<_Rcvr*>(__r), static_cast<__item_sender<_Sigs>&&>(__sndr))};
          };
        }
      };

      template <class _NextSigs, class _Sigs, class... _Queries>
      struct __next_vtable;

      template <class _NextSigs, class... _Sigs, class... _Queries>
      struct __next_vtable<_NextSigs, completion_signatures<_Sigs...>, _Queries...> {
        class __t
          : public __rcvr_next_vfun<_NextSigs>
          , public __rcvr_vfun<_Sigs>...
          , public __query_vfun<_Queries>... {
         public:
          using __query_vfun<_Queries>::operator()...;

         private:
          template <class _Rcvr>
            requires sequence_receiver_of<_Rcvr, _NextSigs>
                  && (__callable<__query_vfun_fn<_Rcvr>, _Queries> && ...)
          friend const __t* tag_invoke(__create_vtable_t, __mtype<__t>, __mtype<_Rcvr>) noexcept {
            static const __t __vtable_{
              {__rcvr_next_vfun_fn<_Rcvr>{}((_NextSigs*) nullptr)},
              {__rcvr_vfun_fn<_Rcvr>{}((_Sigs*) nullptr)}...,
              {__query_vfun_fn<_Rcvr>{}((_Queries) nullptr)}...};
            return &__vtable_;
          }
        };
      };

      template <class _Sigs, class... _Queries>
      struct __receiver_ref;

      template <class... _Sigs, class... _Queries>
      struct __receiver_ref<completion_signatures<_Sigs...>, _Queries...> {
        struct __t {
         private:
          using __return_sigs = completion_signatures<set_value_t(), set_stopped_t()>;
          using __void_sender = any_receiver_ref<__return_sigs>::template any_sender<>;
          using __next_sigs = completion_signatures<_Sigs...>;
          using __compl_sigs = __sequence_to_sender_sigs_t<__next_sigs>;
          using __item_sender = any_receiver_ref<__next_sigs>::template any_sender<>;

          using __vtable_t = stdexec::__t<__next_vtable<__next_sigs, __compl_sigs, _Queries...>>;

          struct __env_t {
            const __vtable_t* __vtable_;
            void* __rcvr_;

            template <class _Tag, same_as<__env_t> _Self, class... _As>
              requires __callable<const __vtable_t&, _Tag, void*, _As...>
            friend auto tag_invoke(_Tag, const _Self& __self, _As&&... __as) noexcept(
              __nothrow_callable<const __vtable_t&, _Tag, void*, _As...>)
              -> __call_result_t<const __vtable_t&, _Tag, void*, _As...> {
              return (*__self.__vtable_)(_Tag{}, __self.__rcvr_, (_As&&) __as...);
            }
          } __env_;
         public:
          using is_receiver = void;

          template <__none_of<__t, const __t, __env_t, const __env_t> _Rcvr>
            requires sequence_receiver_of<_Rcvr, completion_signatures<_Sigs...>>
                  && (__callable<__query_vfun_fn<_Rcvr>, _Queries> && ...)
          __t(_Rcvr& __rcvr) noexcept
            : __env_{__create_vtable(__mtype<__vtable_t>{}, __mtype<_Rcvr>{}), &__rcvr} {
          }

          template <same_as<set_next_t> _SetNext, same_as<__t> _Self, class _Sender>
            requires constructible_from<__item_sender, _Sender>
          friend __void_sender tag_invoke(_SetNext, _Self& __self, _Sender&& __sndr) noexcept {
            return (
              *static_cast<const __rcvr_next_vfun<__next_sigs>*>(__self.__env_.__vtable_)->__fn_)(
              __self.__env_.__rcvr_, static_cast<_Sender&&>(__sndr));
          }

          template <__completion_tag _Tag, __decays_to<__t> _Self, class... _As>
            requires __v<__mapply<__contains<_Tag(_As...)>, __compl_sigs>>
          friend void tag_invoke(_Tag, _Self&& __self, _As&&... __as) noexcept {
            (*static_cast<const __rcvr_vfun<_Tag(_As...)>*>(__self.__env_.__vtable_)->__fn_)(
              ((_Self&&) __self).__env_.__rcvr_, (_As&&) __as...);
          }

          template <std::same_as<__t> Self>
          friend const __env_t& tag_invoke(get_env_t, const Self& __self) noexcept {
            return __self.__env_;
          }
        };
      };
    }

    template <class _Sigs, class _Queries>
    using __next_receiver_ref =
      __t<__mapply<__mbind_front<__q<__next::__receiver_ref>, _Sigs>, _Queries>>;

    template <class _Sigs, class _SenderQueries = __types<>, class _ReceiverQueries = __types<>>
    struct __sequence_sender {
      using __receiver_ref_t = __next_receiver_ref<_Sigs, _ReceiverQueries>;

      class __vtable : public __query_vtable<_SenderQueries> {
       public:
        using __id = __vtable;

        const __query_vtable<_SenderQueries>& __queries() const noexcept {
          return *this;
        }

        __unique_operation_storage (*__sequence_connect_)(void*, __receiver_ref_t);
       private:
        template <class _Sender>
          requires sequence_sender_to<_Sender, __receiver_ref_t>
        friend const __vtable*
          tag_invoke(__create_vtable_t, __mtype<__vtable>, __mtype<_Sender>) noexcept {
          static const __vtable __vtable_{
            {*__create_vtable(__mtype<__query_vtable<_SenderQueries>>{}, __mtype<_Sender>{})},
            [](void* __object_pointer, __receiver_ref_t __receiver) -> __unique_operation_storage {
              _Sender& __sender = *static_cast<_Sender*>(__object_pointer);
              using __op_state_t = sequence_connect_result_t<_Sender, __receiver_ref_t>;
              return __unique_operation_storage{
                std::in_place_type<__op_state_t>, __conv{[&] {
                  return exec::sequence_connect(
                    (_Sender&&) __sender, (__receiver_ref_t&&) __receiver);
                }}};
            }};
          return &__vtable_;
        }
      };

      class __env_t {
       public:
        __env_t(const __vtable* __vtable, void* __sender) noexcept
          : __vtable_{__vtable}
          , __sender_{__sender} {
        }
       private:
        const __vtable* __vtable_;
        void* __sender_;

        template <class _Tag, class... _As>
          requires __callable<const __query_vtable<_SenderQueries>&, _Tag, void*, _As...>
        friend auto tag_invoke(_Tag, const __env_t& __self, _As&&... __as) noexcept(
          __nothrow_callable<const __query_vtable<_SenderQueries>&, _Tag, void*, _As...>)
          -> __call_result_t<const __query_vtable<_SenderQueries>&, _Tag, void*, _As...> {
          return __self.__vtable_->__queries()(_Tag{}, __self.__sender_, (_As&&) __as...);
        }
      };

      class __t {
       public:
        using __id = __sequence_sender;
        using completion_signatures = _Sigs;
        using is_sender = void;

        __t(const __t&) = delete;
        __t& operator=(const __t&) = delete;

        __t(__t&&) = default;
        __t& operator=(__t&&) = default;

        template <__not_decays_to<__t> _Sender>
          requires sequence_sender_to<_Sender, __receiver_ref_t>
        __t(_Sender&& __sndr)
          : __storage_{(_Sender&&) __sndr} {
        }

        __unique_operation_storage __connect(__receiver_ref_t __receiver) {
          return __storage_.__get_vtable()->__sequence_connect_(
            __storage_.__get_object_pointer(), __receiver);
        }

       private:
        __unique_storage_t<__vtable> __storage_;

        template <same_as<__t> _Self, sequence_receiver_of<_Sigs> _Rcvr>
        friend stdexec::__t<__operation<__t, __decay_t<_Rcvr>, _ReceiverQueries>>
          tag_invoke(sequence_connect_t, _Self&& __self, _Rcvr&& __rcvr) {
          return {(__t&&) __self, (_Rcvr&&) __rcvr};
        }

        friend __env_t tag_invoke(get_env_t, const __t& __self) noexcept {
          return {__self.__storage_.__get_vtable(), __self.__storage_.__get_object_pointer()};
        }
      };
    };
  }

  template <class _Completions, auto... _ReceiverQueries>
  class any_sequence_receiver_ref {
    using __receiver_base = __any::__next_receiver_ref<_Completions, queries<_ReceiverQueries...>>;
    using __env_t = stdexec::env_of_t<__receiver_base>;
    __receiver_base __receiver_;

    template <class _Tag, stdexec::__decays_to<any_sequence_receiver_ref> Self, class... _As>
      requires stdexec::tag_invocable<_Tag, stdexec::__copy_cvref_t<Self, __receiver_base>, _As...>
    friend auto tag_invoke(_Tag, Self&& __self, _As&&... __as) noexcept(
      std::is_nothrow_invocable_v< _Tag, stdexec::__copy_cvref_t<Self, __receiver_base>, _As...>) {
      return tag_invoke(_Tag{}, ((Self&&) __self).__receiver_, (_As&&) __as...);
    }

   public:
    using is_receiver = void;
    using __t = any_sequence_receiver_ref;
    using __id = any_sequence_receiver_ref;

    template <stdexec::__not_decays_to<any_sequence_receiver_ref> _Receiver>
      requires sequence_receiver_of<_Receiver, _Completions>
    any_sequence_receiver_ref(_Receiver& __receiver) noexcept
      : __receiver_(__receiver) {
    }

    template <auto... _SenderQueries>
    class any_sender {
      using __sender_base = stdexec::__t<
        __any::
          __sequence_sender<_Completions, queries<_SenderQueries...>, queries<_ReceiverQueries...>>>;
      __sender_base __sender_;

      template <class _Tag, stdexec::__decays_to<any_sender> Self, class... _As>
        requires stdexec::tag_invocable< _Tag, stdexec::__copy_cvref_t<Self, __sender_base>, _As...>
      friend auto tag_invoke(_Tag, Self&& __self, _As&&... __as) noexcept(
        std::is_nothrow_invocable_v< _Tag, stdexec::__copy_cvref_t<Self, __sender_base>, _As...>) {
        return tag_invoke(_Tag{}, ((Self&&) __self).__sender_, (_As&&) __as...);
      }
     public:
      using is_sequence_sender = void;
      using completion_signatures = typename __sender_base::completion_signatures;

      template <class _Sender>
        requires(!stdexec::__decays_to<_Sender, any_sender>) && stdexec::sender<_Sender>
      any_sender(_Sender&& __sender) noexcept(
        stdexec::__nothrow_constructible_from<__sender_base, _Sender>)
        : __sender_((_Sender&&) __sender) {
      }
    };
  };
}