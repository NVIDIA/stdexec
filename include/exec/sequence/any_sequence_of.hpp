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

#include "../../stdexec/concepts.hpp"
#include "../../stdexec/execution.hpp"
#include "../any_sender_of.hpp"
#include "../sequence_senders.hpp"

namespace exec {
  namespace __any {
    namespace __next {
      template <__valid_completion_signatures _Sigs>
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

        template <__valid_completion_signatures _Sigs>
        constexpr auto
          operator()(_Sigs*) const -> __void_sender (*)(void*, __item_sender<_Sigs>&&) noexcept {
          return +[](void* __rcvr, __item_sender<_Sigs>&& __sndr) noexcept -> __void_sender {
            return __void_sender{
              set_next(*static_cast<_Rcvr*>(__rcvr), static_cast<__item_sender<_Sigs>&&>(__sndr))};
          };
        }
      };

      template <class _NextSigs, class _Sigs, class... _Queries>
      struct __next_vtable;

      template <class _NextSigs, class... _Sigs, class... _Queries>
      struct __next_vtable<_NextSigs, completion_signatures<_Sigs...>, _Queries...>
        : __rcvr_next_vfun<_NextSigs>
        , __rcvr_vfun<_Sigs>...
        , __query_vfun<_Queries>... {
        using __item_sender = any_receiver_ref<_NextSigs>::template any_sender<>;
        using __item_types = item_types<__item_sender>;

        using __query_vfun<_Queries>::operator()...;

        template <class _Rcvr>
          requires sequence_receiver_of<_Rcvr, __item_types>
                && (__callable<__query_vfun_fn<_Rcvr>, _Queries> && ...)
        static auto __create_vtable(__mtype<_Rcvr>) noexcept -> const __next_vtable* {
          static const __next_vtable __vtable_{
            {__rcvr_next_vfun_fn<_Rcvr>{}(static_cast<_NextSigs*>(nullptr))},
            {__rcvr_vfun_fn(static_cast<_Rcvr*>(nullptr), static_cast<_Sigs*>(nullptr))}...,
            {__query_vfun_fn<_Rcvr>{}(static_cast<_Queries>(nullptr))}...};
          return &__vtable_;
        }
      };

      template <class _Sigs, class... _Queries>
      struct __env {
        using __sigs = __to_sequence_completions_t<_Sigs>;
        using __vtable_t = __next_vtable<_Sigs, __sigs, _Queries...>;

        template <class _Tag, class... _As>
          requires __callable<const __vtable_t&, _Tag, void*, _As...>
        auto query(_Tag, _As&&... __as) const
          noexcept(__nothrow_callable<const __vtable_t&, _Tag, void*, _As...>)
            -> __call_result_t<const __vtable_t&, _Tag, void*, _As...> {
          return (*__vtable_)(_Tag(), __rcvr_, static_cast<_As&&>(__as)...);
        }

        const __vtable_t* __vtable_;
        void* __rcvr_;
      };

      template <class _Sigs, class... _Queries>
      struct __receiver_ref;

      template <class... _Sigs, class... _Queries>
      struct __receiver_ref<completion_signatures<_Sigs...>, _Queries...> {
        using __return_sigs = completion_signatures<set_value_t(), set_stopped_t()>;
        using __void_sender = any_receiver_ref<__return_sigs>::template any_sender<>;
        using __next_sigs = completion_signatures<_Sigs...>;
        using __sigs = __to_sequence_completions_t<__next_sigs>;
        using __item_sender = any_receiver_ref<__next_sigs>::template any_sender<>;
        using __item_types = item_types<__item_sender>;

        using __vtable_t = __next_vtable<__next_sigs, __sigs, _Queries...>;

        template <class Sig>
        using __vfun = __rcvr_vfun<Sig>;

        using __env_t = __env<__next_sigs, _Queries...>;

        using receiver_concept = STDEXEC::receiver_t;

        template <__none_of<__receiver_ref, const __receiver_ref, __env_t, const __env_t> _Rcvr>
          requires sequence_receiver_of<_Rcvr, __item_types>
                && (__callable<__query_vfun_fn<_Rcvr>, _Queries> && ...)
        __receiver_ref(_Rcvr& __rcvr) noexcept
          : __env_{__create_vtable(__mtype<__vtable_t>{}, __mtype<_Rcvr>{}), &__rcvr} {
        }

        template <class _Sender>
          requires __std::constructible_from<__item_sender, _Sender>
        auto set_next(_Sender&& __sndr) -> __void_sender {
          const __rcvr_next_vfun<__next_sigs>* __vfun = __env_.__vtable_;
          return __vfun->__fn_(__env_.__rcvr_, static_cast<_Sender&&>(__sndr));
        }

        // set_value_t() is always valid for a sequence
        void set_value() noexcept {
          const __vfun<set_value_t()>& __vfun = *__env_.__vtable_;
          __vfun(__env_.__rcvr_, set_value_t());
        }

        template <class Error>
          requires __mapply<__mcontains<set_error_t(Error)>, __sigs>::value
        void set_error(Error&& __error) noexcept {
          const __vfun<set_error_t(Error)>& __vfun = *__env_.__vtable_;
          __vfun(__env_.__rcvr_, set_error_t(), static_cast<Error&&>(__error));
        }

        void set_stopped() noexcept
          requires __mapply<__mcontains<set_stopped_t()>, __sigs>::value
        {
          const __vfun<set_stopped_t()>& __vfun = *__env_.__vtable_;
          __vfun(__env_.__rcvr_, set_stopped_t());
        }

        auto get_env() const noexcept -> const __env_t& {
          return __env_;
        }

        __env_t __env_;
      };
    } // namespace __next

    template <class _Sigs, class _Queries>
    using __next_receiver_ref =
      __mapply<__mbind_front<__q<__next::__receiver_ref>, _Sigs>, _Queries>;

    template <class _Sigs, class _SenderQueries, class _ReceiverQueries>
    struct __sender_vtable : public __query_vtable<_SenderQueries> {
      using __query_vtable_t = __query_vtable<_SenderQueries>;
      using __receiver_ref_t = __next_receiver_ref<_Sigs, _ReceiverQueries>;

      auto queries() const noexcept -> const __query_vtable_t& {
        return *this;
      }

      template <class _Sender>
        requires sequence_sender_to<_Sender, __receiver_ref_t>
      static auto __create_vtable(__mtype<_Sender>) noexcept -> const __sender_vtable* {
        static const __sender_vtable __vtable_{
          {*__any::__create_vtable(__mtype<__query_vtable_t>{}, __mtype<_Sender>{})},
          [](void* __object_pointer, __receiver_ref_t __receiver) -> __immovable_operation_storage {
            _Sender& __sender = *static_cast<_Sender*>(__object_pointer);
            using __op_state_t = subscribe_result_t<_Sender, __receiver_ref_t>;
            return __immovable_operation_storage{
              std::in_place_type<__op_state_t>, __emplace_from{[&] {
                return ::exec::subscribe(
                  static_cast<_Sender&&>(__sender), static_cast<__receiver_ref_t&&>(__receiver));
              }}};
          }};
        return &__vtable_;
      }

      __immovable_operation_storage (*subscribe_)(void*, __receiver_ref_t);
    };

    template <class _Sigs, class _SenderQueries, class _ReceiverQueries>
    struct __sender_env {
      using __query_vtable_t = __query_vtable<_SenderQueries>;
      using __vtable_t = __sender_vtable<_Sigs, _SenderQueries, _ReceiverQueries>;

      explicit __sender_env(const __vtable_t* __vtable, void* __sender) noexcept
        : __vtable_{__vtable}
        , __sender_{__sender} {
      }

     private:
      template <class _Tag, class... _As>
        requires __callable<const __query_vtable_t&, _Tag, void*, _As...>
      auto query(_Tag, _As&&... __as) const
        noexcept(__nothrow_callable<const __query_vtable_t&, _Tag, void*, _As...>)
          -> __call_result_t<const __query_vtable_t&, _Tag, void*, _As...> {
        return __vtable_->queries()(_Tag(), __sender_, static_cast<_As&&>(__as)...);
      }

      const __vtable_t* __vtable_;
      void* __sender_;
    };

    template <class _Sigs, class _SenderQueries = __mlist<>, class _ReceiverQueries = __mlist<>>
    struct __sequence_sender {
      using __receiver_ref_t = __next_receiver_ref<_Sigs, _ReceiverQueries>;
      using __vtable_t = __sender_vtable<_Sigs, _SenderQueries, _ReceiverQueries>;

      using __completions_t = __to_sequence_completions_t<_Sigs>;
      using __item_sender_t = any_receiver_ref<_Sigs>::template any_sender<>;
      using __env_t = __sender_env<_Sigs, _SenderQueries, _ReceiverQueries>;

      using completion_signatures = __completions_t;
      using item_types = exec::item_types<__item_sender_t>;
      using sender_concept = sequence_sender_t;

      template <__not_decays_to<__sequence_sender> _Sender>
        requires sequence_sender_to<_Sender, __receiver_ref_t>
      __sequence_sender(_Sender&& __sndr)
        : __storage_{static_cast<_Sender&&>(__sndr)} {
      }

      __sequence_sender(__sequence_sender&&) = default;
      __sequence_sender(const __sequence_sender&) = delete;

      auto operator=(__sequence_sender&&) -> __sequence_sender& = default;
      auto operator=(const __sequence_sender&) -> __sequence_sender& = delete;

      auto __connect(__receiver_ref_t __receiver) -> __immovable_operation_storage {
        return __storage_.__get_vtable()->subscribe_(__storage_.__get_object_pointer(), __receiver);
      }

      template <class _Rcvr>
      auto subscribe(_Rcvr __rcvr) && -> __operation<_Rcvr, true> {
        return __operation<_Rcvr, true>{
          static_cast<__sequence_sender&&>(*this), static_cast<_Rcvr&&>(__rcvr)};
      }


      auto get_env() const noexcept -> __env_t {
        return __env_t{__storage_.__get_vtable(), __storage_.__get_object_pointer()};
      }

      __unique_storage_t<__vtable_t> __storage_;
    };
  } // namespace __any

  template <class _Completions, auto... _ReceiverQueries>
  class any_sequence_receiver_ref {
    using __receiver_base_t =
      __any::__next_receiver_ref<_Completions, queries<_ReceiverQueries...>>;
    using __env_t = STDEXEC::env_of_t<__receiver_base_t>;
    __receiver_base_t __rcvr_;
   public:
    using receiver_concept = STDEXEC::receiver_t;

    template <auto... _SenderQueries>
    class any_sender;

    template <STDEXEC::__not_decays_to<any_sequence_receiver_ref> _Receiver>
      requires sequence_receiver_of<_Receiver, _Completions>
    any_sequence_receiver_ref(_Receiver& __receiver) noexcept
      : __rcvr_(__receiver) {
    }

    auto get_env() const noexcept -> __env_t {
      return STDEXEC::get_env(__rcvr_);
    }

    template <STDEXEC::sender _Item>
      requires STDEXEC::__callable<set_next_t, __receiver_base_t&, _Item>
    [[nodiscard]]
    auto set_next(_Item&& __item) & noexcept(
      STDEXEC::__nothrow_callable<set_next_t, __receiver_base_t&, _Item>)
      -> STDEXEC::__call_result_t<set_next_t, __receiver_base_t&, _Item> {
      return exec::set_next(__rcvr_, static_cast<_Item&&>(__item));
    }

    void set_value() noexcept {
      STDEXEC::set_value(static_cast<__receiver_base_t&&>(__rcvr_));
    }

    template <class _Error>
    void set_error(_Error&& __error) noexcept {
      STDEXEC::set_error(static_cast<__receiver_base_t&&>(__rcvr_), static_cast<_Error&&>(__error));
    }

    void set_stopped() noexcept {
      STDEXEC::set_stopped(static_cast<__receiver_base_t&&>(__rcvr_));
    }
  };

  template <class _Completions, auto... _ReceiverQueries>
  template <auto... _SenderQueries>
  class any_sequence_receiver_ref<_Completions, _ReceiverQueries...>::any_sender {
    using __base_t = __any::__sequence_sender<
      _Completions,
      queries<_SenderQueries...>,
      queries<_ReceiverQueries...>
    >;
    __base_t __sender_;

   public:
    using sender_concept = sequence_sender_t;
    using completion_signatures = __base_t::completion_signatures;
    using item_types = __base_t::item_types;

    template <STDEXEC::__not_decays_to<any_sender> _Sender>
      requires STDEXEC::sender_in<_Sender, __env_t>
            && sequence_sender_to<_Sender, __receiver_base_t>
    any_sender(_Sender&& __sender)
      noexcept(STDEXEC::__nothrow_constructible_from<__base_t, _Sender>)
      : __sender_(static_cast<_Sender&&>(__sender)) {
    }

    template <sequence_receiver_of<item_types> _Rcvr>
    auto subscribe(_Rcvr __rcvr) && -> subscribe_result_t<__base_t, _Rcvr> {
      return exec::subscribe(static_cast<__base_t&&>(__sender_), static_cast<_Rcvr&&>(__rcvr));
    }

    auto get_env() const noexcept -> STDEXEC::env_of_t<__base_t> {
      return STDEXEC::get_env(__sender_);
    }
  };

} // namespace exec