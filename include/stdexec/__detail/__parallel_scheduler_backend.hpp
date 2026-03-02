/*
 * Copyright (c) 2025 Lucian Radu Teodorescu, Lewis Baker
 * Copyright (c) 2026 NVIDIA Corporation
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

#include "__execution_fwd.hpp"

// include these after __execution_fwd.hpp
#include "../functional.hpp"  // IWYU pragma: keep for __with_default
#include "../stop_token.hpp"  // IWYU pragma: keep for get_stop_token_t
#include "__any_allocator.hpp"
#include "__optional.hpp"
#include "__queries.hpp"
#include "__typeinfo.hpp"

#include <exception>
#include <optional>
#include <span>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_MSVC(4702)  // warning C4702: unreachable code

namespace STDEXEC
{
  class task_scheduler;

  namespace system_context_replaceability
  {
    /// Interface for completing a sender operation. Backend will call frontend though
    /// this interface for completing the `schedule` and `schedule_bulk` operations.
    class receiver_proxy
    {
     public:
      virtual constexpr ~receiver_proxy() = 0;

      virtual constexpr void set_value() noexcept                   = 0;
      virtual void           set_error(std::exception_ptr) noexcept = 0;
      virtual constexpr void set_stopped() noexcept                 = 0;

      /// Query the receiver for a property of type `_Query`.
      template <class _Value, __class _Query>
      constexpr auto try_query(_Query) const noexcept -> std::optional<_Value>
      {
        std::optional<_Value> __val;
        __query_env(__mtypeid<_Query>, __mtypeid<_Value>, &__val);
        return __val;
      }

     protected:
      virtual constexpr void __query_env(__type_index, __type_index, void*) const noexcept = 0;
    };

    inline constexpr receiver_proxy::~receiver_proxy() = default;

    struct bulk_item_receiver_proxy : receiver_proxy
    {
      virtual constexpr void execute(size_t, size_t) noexcept = 0;
    };

    /// Interface for the parallel scheduler backend.
    struct parallel_scheduler_backend
    {
      virtual constexpr ~parallel_scheduler_backend() = 0;

      /// Future-proofing: in case we need to add more virtual functions, we can use this
      /// to query for additional interfaces without breaking ABI.
      [[nodiscard]]
      virtual constexpr auto __query_interface(__type_index __id) const noexcept -> void*
      {
        if (__id == __mtypeid<parallel_scheduler_backend>)
        {
          return const_cast<parallel_scheduler_backend*>(this);
        }
        return nullptr;
      }

      /// Schedule work on parallel scheduler, calling `__r` when done and using `__s` for preallocated
      /// memory.
      virtual constexpr void schedule(receiver_proxy&, std::span<std::byte>) noexcept = 0;

      /// Schedule bulk work of size `__n` on parallel scheduler, calling `__r` for different
      /// subranges of [0, __n), and using `__s` for preallocated memory.
      virtual constexpr void schedule_bulk_chunked(std::size_t,
                                                   bulk_item_receiver_proxy&,
                                                   std::span<std::byte>) noexcept = 0;

      /// Schedule bulk work of size `__n` on parallel scheduler, calling `__r` for each item, and
      /// using `__s` for preallocated memory.
      virtual constexpr void schedule_bulk_unchunked(std::size_t,
                                                     bulk_item_receiver_proxy&,
                                                     std::span<std::byte>) noexcept = 0;
    };

    inline constexpr parallel_scheduler_backend::~parallel_scheduler_backend() = default;
  }  // namespace system_context_replaceability

  namespace __detail
  {
    template <class _Token>
    struct __stop_callback_for
    {
      using __callback_t = stop_callback_for_t<_Token, __forward_stop_request>;

      bool __register_stop_callback(_Token __token)
      {
        if (__token.stop_possible())
        {
          __callback_.emplace(__token, __forward_stop_request{__source_});
        }
        return __token.stop_requested();
      }

      void __unregister_stop_callback()
      {
        __callback_.reset();
      }

      inplace_stop_source      __source_{};
      __optional<__callback_t> __callback_{};
    };

    template <>
    struct __stop_callback_for<inplace_stop_token>
    {
      bool __register_stop_callback(inplace_stop_token __token)
      {
        return __token.stop_requested();
      }

      void __unregister_stop_callback() {}
    };

    template <unstoppable_token _Token>
    struct __stop_callback_for<_Token>
    {
      bool __register_stop_callback(__ignore)
      {
        return false;
      }

      void __unregister_stop_callback() {}
    };

    // Partially implements the _RcvrProxy interface (either receiver_proxy or
    // bulk_item_receiver_proxy) in terms of a concrete receiver type _Rcvr.
    template <class _Rcvr, class _RcvrProxy, bool _Infallible = false>
    struct STDEXEC_ATTRIBUTE(empty_bases) __receiver_proxy_base
      : _RcvrProxy
      , __stop_callback_for<stop_token_of_t<env_of_t<_Rcvr>>>
    {
     public:
      using receiver_concept = receiver_t;
      using __stop_token_t   = stop_token_of_t<env_of_t<_Rcvr>>;
      using __allocator_t    = std::allocator_traits<
           __call_result_or_t<get_allocator_t,
                              __any_allocator<std::byte>,
                              env_of_t<_Rcvr>>>::template rebind_alloc<std::byte>;

      constexpr explicit __receiver_proxy_base(_Rcvr rcvr) noexcept
        : __rcvr_(static_cast<_Rcvr&&>(rcvr))
      {}

      // Returns true if stop was requested at the time of registration.
      bool __register_stop_callback()
      {
        if constexpr (!unstoppable_token<__stop_token_t>)
        {
          __stop_callback_for<stop_token_of_t<env_of_t<_Rcvr>>>& __self = *this;
          return __self.__register_stop_callback(
            STDEXEC::get_stop_token(STDEXEC::get_env(__rcvr_)));
        }
        return false;
      }

      void set_error(std::exception_ptr eptr) noexcept final
      {
        if constexpr (_Infallible)
        {
          STDEXEC_ASSERT(!+"set_error called on an infallible receiver proxy");
          STDEXEC_UNREACHABLE();
        }
        else
        {
          this->__unregister_stop_callback();
          STDEXEC::set_error(std::move(__rcvr_), std::move(eptr));
        }
      }

      constexpr void set_stopped() noexcept final
      {
        if constexpr (_Infallible && unstoppable_token<__stop_token_t>)
        {
          STDEXEC_ASSERT(!+"set_stopped called on an unstoppable receiver proxy");
          STDEXEC_UNREACHABLE();
        }
        else
        {
          this->__unregister_stop_callback();
          STDEXEC::set_stopped(std::move(__rcvr_));
        }
      }

     protected:
      constexpr void
      __query_env(__type_index __query_id, __type_index __value, void* __dest) const noexcept final
      {
        if (__query_id == __mtypeid<get_stop_token_t>)
        {
          __query(get_stop_token, __value, __dest);
        }
        else if (__query_id == __mtypeid<get_allocator_t>)
        {
          __query(get_allocator, __value, __dest);
        }
        else if (__query_id == __mtypeid<get_scheduler_t>)
        {
          __query(get_scheduler, __value, __dest);
        }
      }

     private:
      constexpr void
      __query(get_stop_token_t, __type_index __value_type, void* __dest) const noexcept
      {
        // Branch taken when the user has requested an inplace_stop_token
        if (__value_type == __mtypeid<inplace_stop_token>)
        {
          auto& __val = *static_cast<std::optional<inplace_stop_token>*>(__dest);
          if constexpr (__same_as<__stop_token_t, inplace_stop_token>)
          {
            __val.emplace(STDEXEC::get_stop_token(STDEXEC::get_env(__rcvr_)));
          }
          else if constexpr (unstoppable_token<__stop_token_t>)
          {
            __val.emplace();
          }
          else
          {
            __val.emplace(this->__source_.get_token());
          }
        }
        else if (__value_type == __mtypeid<never_stop_token>)
        {
          auto& __val = *static_cast<std::optional<never_stop_token>*>(__dest);
          __val.emplace();
        }
        else if (__value_type == __mtypeid<__stop_token_t>)
        {
          auto& __val = *static_cast<std::optional<__stop_token_t>*>(__dest);
          __val.emplace(STDEXEC::get_stop_token(STDEXEC::get_env(__rcvr_)));
        }
      }

      constexpr void
      __query(get_allocator_t, __type_index __value_type, void* __dest) const noexcept
      {
        auto __alloc_def  = std::allocator<std::byte>();
        auto __alloc      = __with_default(get_allocator, __alloc_def)(STDEXEC::get_env(__rcvr_));
        auto __byte_alloc = STDEXEC::__rebind_allocator<std::byte>(__alloc);

        if (__value_type == __mtypeid<__any_allocator<std::byte>>)
        {
          auto& __val = *static_cast<std::optional<__any_allocator<std::byte>>*>(__dest);
          __val.emplace(__byte_alloc);
        }
        else if (__value_type == __mtypeid<__allocator_t>)
        {
          auto& __val = *static_cast<std::optional<__allocator_t>*>(__dest);
          __val.emplace(__byte_alloc);
        }
      }

      // Defined in __task_scheduler.hpp
      constexpr void __query(get_scheduler_t, __type_index, void*) const noexcept;

     public:
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Rcvr __rcvr_;
    };

    template <class _Rcvr, bool _Infallible = false>
    struct __receiver_proxy
      : __receiver_proxy_base<_Rcvr, system_context_replaceability::receiver_proxy, _Infallible>
    {
      using __receiver_proxy::__receiver_proxy_base::__receiver_proxy_base;

      constexpr void set_value() noexcept final
      {
        this->__unregister_stop_callback();
        STDEXEC::set_value(std::move(this->__rcvr_));
      }
    };

    struct __proxy_env
    {
      [[nodiscard]]
      auto query(get_allocator_t) const noexcept -> __any_allocator<std::byte>
      {
        auto __alloc = __rcvr_.template try_query<__any_allocator<std::byte>>(get_allocator);
        return __alloc ? *__alloc : __any_allocator<std::byte>{std::allocator<std::byte>()};
      }

      [[nodiscard]]
      auto query(get_stop_token_t) const noexcept -> inplace_stop_token
      {
        auto __token = __rcvr_.template try_query<inplace_stop_token>(get_stop_token);
        return __token ? *__token : inplace_stop_token{};
      }

      // Implemented in __task_scheduler.hpp
      [[nodiscard]]
      auto query(get_scheduler_t) const noexcept -> task_scheduler;

      system_context_replaceability::receiver_proxy& __rcvr_;
    };

    // A receiver type that forwards its completion operations to a _RcvrProxy member held
    // by reference (where _RcvrProxy is one of receiver_proxy or
    // bulk_item_receiver_proxy). It is also responsible for destroying and, if necessary,
    // deallocating the operation state.
    template <class _RcvrProxy, class _Env>
    struct __proxy_receiver
    {
      using receiver_concept = receiver_t;
      using __delete_fn_t    = void(void*) noexcept;

      constexpr void set_value() noexcept
      {
        auto& __proxy = __rcvr_proxy_;
        __delete_fn_(__opstate_storage_);  // NB: destroys *this
        __proxy.set_value();
      }

      void set_error(std::exception_ptr __eptr) noexcept
      {
        auto& __proxy = __rcvr_proxy_;
        __delete_fn_(__opstate_storage_);  // NB: destroys *this
        __proxy.set_error(std::move(__eptr));
      }

      constexpr void set_stopped() noexcept
      {
        auto& __proxy = __rcvr_proxy_;
        __delete_fn_(__opstate_storage_);  // NB: destroys *this
        __proxy.set_stopped();
      }

      [[nodiscard]]
      constexpr auto get_env() const noexcept -> _Env
      {
        return _Env{__rcvr_proxy_};
      }

      _RcvrProxy&    __rcvr_proxy_;
      void*          __opstate_storage_;
      __delete_fn_t* __delete_fn_;
    };
  }  // namespace __detail
}  // namespace STDEXEC

STDEXEC_PRAGMA_POP()
