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
#include "../functional.hpp" // IWYU pragma: keep for __with_default
#include "../stop_token.hpp" // IWYU pragma: keep for get_stop_token_t
#include "__any_allocator.hpp"
#include "__queries.hpp"
#include "__typeinfo.hpp"

#include <exception>
#include <optional>
#include <span>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_MSVC(4702) // warning C4702: unreachable code

namespace STDEXEC {
  class task_scheduler;

  namespace system_context_replaceability {
    /// Interface for completing a sender operation. Backend will call frontend though
    /// this interface for completing the `schedule` and `schedule_bulk` operations.
    class receiver_proxy { //: __detail::__env_proxy {
     public:
      virtual constexpr ~receiver_proxy() = 0;

      virtual constexpr void set_value() noexcept = 0;
      virtual STDEXEC_CONSTEXPR_CXX23 void set_error(std::exception_ptr) noexcept = 0;
      virtual constexpr void set_stopped() noexcept = 0;

      /// Query the receiver for a property of type `_P`.
      template <class _Value, __class _Query>
      constexpr auto try_query(_Query) const noexcept -> std::optional<_Value> {
        std::optional<_Value> __p;
        __query_env(__mtypeid<_Query>, __mtypeid<_Value>, &__p);
        return __p;
      }

     protected:
      virtual constexpr void __query_env(__type_index, __type_index, void*) const noexcept = 0;
    };

    inline constexpr receiver_proxy::~receiver_proxy() = default;

    struct bulk_item_receiver_proxy : receiver_proxy {
      virtual constexpr void execute(size_t, size_t) noexcept = 0;
    };

    /// Interface for the parallel scheduler backend.
    struct parallel_scheduler_backend {
      virtual constexpr ~parallel_scheduler_backend() = 0;

      /// Future-proofing: in case we need to add more virtual functions, we can use this
      /// to query for additional interfaces without breaking ABI.
      [[nodiscard]]
      virtual constexpr auto __query_interface(__type_index __id) const noexcept -> void* {
        if (__id == __mtypeid<parallel_scheduler_backend>) {
          return const_cast<parallel_scheduler_backend*>(this);
        }
        return nullptr;
      }

      /// Schedule work on parallel scheduler, calling `__r` when done and using `__s` for preallocated
      /// memory.
      virtual constexpr void schedule(receiver_proxy&, std::span<std::byte>) noexcept = 0;

      /// Schedule bulk work of size `__n` on parallel scheduler, calling `__r` for different
      /// subranges of [0, __n), and using `__s` for preallocated memory.
      virtual constexpr void schedule_bulk_chunked(
        std::size_t,
        bulk_item_receiver_proxy&,
        std::span<std::byte>) noexcept = 0;

      /// Schedule bulk work of size `__n` on parallel scheduler, calling `__r` for each item, and
      /// using `__s` for preallocated memory.
      virtual constexpr void schedule_bulk_unchunked(
        std::size_t,
        bulk_item_receiver_proxy&,
        std::span<std::byte>) noexcept = 0;
    };

    inline constexpr parallel_scheduler_backend::~parallel_scheduler_backend() = default;
  } // namespace system_context_replaceability

  namespace __detail {
    // Partially implements the _RcvrProxy interface (either receiver_proxy or
    // bulk_item_receiver_proxy) in terms of a concrete receiver type _Rcvr.
    template <class _Rcvr, class _RcvrProxy>
    struct __receiver_proxy_base : _RcvrProxy {
     public:
      using receiver_concept = receiver_t;

      constexpr explicit __receiver_proxy_base(_Rcvr rcvr) noexcept
        : __rcvr_(static_cast<_Rcvr&&>(rcvr)) {
      }

      STDEXEC_CONSTEXPR_CXX23 void set_error(std::exception_ptr eptr) noexcept final {
        STDEXEC::set_error(std::move(__rcvr_), std::move(eptr));
      }

      constexpr void set_stopped() noexcept final {
        STDEXEC::set_stopped(std::move(__rcvr_));
      }

     protected:
      constexpr void __query_env(__type_index __query_id, __type_index __value, void* __dest)
        const noexcept final {
        if (__query_id == __mtypeid<get_stop_token_t>) {
          __query(get_stop_token, __value, __dest);
        } else if (__query_id == __mtypeid<get_allocator_t>) {
          __query(get_allocator, __value, __dest);
        }
      }

     private:
      constexpr void __query(get_stop_token_t, __type_index __value_type, void* __dest) const noexcept {
        using __stop_token_t = stop_token_of_t<env_of_t<_Rcvr>>;
        if constexpr (std::is_same_v<inplace_stop_token, __stop_token_t>) {
          if (__value_type == __mtypeid<inplace_stop_token>) {
            using __dest_t = std::optional<inplace_stop_token>;
            *static_cast<__dest_t*>(__dest) = STDEXEC::get_stop_token(STDEXEC::get_env(__rcvr_));
          }
        }
      }

      constexpr void __query(get_allocator_t, __type_index __value_type, void* __dest) const noexcept {
        if (__value_type == __mtypeid<__any_allocator<std::byte>>) {
          using __dest_t = std::optional<__any_allocator<std::byte>>;
          constexpr auto __get_alloc = __with_default(get_allocator, std::allocator<std::byte>());
          auto __alloc = STDEXEC::__rebind_allocator<std::byte>(
            __get_alloc(STDEXEC::get_env(__rcvr_)));
          *static_cast<__dest_t*>(__dest) = __any_allocator{std::move(__alloc)};
        }
      }

     public:
      _Rcvr __rcvr_;
    };

    template <class _Rcvr>
    struct __receiver_proxy
      : __receiver_proxy_base<_Rcvr, system_context_replaceability::receiver_proxy> {
      using __receiver_proxy::__receiver_proxy_base::__receiver_proxy_base;

      constexpr void set_value() noexcept final {
        STDEXEC::set_value(std::move(this->__rcvr_));
      }
    };

    // A receiver type that forwards its completion operations to a _RcvrProxy member held
    // by reference (where _RcvrProxy is one of receiver_proxy or
    // bulk_item_receiver_proxy). It is also responsible for destroying and, if necessary,
    // deallocating the operation state.
    template <class _RcvrProxy>
    struct __proxy_receiver {
      using receiver_concept = receiver_t;
      using __delete_fn_t = void(void*) noexcept;

      constexpr void set_value() noexcept {
        auto& __proxy = __rcvr_proxy_;
        __delete_fn_(__opstate_storage_); // NB: destroys *this
        __proxy.set_value();
      }

      STDEXEC_CONSTEXPR_CXX23 void set_error(std::exception_ptr __eptr) noexcept {
        auto& __proxy = __rcvr_proxy_;
        __delete_fn_(__opstate_storage_); // NB: destroys *this
        __proxy.set_error(std::move(__eptr));
      }

      constexpr void set_stopped() noexcept {
        auto& __proxy = __rcvr_proxy_;
        __delete_fn_(__opstate_storage_); // NB: destroys *this
        __proxy.set_stopped();
      }

      [[nodiscard]]
      constexpr auto get_env() const noexcept -> env_of_t<_RcvrProxy> {
        return STDEXEC::get_env(__rcvr_proxy_);
      }

      _RcvrProxy& __rcvr_proxy_;
      void* __opstate_storage_;
      __delete_fn_t* __delete_fn_;
    };
  } // namespace __detail
} // namespace STDEXEC

STDEXEC_PRAGMA_POP()
