/*
 * Copyright (c) 2025 Lucian Radu Teodorescu, Lewis Baker
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

#ifndef STDEXEC_SYSTEM_CONTEXT_REPLACEABILITY_API_H
#define STDEXEC_SYSTEM_CONTEXT_REPLACEABILITY_API_H

#include "stdexec/__detail/__execution_fwd.hpp"

#include <typeindex>
#include <optional>

struct __uuid {
  uint64_t __parts1;
  uint64_t __parts2;

  friend bool operator==(__uuid, __uuid) noexcept = default;
};

namespace exec::system_context_replaceability {

  /// Helper for the `__queryable_interface` concept.
  template <__uuid X>
  using __check_constexpr_uuid = void;

  /// Concept for a queryable interface. Ensures that the interface has a `__interface_identifier` member.
  template <typename _T>
  concept __queryable_interface =
    requires() { typename __check_constexpr_uuid<_T::__interface_identifier>; };

  /// The details for making `_T` a runtime property.
  template <typename _T>
  struct __runtime_property_helper {
    /// Is `_T` a property?
    static constexpr bool __is_property = false;
    /// The unique identifier for the property.
    static constexpr __uuid __property_identifier{0, 0};
  };

  /// `inplace_stope_token` is a runtime property.
  template <>
  struct __runtime_property_helper<stdexec::inplace_stop_token> {
    static constexpr bool __is_property = true;
    static constexpr __uuid __property_identifier{0x8779c09d8aa249df, 0x867db0e653202604};
  };

  /// Concept for a runtime property.
  template <typename _T>
  concept __runtime_property = __runtime_property_helper<_T>::__is_property;

  /// Query the system context for an interface of type `_Interface`.
  template <__queryable_interface _Interface>
  extern std::shared_ptr<_Interface> query_system_context();

  /// The type of a factory that can create interfaces of type `_Interface`.
  template <__queryable_interface _Interface>
  using __system_context_backend_factory = std::shared_ptr<_Interface> (*)();

  /// Sets the factory that creates the system context backend for an interface of type `_Interface`.
  template <__queryable_interface _Interface>
  extern __system_context_backend_factory<_Interface>
    set_system_context_backend_factory(__system_context_backend_factory<_Interface> __new_factory);

  /// Interface for completing a sender operation.
  /// Backend will call frontend though this interface for completing the `schedule` and `schedule_bulk` operations.
  struct receiver {
    virtual ~receiver() = default;

    /// Called when the system scheduler completes successfully.
    virtual void set_value() noexcept = 0;
    /// Called when the system scheduler completes with an error.
    virtual void set_error(std::exception_ptr) noexcept = 0;
    /// Called when the system scheduler was stopped.
    virtual void set_stopped() noexcept = 0;
  };

  /// Receiver for bulk sheduling operations.
  struct bulk_item_receiver : receiver {
    /// Called for each item of a bulk operation, possible on different threads.
    virtual void start(uint32_t) noexcept = 0;
  };

  /// Describes a storage space.
  /// Used to pass preallocated storage from the frontend to the backend.
  struct storage {
    void* __data;
    uint32_t __size;
  };

  struct env {
    /// Query the system context for a property of type `_P`.
    template <typename _P>
    std::optional<_P> try_query() noexcept {
      if constexpr (__runtime_property<_P>) {
        const void* __r = __query_(__data_, __runtime_property_helper<_P>::__property_identifier);
        return __r ? std::make_optional(*static_cast<const _P*>(__r)) : std::nullopt;
      } else {
        return std::nullopt;
      }
    }

    // IMPLEMENTATION DETAIL: used by the frontend with no proerties.
    explicit env() noexcept
      : __data_{nullptr}
      , __query_{&__query_none} {
    }

    // IMPLEMENTATION DETAIL: used by the frontend with a single property.
    template <__runtime_property _P>
    explicit env(const _P& __p) noexcept
      : __data_{std::addressof(__p)}
      , __query_{&__query_single<_P>} {
    }

    // IMPLEMENTATION DETAIL: used by the frontend with multiple properties.
    template <__runtime_property... _Ps>
    explicit env(const std::tuple<_Ps...>& __properties) noexcept
      : __data_{std::addressof(__properties)}
      , __query_{&__query_multi<_Ps...>} {
    }


   private:
    /// The source data containing all the properties.
    const void* __data_;
    /// Function called by the backend to query for a specific property.
    const void* (*__query_)(const void*, __uuid) noexcept;

    static const void* __query_none(const void* __data, __uuid __id) noexcept {
      return nullptr;
    }

    template <__runtime_property _P>
    static const void* __query_single(const void* __data, __uuid __id) noexcept {
      if (__id == __runtime_property_helper<_P>::__property_identifier)
        return __data;
      return nullptr;
    }

    template <__runtime_property _P, typename _Tuple>
    static uintptr_t __select_property_as_int(const _Tuple& __tuple, __uuid __id) noexcept {
      if (__id == __runtime_property_helper<_P>::__property_identifier)
        return reinterpret_cast<uintptr_t>(std::addressof(std::get<_P>(__tuple)));
      return 0;
    }

    template <__runtime_property... _Ps>
    static const void* __query_multi(const void* __data, __uuid __id) noexcept {
      const std::tuple<_Ps...>& __properties = *static_cast<const std::tuple<_Ps...>*>(__data);
      uintptr_t __result = (... + __select_property_as_int<_Ps>(__properties, __id));
      return reinterpret_cast<const void*>(__result);
    }
  };

  /// Interface for the system scheduler
  struct system_scheduler {
    static constexpr __uuid __interface_identifier{0x5ee9202498c4bd4f, 0xa1df2508ffcd9d7e};

    virtual ~system_scheduler() = default;

    /// Schedule work on system scheduler, calling `__r` when done and using `__s` for preallocated memory, using `__e` for environment.
    virtual void schedule(storage __s, receiver* __r, env __e) noexcept = 0;
    /// Schedule bulk work of size `__n` on system scheduler, calling `__r` for each item and then when done, and using `__s` for preallocated memory, using `__e` for environment.
    virtual void
      bulk_schedule(uint32_t __n, storage __s, bulk_item_receiver* __r, env __e) noexcept = 0;
  };

} // namespace exec::system_context_replaceability

#endif