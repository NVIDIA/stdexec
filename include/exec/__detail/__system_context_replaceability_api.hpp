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

#include "../../stdexec/__detail/__execution_fwd.hpp"

#include <cstdint>
#include <cstddef>
#include <exception>
#include <optional>
#include <memory>
#include <span>

struct __uuid {
  std::uint64_t __parts1;
  std::uint64_t __parts2;

  friend auto operator==(__uuid, __uuid) noexcept -> bool = default;
};

namespace exec::system_context_replaceability {

  /// Helper for the `__queryable_interface` concept.
  template <__uuid X>
  using __check_constexpr_uuid = void;

  /// Concept for a queryable interface. Ensures that the interface has a `__interface_identifier` member.
  template <typename _T>
  concept __queryable_interface = requires() {
    typename __check_constexpr_uuid<_T::__interface_identifier>;
  };

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

  struct parallel_scheduler_backend;

  /// Get the backend for the parallel scheduler.
  /// Users might replace this function.
  auto query_parallel_scheduler_backend() -> std::shared_ptr<parallel_scheduler_backend>;

  /// The type of a factory that can create `parallel_scheduler_backend` instances.
  /// Out of spec.
  using __parallel_scheduler_backend_factory = std::shared_ptr<parallel_scheduler_backend> (*)();

  /// Set a factory for the parallel scheduler backend.
  /// Can be used to replace the parallel scheduler at runtime.
  /// Out of spec.
  auto set_parallel_scheduler_backend(__parallel_scheduler_backend_factory __new_factory)
    -> __parallel_scheduler_backend_factory;

  /// Interface for completing a sender operation. Backend will call frontend though this interface
  /// for completing the `schedule` and `schedule_bulk` operations.
  struct receiver {
    virtual ~receiver() = default;

   protected:
    virtual auto __query_env(__uuid, void*) noexcept -> bool = 0;

   public:
    /// Called when the system scheduler completes successfully.
    virtual void set_value() noexcept = 0;
    /// Called when the system scheduler completes with an error.
    virtual void set_error(std::exception_ptr) noexcept = 0;
    /// Called when the system scheduler was stopped.
    virtual void set_stopped() noexcept = 0;

    /// Query the receiver for a property of type `_P`.
    template <typename _P>
    auto try_query() noexcept -> std::optional<std::decay_t<_P>> {
      if constexpr (__runtime_property<_P>) {
        std::decay_t<_P> __p;
        bool __success =
          __query_env(__runtime_property_helper<std::decay_t<_P>>::__property_identifier, &__p);
        return __success ? std::make_optional(std::move(__p)) : std::nullopt;
      } else {
        return std::nullopt;
      }
    }
  };

  /// Receiver for bulk sheduling operations.
  struct bulk_item_receiver : receiver {
    /// Called for each item of a bulk operation, possible on different threads.
    virtual void execute(std::uint32_t, std::uint32_t) noexcept = 0;
  };

  /// Interface for the parallel scheduler backend.
  struct parallel_scheduler_backend {
    static constexpr __uuid __interface_identifier{0x5ee9202498c4bd4f, 0xa1df2508ffcd9d7e};

    virtual ~parallel_scheduler_backend() = default;

    /// Schedule work on parallel scheduler, calling `__r` when done and using `__s` for preallocated
    /// memory.
    virtual void schedule(std::span<std::byte> __s, receiver& __r) noexcept = 0;
    /// Schedule bulk work of size `__n` on parallel scheduler, calling `__r` for different
    /// subranges of [0, __n), and using `__s` for preallocated memory.
    virtual void schedule_bulk_chunked(
      std::uint32_t __n,
      std::span<std::byte> __s,
      bulk_item_receiver& __r) noexcept = 0;
    /// Schedule bulk work of size `__n` on parallel scheduler, calling `__r` for each item, and
    /// using `__s` for preallocated memory.
    virtual void schedule_bulk_unchunked(
      std::uint32_t __n,
      std::span<std::byte> __s,
      bulk_item_receiver& __r) noexcept = 0;
  };

} // namespace exec::system_context_replaceability

#endif
