/*
 * Copyright (c) 2024 Lucian Radu Teodorescu
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

// Assumes STDEXEC_SYSTEM_CONTEXT_INLINE is defined

#if !defined(STDEXEC_SYSTEM_CONTEXT_INLINE)
#  error "STDEXEC_SYSTEM_CONTEXT_INLINE must be defined before including this header"
#endif

#include "__system_context_default_impl.hpp"

#include <atomic>
#include <thread>

namespace exec {
  namespace __detail {
    STDEXEC_SYSTEM_CONTEXT_INLINE constinit __spin_lock __system_context_mutex{};

    STDEXEC_SYSTEM_CONTEXT_INLINE constinit std::atomic<system_context_interface*>
      __system_context_instance{nullptr};

    STDEXEC_SYSTEM_CONTEXT_INLINE constinit std::atomic<new_system_context_handler>
      __new_system_context_handler{[]() -> system_context_interface* {
        return new __detail::__system_context_impl{};
      }};
  } // namespace __detail

  /// Gets the default system context implementation.
  extern "C" STDEXEC_SYSTEM_CONTEXT_INLINE system_context_interface* get_system_context_instance() {
    // spin until we get the lock:
    __detail::__system_context_mutex.__lock();
    // load the instance
    auto* __instance = __detail::__system_context_instance.load();
    // if it is not null, increment the reference count and return the ptr
    if (__instance != nullptr) {
      ++__instance->ref_count;
      __detail::__system_context_mutex.__unlock();
      return __instance;
    }
    __detail::__system_context_mutex.__unlock();

    // create a new instance:
    auto* __new_instance = __detail::__new_system_context_handler.load()();

    // spin until we get the lock:
    __detail::__system_context_mutex.__lock();

    // try to set the new instance. if this fails, we have been usurped.
    // increment the ref count on the usurper, destroy the instance we
    // just created, and return the usurper.
    if (!__detail::__system_context_instance.compare_exchange_strong(__instance, __new_instance)) {
      ++__instance->ref_count;
      __detail::__system_context_mutex.__unlock();
      __new_instance->destroy_fn(__new_instance);
      return __instance;
    }

    __detail::__system_context_mutex.__unlock();
    return __new_instance;
  }

  /// Releases a reference to the specified system context object.
  /// If the ref count drops to zero, then
  ///   - if the specified object is the active global object, set
  ///     the active global object to null.
  ///   - destroy the instance
  extern "C" STDEXEC_SYSTEM_CONTEXT_INLINE void
    release_system_context_instance(system_context_interface* __instance) noexcept {
    // spin until we get the lock:
    __detail::__system_context_mutex.__lock();
    if (0 != --__instance->ref_count) {
      __detail::__system_context_mutex.__unlock();
      return;
    }

    // if we have just released the last reference on the global system
    // context, set the global pointer to null:
    auto* __instance_copy = __instance;
    (void) __detail::__system_context_instance.compare_exchange_strong(__instance_copy, nullptr);

    __detail::__system_context_mutex.__unlock();
    __instance->destroy_fn(__instance);
  }

  /// Sets the default system context implementation.
  extern "C" STDEXEC_SYSTEM_CONTEXT_INLINE new_system_context_handler
    set_new_system_context_handler(new_system_context_handler __handler) {
    auto __old_handler = __detail::__new_system_context_handler.exchange(__handler);
    auto* __new_instance = __handler();

    auto* __instance = __detail::__system_context_instance.exchange(__new_instance);

    if (__instance != nullptr) {
      release_system_context_instance(__instance);
    }
    return __old_handler;
  }
} // namespace exec
