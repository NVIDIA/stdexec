/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *                         Copyright (c) 2025 Robert Leahy. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <catch2/catch_all.hpp>

#include <stdexec/execution.hpp>

#if STDEXEC_USE_MODULES()
import std;
#else
#  include <atomic>
#  include <chrono>
#  include <memory>
#  include <optional>
#  include <thread>
#  include <type_traits>
#endif

namespace
{

  struct on_stop_request
  {
    void operator()() && noexcept {}
  };

  TEST_CASE("inplace_stop_callback exposes its callback type", "[stop_token]")
  {
    STATIC_REQUIRE(std::is_same_v<::STDEXEC::inplace_stop_callback<on_stop_request>::callback_type,
                                  on_stop_request>);
  }

  TEST_CASE("(un)stoppable_token correctly categorizes various standard stop token types",
            "[concepts]")
  {
    STATIC_REQUIRE(::STDEXEC::stoppable_token<::STDEXEC::never_stop_token>);
    STATIC_REQUIRE(::STDEXEC::unstoppable_token<::STDEXEC::never_stop_token>);
    STATIC_REQUIRE(::STDEXEC::stoppable_token<::STDEXEC::inplace_stop_token>);
    STATIC_REQUIRE(!::STDEXEC::unstoppable_token<::STDEXEC::inplace_stop_token>);
    STATIC_REQUIRE(
      std::is_same_v<::STDEXEC::stop_callback_for_t<::STDEXEC::never_stop_token, on_stop_request>,
                     ::STDEXEC::never_stop_token::callback_type<on_stop_request>>);

#if defined(__cpp_lib_jthread) && __cpp_lib_jthread >= 201911L
    STATIC_REQUIRE(::STDEXEC::stoppable_token<std::stop_token>);
    STATIC_REQUIRE(std::is_same_v<::STDEXEC::stop_callback_for_t<std::stop_token, on_stop_request>,
                                  std::stop_callback<on_stop_request>>);
#endif
  }

  TEST_CASE("inplace_stop_callback supports class template argument deduction", "[stop_token]")
  {
    ::STDEXEC::inplace_stop_source   source;
    ::STDEXEC::inplace_stop_callback cb{source.get_token(), on_stop_request{}};
    STATIC_REQUIRE(std::is_same_v<decltype(cb), ::STDEXEC::inplace_stop_callback<on_stop_request>>);
  }

  struct stop_state;

  struct destroy_on_stop
  {
    std::unique_ptr<stop_state>* state;

    void operator()() noexcept;
  };

  struct stop_state
  {
    ::STDEXEC::inplace_stop_source                                   source;
    std::optional<::STDEXEC::inplace_stop_callback<on_stop_request>> other;
    std::optional<::STDEXEC::inplace_stop_callback<destroy_on_stop>> callback;
  };

  void destroy_on_stop::operator()() noexcept
  {
    state->reset();
  }

  TEST_CASE("a stop callback can destroy the inplace_stop_source", "[stop_token]")
  {
    auto state = std::make_unique<stop_state>();
    state->other.emplace(state->source.get_token(), on_stop_request{});
    state->callback.emplace(state->source.get_token(), destroy_on_stop{&state});

    CHECK(state->source.request_stop());
    CHECK(state == nullptr);
  }

  struct slow_state;

  struct remove_and_sleep
  {
    slow_state*        state;
    std::atomic<bool>* removed;

    void operator()() noexcept;
  };

  struct slow_state
  {
    ::STDEXEC::inplace_stop_source                                    source;
    std::optional<::STDEXEC::inplace_stop_callback<remove_and_sleep>> callback;
  };

  void remove_and_sleep::operator()() noexcept
  {
    // Like an operation that completes from inside its stop callback.
    auto* removed_flag = removed;
    state->callback.reset();
    removed_flag->store(true);

    // Keep request_stop() running while the main thread destroys the source.
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  TEST_CASE("inplace_stop_source destructor waits for request_stop on another thread",
            "[stop_token]")
  {
    std::atomic<bool> removed{false};
    auto              state = std::make_unique<slow_state>();
    state->callback.emplace(state->source.get_token(), remove_and_sleep{state.get(), &removed});

    std::thread thread([source = &state->source] { source->request_stop(); });
    while (!removed.load())
      std::this_thread::yield();

    state.reset();  // request_stop() is still running on the other thread
    thread.join();
  }
}  // namespace
