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

#include <type_traits>
#include <utility>

#include <exec/completion_signatures.hpp>
#include <exec/single_thread_context.hpp>
#include <stdexec/execution.hpp>
#include <test_common/receivers.hpp>

namespace
{

  template <typename T>
  struct receiver : expect_void_receiver<>
  {
    [[nodiscard]]
    constexpr ::STDEXEC::env<> get_env() const noexcept
    {
      return state_->get_env();
    }
    T* state_;
  };

  struct state
  {
    [[nodiscard]]
    constexpr ::STDEXEC::env<> get_env() const noexcept
    {
      return {};
    }
  };

  static_assert(!std::is_same_v<void,
                                decltype(::STDEXEC::connect(
                                  ::STDEXEC::just()
                                    | ::STDEXEC::write_env(::STDEXEC::prop{
                                      ::STDEXEC::get_stop_token,
                                      std::declval<::STDEXEC::inplace_stop_source&>().get_token()}),
                                  receiver<state>{{}, nullptr}))>);

  TEST_CASE("write_env works when the actual environment is sourced from a type which was "
            "initially "
            "incomplete but has since been completed",
            "[adaptors][write_env]")
  {
    ::STDEXEC::inplace_stop_source source;
    state                          s;
    auto                           op = ::STDEXEC::connect(::STDEXEC::just()
                                   | ::STDEXEC::write_env(::STDEXEC::prop{::STDEXEC::get_stop_token,
                                                                          source.get_token()}),
                                 receiver<state>{{}, &s});
    ::STDEXEC::start(op);
  }

  template <class IncompleteType, class Env = STDEXEC::env_of_t<IncompleteType>>
  struct ReceiverIncomplete
  {
    using receiver_concept = STDEXEC::receiver_tag;

    IncompleteType* m_ptr;

    void set_value() && noexcept
    {
      STDEXEC::set_value(std::move(m_ptr->rcvr));
    }

    template <typename Error>
    void set_error(Error&& error) && noexcept
    {
      STDEXEC::set_error(std::move(m_ptr->rcvr), std::forward<Error>(error));
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> Env
    {
      return STDEXEC::get_env(*m_ptr);
    }
  };

  template <STDEXEC::sender Sndr, STDEXEC::receiver Rcvr>
  struct OpStateIncomplete
  {
    using operation_state_concept = STDEXEC::operation_state_tag;

    using rcvr_t          = ReceiverIncomplete<OpStateIncomplete, STDEXEC::env_of_t<Rcvr>>;
    using inner_opstate_t = STDEXEC::connect_result_t<Sndr, rcvr_t>;

    Rcvr            rcvr;
    inner_opstate_t inner_opstate;

    OpStateIncomplete(Sndr&& sndr, Rcvr rcvr_)
      : rcvr(std::move(rcvr_))
      , inner_opstate(STDEXEC::connect(std::forward<Sndr>(sndr), rcvr_t{this}))
    {}

    void start() & noexcept
    {
      STDEXEC::start(inner_opstate);
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> STDEXEC::env_of_t<Rcvr>
    {
      return STDEXEC::get_env(rcvr);
    }
  };

  template <STDEXEC::sender Sndr>
  struct SenderIncomplete
  {
    using sender_concept = STDEXEC::sender_tag;

    template <class Self, class... Env>
    static consteval auto get_completion_signatures()
    {
      return exec::get_child_completion_signatures<Self, Sndr, Env...>();
    }

    template <STDEXEC::receiver Rcvr>
    auto connect(Rcvr rcvr) && -> OpStateIncomplete<Sndr, Rcvr>
    {
      return {std::forward<Sndr>(sndr), std::move(rcvr)};
    }

    Sndr sndr;
  };

  struct incomplete_t
  {
    constexpr auto operator()() const noexcept
    {
      return STDEXEC::__closure(*this);
    }

    template <typename Sndr>
    constexpr auto operator()(Sndr&& sndr) const -> SenderIncomplete<Sndr>
    {
      return {std::forward<Sndr>(sndr)};
    }
  };

  inline constexpr incomplete_t incomplete{};

  TEST_CASE("write_env with a receiver pointing to an incomplete operation state",
            "[adaptors][write_env]")
  {
    exec::single_thread_context stc{};

    int value = 0;

    STDEXEC::sender auto sndr =
      STDEXEC::read_env(STDEXEC::get_allocator)
      | STDEXEC::then([](auto allocator)
                      { static_assert(std::same_as<decltype(allocator), std::allocator<int>>); })
      | STDEXEC::continues_on(stc.get_scheduler()) | incomplete()
      | STDEXEC::write_env(STDEXEC::prop{STDEXEC::get_allocator, std::allocator<int>{}})
      | STDEXEC::then([&value]() { ++value; });

    STDEXEC::sync_wait(std::move(sndr));

    CHECK(value == 1);
  }

}  // namespace
