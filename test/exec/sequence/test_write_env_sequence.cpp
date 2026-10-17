/*
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

// Regression tests for issue #2053: `write_env` must be transparent to
// sequence sender semantics. Wrapping a sequence sender in `write_env` (or in
// several stacked `write_env` layers) must preserve the item types, and items
// must flow through the wrapper unchanged at runtime.

#include "exec/sequence/transform_each.hpp"

#include "exec/sequence/ignore_all_values.hpp"
#include "exec/sequence_senders.hpp"
#include <test_common/catch2.hpp>

#include <test_common/receivers.hpp>
#include <test_common/senders.hpp>
#include <test_common/type_helpers.hpp>

#include <optional>
#include <utility>

namespace
{
  // A minimal sequence sender that produces a single `just(42)` item.
  template <int Value>
  struct single_item_sequence
  {
    using sender_concept = exec::sequence_sender_tag;
    using item_types     = exec::item_types<decltype(STDEXEC::just(int{}))>;
    using completion_signatures =
      STDEXEC::completion_signatures<STDEXEC::set_value_t(), STDEXEC::set_stopped_t()>;

    template <class Rcvr>
    struct op
    {
      using operation_state_concept = STDEXEC::operation_state_t;
      Rcvr rcvr_;

      void start() noexcept
      {
        auto next    = exec::set_next(rcvr_, STDEXEC::just(Value));
        auto item_op = STDEXEC::connect(std::move(next), std::move(rcvr_));
        STDEXEC::start(item_op);
      }
    };

    template <STDEXEC::receiver Rcvr>
    auto subscribe(Rcvr rcvr) const -> op<Rcvr>
    {
      return op<Rcvr>{static_cast<Rcvr&&>(rcvr)};
    }
  };

  TEST_CASE("write_env is transparent to sequence senders - item types", "[sequence][write_env]")
  {
    using wrapper_t = STDEXEC::__decay_t<decltype(STDEXEC::write_env(single_item_sequence<42>{},
                                                                     STDEXEC::env<>{}))>;
    using items_t   = decltype(exec::get_item_types<wrapper_t, STDEXEC::env<>>());
    static_assert(STDEXEC::__same_as<items_t, exec::item_types<decltype(STDEXEC::just(int{}))>>);
  }

  TEST_CASE("stacked write_env is transparent to sequence senders", "[sequence][write_env]")
  {
    using wrapper_t = STDEXEC::__decay_t<
      decltype(STDEXEC::write_env(STDEXEC::write_env(single_item_sequence<42>{}, STDEXEC::env<>{}),
                                  STDEXEC::env<>{}))>;
    using items_t = decltype(exec::get_item_types<wrapper_t, STDEXEC::env<>>());
    static_assert(STDEXEC::__same_as<items_t, exec::item_types<decltype(STDEXEC::just(int{}))>>);
  }

  TEST_CASE("write_env preserves sequence item flow", "[sequence][write_env]")
  {
    int value = 0;
    STDEXEC::sync_wait(STDEXEC::write_env(single_item_sequence<42>{}, STDEXEC::env<>{})
                       | exec::transform_each(STDEXEC::then([&](int v) { value = v; }))
                       | exec::ignore_all_values());
    CHECK(value == 42);
  }

  TEST_CASE("stacked write_env preserves sequence item flow", "[sequence][write_env]")
  {
    int value = 0;
    STDEXEC::sync_wait(
      STDEXEC::write_env(STDEXEC::write_env(single_item_sequence<42>{}, STDEXEC::env<>{}),
                         STDEXEC::env<>{})
      | exec::transform_each(STDEXEC::then([&](int v) { value = v; })) | exec::ignore_all_values());
    CHECK(value == 42);
  }

  // A user-defined adaptor, to demonstrate that the transparent-adaptor
  // customization point is open: any single-child adaptor can declare itself
  // transparent to sequence sender semantics by specializing
  // `__sequence_adaptor_traits` for its tag, without any changes to the
  // sequence machinery. See issue #2053.
  struct my_adapt_t
  {};

  template <STDEXEC::sender Sndr>
  auto my_adapt(Sndr&& sndr)
  {
    return STDEXEC::__sexpr{my_adapt_t(), STDEXEC::env<>{}, static_cast<Sndr&&>(sndr)};
  }

  struct my_adapt_impl : STDEXEC::__sexpr_defaults
  {
    template <class Sndr, class... Env>
    static consteval auto __get_completion_signatures()
    {
      return STDEXEC::get_completion_signatures<STDEXEC::__child_of<Sndr>, Env...>();
    }
  };
}  // namespace

template <>
struct STDEXEC::__sexpr_impl<my_adapt_t> : my_adapt_impl
{};

template <>
struct experimental::execution::__sequence_adaptor_traits<my_adapt_t>
{
  static constexpr bool __transparent = true;
};

namespace
{
  TEST_CASE("a custom adaptor can declare itself transparent", "[sequence][write_env]")
  {
    using wrapper_t = STDEXEC::__decay_t<decltype(my_adapt(single_item_sequence<42>{}))>;
    using items_t   = decltype(exec::get_item_types<wrapper_t, STDEXEC::env<>>());
    static_assert(STDEXEC::__same_as<items_t, exec::item_types<decltype(STDEXEC::just(int{}))>>);
  }

  TEST_CASE("a custom transparent adaptor preserves sequence item flow", "[sequence][write_env]")
  {
    int value = 0;
    STDEXEC::sync_wait(my_adapt(single_item_sequence<42>{})
                       | exec::transform_each(STDEXEC::then([&](int v) { value = v; }))
                       | exec::ignore_all_values());
    CHECK(value == 42);
  }

  // Lifetime tests for issue #2305: `write_env` over a sequence sender injects
  // its data into the environment the child is subscribed with, and the
  // resulting operation state must own that data rather than reference the
  // (long-dead) sender.

  struct read_env_query : STDEXEC::__query<read_env_query>
  {
    static consteval auto query(STDEXEC::forwarding_query_t) noexcept -> bool
    {
      return true;
    }
  };

  inline constexpr int poisoned_value = -1;

  struct injected_env_data
  {
    int value;

    explicit injected_env_data(int value) noexcept
      : value(value)
    {}

    injected_env_data(injected_env_data const &)    = default;
    injected_env_data(injected_env_data&&) noexcept = default;

    // Poison the value on destruction so that a read through a dangling
    // reference after the sender is gone fails deterministically.
    ~injected_env_data()
    {
      value = poisoned_value;
    }

    auto query(read_env_query) const noexcept -> int
    {
      return value;
    }
  };

  struct move_only_env_data
  {
    int value;

    explicit move_only_env_data(int value) noexcept
      : value(value)
    {}

    move_only_env_data(move_only_env_data&&) noexcept = default;
    move_only_env_data(move_only_env_data const &)    = delete;

    ~move_only_env_data()
    {
      value = poisoned_value;
    }

    auto query(read_env_query) const noexcept -> int
    {
      return value;
    }
  };

  // A sequence sender that reads an injected query out of its receiver's
  // environment when started.
  struct env_reading_sequence
  {
    using sender_concept = exec::sequence_sender_tag;
    using item_types     = exec::item_types<decltype(STDEXEC::just(int{}))>;
    using completion_signatures =
      STDEXEC::completion_signatures<STDEXEC::set_value_t(), STDEXEC::set_stopped_t()>;

    int* observed_;

    template <class Rcvr>
    struct op
    {
      using operation_state_concept = STDEXEC::operation_state_t;
      Rcvr rcvr_;
      int* observed_;

      void start() noexcept
      {
        *observed_ = read_env_query{}(STDEXEC::get_env(rcvr_));
        STDEXEC::set_value(static_cast<Rcvr&&>(rcvr_));
      }
    };

    template <STDEXEC::receiver Rcvr>
    auto subscribe(Rcvr rcvr) const -> op<Rcvr>
    {
      return op<Rcvr>{static_cast<Rcvr&&>(rcvr), observed_};
    }
  };

  struct test_sequence_rcvr
  {
    using receiver_concept = STDEXEC::receiver_tag;

    template <class Item>
    auto set_next(Item&& item) noexcept
    {
      return static_cast<Item&&>(item);
    }

    void set_value() noexcept {}

    void set_stopped() noexcept {}

    template <class Error>
    void set_error(Error&&) noexcept
    {}

    auto get_env() const noexcept
    {
      return STDEXEC::prop(STDEXEC::get_stop_token, STDEXEC::inplace_stop_token{});
    }
  };

  TEST_CASE("write_env's env data outlives a temporary sequence sender", "[sequence][write_env]")
  {
    int  observed = 0;
    auto op       = exec::subscribe(STDEXEC::write_env(env_reading_sequence{&observed},
                                                 injected_env_data{42}),
                              test_sequence_rcvr{});
    STDEXEC::start(op);
    CHECK(observed == 42);
  }

  TEST_CASE("write_env with a move-only env over a sequence sender", "[sequence][write_env]")
  {
    int  observed = 0;
    auto op       = exec::subscribe(STDEXEC::write_env(env_reading_sequence{&observed},
                                                 move_only_env_data{42}),
                              test_sequence_rcvr{});
    STDEXEC::start(op);
    CHECK(observed == 42);
  }

  TEST_CASE("write_env's env data outlives a named sequence sender", "[sequence][write_env]")
  {
    int observed = 0;
    using sndr_t = decltype(STDEXEC::write_env(env_reading_sequence{}, injected_env_data{0}));
    using op_t   = exec::subscribe_result_t<sndr_t&, test_sequence_rcvr>;
    std::optional<sndr_t> sndr;
    sndr.emplace(STDEXEC::write_env(env_reading_sequence{&observed}, injected_env_data{42}));
    std::optional<op_t> op;
    op.emplace(exec::subscribe(*sndr, test_sequence_rcvr{}));
    sndr.reset();
    STDEXEC::start(*op);
    CHECK(observed == 42);
  }
}  // namespace
