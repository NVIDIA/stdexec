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

#include "exec/sequence/transform_each.hpp"

#include "exec/sequence.hpp"
#include "exec/sequence/ignore_all_values.hpp"
#include "exec/sequence/iterate.hpp"
#include "exec/sequence_senders.hpp"
#include "exec/static_thread_pool.hpp"
#include <test_common/catch2.hpp>

#include <array>
#include <atomic>
#include <ranges>
#include <thread>
#include <utility>

namespace
{
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
        auto next    = exec::set_next(rcvr_, STDEXEC::just(42));
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

  TEST_CASE("starts_on keeps the item types of a sequence sender", "[sequence][starts_on]")
  {
    exec::static_thread_pool pool{1};
    using sndr_t  = decltype(STDEXEC::starts_on(pool.get_scheduler(), single_item_sequence{}));
    using items_t = decltype(exec::get_item_types<sndr_t, STDEXEC::env<>>());
    STATIC_REQUIRE(STDEXEC::__same_as<items_t, exec::item_types<decltype(STDEXEC::just(int{}))>>);
  }

  TEST_CASE("starts_on over a sequence sender delivers its items on the scheduler",
            "[sequence][starts_on]")
  {
    exec::static_thread_pool pool{1};
    auto [pool_id] = STDEXEC::sync_wait(STDEXEC::schedule(pool.get_scheduler())
                                        | STDEXEC::then([] { return std::this_thread::get_id(); }))
                       .value();
    std::atomic<int> seen{0};
    std::thread::id  item_id{};
    STDEXEC::sync_wait(STDEXEC::starts_on(pool.get_scheduler(), single_item_sequence{})
                       | exec::transform_each(STDEXEC::then(
                         [&](int v)
                         {
                           item_id = std::this_thread::get_id();
                           seen    = v;
                         }))
                       | exec::ignore_all_values());
    CHECK(seen == 42);
    CHECK(item_id == pool_id);
    CHECK(item_id != std::this_thread::get_id());
  }

  TEST_CASE("starts_on over iterate delivers every item", "[sequence][starts_on]")
  {
    exec::static_thread_pool pool{1};
    std::array<int, 4>       values{1, 2, 3, 4};
    std::atomic<int>         sum{0};
    STDEXEC::sync_wait(
      STDEXEC::starts_on(pool.get_scheduler(), exec::iterate(std::views::all(values)))
      | exec::transform_each(STDEXEC::then([&](int v) { sum += v; })) | exec::ignore_all_values());
    CHECK(sum == 10);
  }

  // A sender that says nothing about where it completes
  struct opaque_sender
  {
    using sender_concept        = STDEXEC::sender_tag;
    using completion_signatures = STDEXEC::completion_signatures<STDEXEC::set_value_t()>;

    template <class Rcvr>
    struct op
    {
      using operation_state_concept = STDEXEC::operation_state_t;
      Rcvr rcvr_;

      void start() noexcept
      {
        STDEXEC::set_value(std::move(rcvr_));
      }
    };

    template <STDEXEC::receiver Rcvr>
    auto connect(Rcvr rcvr) const -> op<Rcvr>
    {
      return op<Rcvr>{static_cast<Rcvr&&>(rcvr)};
    }
  };

  // Attributes that say that a sender completes where it starts
  struct inline_attrs
  {
    template <class Tag>
    static constexpr auto query(STDEXEC::__get_completion_behavior_t<Tag>) noexcept
    {
      return STDEXEC::__completion_behavior::__inline_completion;
    }
  };

  // A sequence sender without items that completes where it starts, and records whether its
  // receiver's environment says where that is
  struct start_scheduler_probe_sequence
  {
    using sender_concept        = exec::sequence_sender_tag;
    using item_types            = exec::item_types<>;
    using completion_signatures = STDEXEC::completion_signatures<STDEXEC::set_value_t()>;

    template <class Rcvr>
    struct op
    {
      using operation_state_concept = STDEXEC::operation_state_t;
      Rcvr  rcvr_;
      bool* sees_start_scheduler_;

      void start() noexcept
      {
        *sees_start_scheduler_ = requires(STDEXEC::env_of_t<Rcvr> const & env) {
          STDEXEC::get_start_scheduler(env);
        };
        STDEXEC::set_value(std::move(rcvr_));
      }
    };

    template <STDEXEC::receiver Rcvr>
    auto subscribe(Rcvr rcvr) const -> op<Rcvr>
    {
      return op<Rcvr>{static_cast<Rcvr&&>(rcvr), sees_start_scheduler_};
    }

    static constexpr auto get_env() noexcept -> inline_attrs
    {
      return {};
    }

    bool* sees_start_scheduler_;
  };

  template <class Sch>
  struct start_scheduler_rcvr
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

    [[nodiscard]]
    auto get_env() const noexcept
    {
      return STDEXEC::prop{STDEXEC::get_start_scheduler, sch_};
    }

    Sch sch_;
  };

  template <class Sndr, class Env>
  concept has_value_completion_scheduler = requires(Sndr const & sndr, Env const & env) {
    STDEXEC::get_completion_scheduler<STDEXEC::set_value_t>(STDEXEC::get_env(sndr), env);
  };

  // starts_on over a sequence sender is lowered to a sequence that ends in the sequence sender.
  // Such a sequence too must hide the receiver's start scheduler from the sequence sender,
  // unless the sender before it completes where it starts.
  TEST_CASE("a sequence hides the receiver's start scheduler from a sequence sender that may "
            "not start there",
            "[sequence]")
  {
    STDEXEC::run_loop loop;
    using sch_t = decltype(loop.get_scheduler());
    using env_t = STDEXEC::prop<STDEXEC::get_start_scheduler_t, sch_t>;

    STATIC_REQUIRE(!has_value_completion_scheduler<
                   decltype(exec::sequence(opaque_sender{}, start_scheduler_probe_sequence{})),
                   env_t>);

    bool sees_start_scheduler = true;
    auto sndr                 = exec::sequence(opaque_sender{},
                               start_scheduler_probe_sequence{&sees_start_scheduler});
    auto op = exec::subscribe(std::move(sndr), start_scheduler_rcvr<sch_t>{loop.get_scheduler()});
    STDEXEC::start(op);
    CHECK_FALSE(sees_start_scheduler);
  }

  TEST_CASE("a sequence shows the receiver's start scheduler to a sequence sender that starts "
            "there",
            "[sequence]")
  {
    STDEXEC::run_loop loop;
    using sch_t = decltype(loop.get_scheduler());

    bool sees_start_scheduler = false;
    auto sndr                 = exec::sequence(STDEXEC::just(),
                               start_scheduler_probe_sequence{&sees_start_scheduler});
    CHECK(STDEXEC::get_completion_scheduler<STDEXEC::set_value_t>(
            STDEXEC::get_env(sndr),
            STDEXEC::prop{STDEXEC::get_start_scheduler, loop.get_scheduler()})
          == loop.get_scheduler());

    auto op = exec::subscribe(std::move(sndr), start_scheduler_rcvr<sch_t>{loop.get_scheduler()});
    STDEXEC::start(op);
    CHECK(sees_start_scheduler);
  }
}  // namespace
