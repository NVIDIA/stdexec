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

#include "exec/sequence/iterate.hpp"
#include "stdexec/execution.hpp"

#include <array>
#include <initializer_list>
#include <numeric>
#include <optional>
#include <test_common/catch2.hpp>
#include <utility>
#include <vector>

namespace
{

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
  struct begin_error
  {};

  struct begin_throws_range
  {
    using iterator = std::array<int, 1>::iterator;

    iterator begin()
    {
      throw begin_error{};
    }

    iterator end() noexcept
    {
      return values.end();
    }

    std::array<int, 1> values{0};
  };
#endif  // !STDEXEC_NO_STDCPP_EXCEPTIONS()

  template <class Receiver>
  struct sum_item_rcvr
  {
    using receiver_concept = STDEXEC::receiver_tag;
    Receiver rcvr;
    int*     sum_;

    [[nodiscard]]
    auto get_env() const noexcept -> STDEXEC::env_of_t<Receiver>
    {
      return STDEXEC::get_env(rcvr);
    }

    template <class... As>
    void set_value(int x) noexcept
    {
      *sum_ += x;
      STDEXEC::set_value(static_cast<Receiver&&>(rcvr));
    }

    void set_stopped() noexcept
    {
      STDEXEC::set_value(static_cast<Receiver&&>(rcvr));
    }

    template <class E>
    void set_error(E&&) noexcept
    {
      STDEXEC::set_value(static_cast<Receiver&&>(rcvr));
    }
  };

  template <class Item>
  struct sum_sender
  {
    using sender_concept        = STDEXEC::sender_tag;
    using completion_signatures = STDEXEC::completion_signatures<STDEXEC::set_value_t()>;

    Item item_;
    int* sum_;

    template <STDEXEC::__decays_to<sum_sender>            Self,
              STDEXEC::receiver_of<completion_signatures> Receiver>
    STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this Self&& self, Receiver rcvr) noexcept
    {
      return STDEXEC::connect(static_cast<Self&&>(self).item_,
                              sum_item_rcvr<Receiver>{static_cast<Receiver&&>(rcvr), self.sum_});
    }
    STDEXEC_EXPLICIT_THIS_END(connect)
  };

  template <class Env = STDEXEC::env<>>
  struct sum_receiver
  {
    using receiver_concept = STDEXEC::receiver_tag;

    int& sum_;
    Env  env_{};

    template <class Item>
    auto set_next(Item&& item) noexcept -> sum_sender<STDEXEC::__decay_t<Item>>
    {
      return {static_cast<Item&&>(item), &sum_};
    }

    void set_value() noexcept {}

    void set_stopped() noexcept {}

    void set_error(std::exception_ptr) noexcept {}

    [[nodiscard]]
    auto get_env() const noexcept -> Env
    {
      return env_;
    }
  };

  struct move_only_sum_receiver : sum_receiver<>
  {
    move_only_sum_receiver(int& sum, bool& completed) noexcept
      : sum_receiver<>{.sum_ = sum}
      , completed_(completed)
    {}

    move_only_sum_receiver(move_only_sum_receiver const &) = delete;
    move_only_sum_receiver(move_only_sum_receiver&&)       = default;

    void set_value() noexcept
    {
      completed_ = true;
    }

    bool& completed_;
  };

  TEST_CASE("iterate - accepts a move-only receiver", "[sequence_senders][iterate]")
  {
    std::array<int, 3> array{42, 43, 44};
    auto               count = GENERATE(0, 1, 3);
    auto iterate   = exec::iterate(std::ranges::subrange(array.begin(), array.begin() + count));
    int  sum       = 0;
    bool completed = false;

    STATIC_REQUIRE(STDEXEC::receiver<move_only_sum_receiver>);

    auto check = [&](auto&& sequence)
    {
      auto op = exec::subscribe(static_cast<decltype(sequence)&&>(sequence),
                                move_only_sum_receiver{sum, completed});
      STDEXEC::start(op);
      CHECK(completed);
      CHECK(sum == std::accumulate(array.begin(), array.begin() + count, 0));
    };

    SECTION("lvalue sequence")
    {
      check(iterate);
    }
    SECTION("const lvalue sequence")
    {
      check(std::as_const(iterate));
    }
    SECTION("rvalue sequence")
    {
      check(std::move(iterate));
    }
  }

  TEST_CASE("iterate - sum up an array ", "[sequence_senders][iterate]")
  {
    std::array<int, 3> array{42, 43, 44};
    int                sum     = 0;
    auto               iterate = exec::iterate(std::views::all(array));
    STATIC_REQUIRE(exec::sequence_sender_in<decltype(iterate), STDEXEC::env<>>);
    STATIC_REQUIRE(STDEXEC::__sender_for<decltype(iterate), exec::iterate_t>);
    auto op = exec::subscribe(iterate, sum_receiver<>{.sum_ = sum});
    STDEXEC::start(op);
    CHECK(sum == (42 + 43 + 44));
  }

  struct my_domain
  {
    template <STDEXEC::__sender_for<exec::iterate_t> Sender, class _Env>
    auto transform_sender(STDEXEC::start_t, Sender&& sender, _Env&&) const noexcept
    {
      auto range = STDEXEC::__get<1>(std::forward<Sender>(sender));
      auto sum   = std::accumulate(std::ranges::begin(range), std::ranges::end(range), 0);
      return STDEXEC::just(sum + 1);
    }
  };

  TEST_CASE("iterate - sum up an array with custom domain", "[sequence_senders][iterate]")
  {
    std::array<int, 3> array{42, 43, 44};
    auto               iterate = exec::iterate(std::views::all(array));
    STATIC_REQUIRE(exec::sequence_sender_in<decltype(iterate), STDEXEC::env<>>);
    STATIC_REQUIRE(STDEXEC::__sender_for<decltype(iterate), exec::iterate_t>);
    auto env  = STDEXEC::prop{STDEXEC::get_domain, my_domain{}};
    using Env = decltype(env);
    int  sum  = 0;
    auto op   = exec::subscribe(iterate, sum_receiver<Env>{.sum_ = sum, .env_ = env});
    STDEXEC::start(op);
    CHECK(sum == (42 + 43 + 44 + 1));
  }

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
  TEST_CASE("iterate - subscribe propagates begin exceptions", "[sequence_senders][iterate]")
  {
    auto iterate = exec::iterate(begin_throws_range{});
    int  sum     = 0;

    STATIC_REQUIRE_FALSE(noexcept(exec::subscribe(iterate, sum_receiver<>{.sum_ = sum})));
    CHECK_THROWS_AS(exec::subscribe(iterate, sum_receiver<>{.sum_ = sum}), begin_error);
  }
#endif  // !STDEXEC_NO_STDCPP_EXCEPTIONS()

  // A forward range over a fixed array that poisons its storage when
  // destroyed, so that a read through an iterator that dangles into a
  // destroyed sender is caught deterministically, without a sanitizer.
  struct poison_on_destroy_range
  {
    static constexpr int poison = -1;

    int values_[3]{};

    explicit poison_on_destroy_range(std::initializer_list<int> list) noexcept
    {
      std::size_t index = 0;
      for (int value: list)
      {
        values_[index++] = value;
      }
    }

    poison_on_destroy_range(poison_on_destroy_range const &) = default;

    ~poison_on_destroy_range()
    {
      for (int& value: values_)
      {
        value = poison;
      }
    }

    int* begin() noexcept
    {
      return values_;
    }

    int* end() noexcept
    {
      return values_ + 3;
    }

    int const * begin() const noexcept
    {
      return values_;
    }

    int const * end() const noexcept
    {
      return values_ + 3;
    }
  };

  // A forward range whose elements cannot be copied: a fix that always
  // copies the range instead of moving out of an rvalue sender fails to
  // compile against this range.
  struct move_only_range
  {
    std::vector<int> values_;

    explicit move_only_range(std::initializer_list<int> list)
      : values_(list)
    {}

    move_only_range(move_only_range&&)       = default;
    move_only_range(move_only_range const &) = delete;

    auto begin() noexcept
    {
      return values_.begin();
    }

    auto end() noexcept
    {
      return values_.end();
    }

    auto begin() const noexcept
    {
      return values_.begin();
    }

    auto end() const noexcept
    {
      return values_.end();
    }
  };

  TEST_CASE("iterate - operation state outlives the sequence sender", "[sequence_senders][iterate]")
  {
    using seq_t = decltype(exec::iterate(poison_on_destroy_range{}));

    int                  sum = 0;
    std::optional<seq_t> sndr;
    sndr.emplace(exec::iterate(poison_on_destroy_range{10, 20, 30}));
    auto op = exec::subscribe(*sndr, sum_receiver<>{.sum_ = sum});
    sndr.reset();  // the sender is gone; the operation state must not depend on it
    STDEXEC::start(op);
    CHECK(sum == 60);
  }

  TEST_CASE("iterate - rvalue subscribe moves an owning range", "[sequence_senders][iterate]")
  {
    int  sum = 0;
    auto op = exec::subscribe(exec::iterate(move_only_range{1, 2, 3}), sum_receiver<>{.sum_ = sum});
    STDEXEC::start(op);
    CHECK(sum == 6);
  }

  TEST_CASE("iterate - const lvalue subscribe copies an owning range",
            "[sequence_senders][iterate]")
  {
    using seq_t = decltype(exec::iterate(poison_on_destroy_range{}));

    int                  sum = 0;
    std::optional<seq_t> sndr;
    sndr.emplace(exec::iterate(poison_on_destroy_range{1, 2, 3}));
    auto op = exec::subscribe(std::as_const(*sndr), sum_receiver<>{.sum_ = sum});
    sndr.reset();
    STDEXEC::start(op);
    CHECK(sum == 6);
  }

}  // namespace
