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

#if STDEXEC_HAS_STD_RANGES()

#  include <array>
#  include <catch2/catch.hpp>
#  include <numeric>

namespace {

  template <class Receiver>
  struct sum_item_rcvr {
    using receiver_concept = stdexec::receiver_t;
    Receiver rcvr;
    int* sum_;

    [[nodiscard]]
    auto get_env() const noexcept -> stdexec::env_of_t<Receiver> {
      return stdexec::get_env(rcvr);
    }

    template <class... As>
    void set_value(int x) noexcept {
      *sum_ += x;
      stdexec::set_value(static_cast<Receiver&&>(rcvr));
    }

    void set_stopped() noexcept {
      stdexec::set_value(static_cast<Receiver&&>(rcvr));
    }

    template <class E>
    void set_error(E&&) noexcept {
      stdexec::set_value(static_cast<Receiver&&>(rcvr));
    }
  };

  template <class Item>
  struct sum_sender {
    using sender_concept = stdexec::sender_t;
    using completion_signatures = stdexec::completion_signatures<stdexec::set_value_t()>;

    Item item_;
    int* sum_;

    template <
      stdexec::__decays_to<sum_sender> Self,
      stdexec::receiver_of<completion_signatures> Receiver
    >
    friend auto tag_invoke(stdexec::connect_t, Self&& self, Receiver rcvr) noexcept {
      return stdexec::connect(
        static_cast<Self&&>(self).item_,
        sum_item_rcvr<Receiver>{static_cast<Receiver&&>(rcvr), self.sum_});
    }
  };

  template <class Env = stdexec::env<>>
  struct sum_receiver {
    using receiver_concept = stdexec::receiver_t;

    int& sum_;
    Env env_{};

    template <class Item>
    friend auto tag_invoke(exec::set_next_t, sum_receiver& self, Item&& item) noexcept
      -> sum_sender<stdexec::__decay_t<Item>> {
      return {static_cast<Item&&>(item), &self.sum_};
    }

    void set_value() noexcept {
    }

    void set_stopped() noexcept {
    }

    void set_error(std::exception_ptr) noexcept {
    }

    [[nodiscard]]
    auto get_env() const noexcept -> Env {
      return env_;
    }
  };

  TEST_CASE("iterate - sum up an array ", "[sequence_senders][iterate]") {
    std::array<int, 3> array{42, 43, 44};
    int sum = 0;
    auto iterate = exec::iterate(std::views::all(array));
    STATIC_REQUIRE(exec::sequence_sender_in<decltype(iterate), stdexec::env<>>);
    STATIC_REQUIRE(stdexec::sender_expr_for<decltype(iterate), exec::iterate_t>);
    auto op = exec::subscribe(iterate, sum_receiver<>{.sum_ = sum});
    stdexec::start(op);
    CHECK(sum == (42 + 43 + 44));
  }

  struct my_domain {
    template <stdexec::sender_expr_for<exec::iterate_t> Sender, class _Env>
    auto transform_sender(Sender&& sender, _Env&&) const noexcept {
      auto range =
        stdexec::__sexpr_apply(std::forward<Sender>(sender), stdexec::__detail::__get_data{});
      auto sum = std::accumulate(std::ranges::begin(range), std::ranges::end(range), 0);
      return stdexec::just(sum + 1);
    }
  };

  TEST_CASE("iterate - sum up an array with custom domain", "[sequence_senders][iterate]") {
    std::array<int, 3> array{42, 43, 44};
    auto iterate = exec::iterate(std::views::all(array));
    STATIC_REQUIRE(exec::sequence_sender_in<decltype(iterate), stdexec::env<>>);
    STATIC_REQUIRE(stdexec::sender_expr_for<decltype(iterate), exec::iterate_t>);
    auto env = stdexec::prop{stdexec::get_domain, my_domain{}};
    using Env = decltype(env);
    int sum = 0;
    auto op = exec::subscribe(iterate, sum_receiver<Env>{.sum_ = sum, .env_ = env});
    stdexec::start(op);
    CHECK(sum == (42 + 43 + 44 + 1));
  }

} // namespace

#endif // STDEXEC_HAS_STD_RANGES()
