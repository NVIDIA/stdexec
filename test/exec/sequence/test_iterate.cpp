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

#include <array>
#include <catch2/catch.hpp>

template <class Receiver>
struct sum_item_rcvr {
  using is_receiver = void;
  Receiver rcvr;
  int* sum_;

  friend stdexec::env_of_t<Receiver>
    tag_invoke(stdexec::get_env_t, const sum_item_rcvr& self) noexcept {
    return stdexec::get_env(self.rcvr);
  }

  template <class... As>
  friend void tag_invoke(stdexec::set_value_t, sum_item_rcvr&& self, int x) noexcept {
    *self.sum_ += x;
    stdexec::set_value(static_cast<Receiver&&>(self.rcvr));
  }

  friend void tag_invoke(stdexec::set_stopped_t, sum_item_rcvr&& self) noexcept {
    stdexec::set_value(static_cast<Receiver&&>(self.rcvr));
  }

  template <class E>
  friend void tag_invoke(stdexec::set_error_t, sum_item_rcvr&& self, E&&) noexcept {
    stdexec::set_value(static_cast<Receiver&&>(self.rcvr));
  }
};

template <class Item>
struct sum_sender {
  using is_sender = void;
  using completion_signatures = stdexec::completion_signatures<stdexec::set_value_t()>;

  Item item_;
  int* sum_;

  template <
    stdexec::__decays_to<sum_sender> Self,
    stdexec::receiver_of<completion_signatures> Receiver>
  friend auto tag_invoke(stdexec::connect_t, Self&& self, Receiver rcvr) noexcept {
    return stdexec::connect(
      static_cast<Self&&>(self).item_,
      sum_item_rcvr<Receiver>{static_cast<Receiver&&>(rcvr), self.sum_});
  }
};

template <class Env = stdexec::empty_env>
struct sum_receiver {
  using is_receiver = void;

  int& sum_;
  Env env_{};

  template <class Item>
  friend sum_sender<stdexec::__decay_t<Item>>
    tag_invoke(exec::set_next_t, sum_receiver& self, Item&& item) noexcept {
    return {static_cast<Item&&>(item), &self.sum_};
  }

  friend void tag_invoke(stdexec::set_value_t, sum_receiver&&) noexcept {
  }

  friend void tag_invoke(stdexec::set_stopped_t, sum_receiver&&) noexcept {
  }

  friend void tag_invoke(stdexec::set_error_t, sum_receiver&&, std::exception_ptr) noexcept {
  }

  friend Env tag_invoke(stdexec::get_env_t, const sum_receiver& self) noexcept {
    return self.env_;
  }
};

TEST_CASE("iterate - sum up an array ", "[sequence_senders][iterate]") {
  std::array<int, 3> array{42, 43, 44};
  int sum = 0;
  auto iterate = exec::iterate(std::views::all(array));
  STATIC_REQUIRE(exec::sequence_sender_in<decltype(iterate), stdexec::empty_env>);
  STATIC_REQUIRE(stdexec::sender_expr_for<decltype(iterate), exec::iterate_t>);
  auto op = exec::subscribe(iterate, sum_receiver<>{sum});
  stdexec::start(op);
  CHECK(sum == (42 + 43 + 44));
}

struct my_domain {
  template <stdexec::sender_expr_for<exec::iterate_t> Sender, class _Env>
  auto transform_sender(Sender&& sender, _Env&&) const noexcept {
    auto range = stdexec::apply_sender(
      std::forward<Sender>(sender), stdexec::__detail::__get_data{});
    auto sum = std::accumulate(std::ranges::begin(range), std::ranges::end(range), 0);
    return stdexec::just(sum + 1);
  }
};

TEST_CASE("iterate - sum up an array with custom domain", "[sequence_senders][iterate]") {
  std::array<int, 3> array{42, 43, 44};
  auto iterate = exec::iterate(std::views::all(array));
  STATIC_REQUIRE(exec::sequence_sender_in<decltype(iterate), stdexec::empty_env>);
  STATIC_REQUIRE(stdexec::sender_expr_for<decltype(iterate), exec::iterate_t>);
  auto env = exec::make_env(exec::with(stdexec::get_domain, my_domain{}));
  using Env = decltype(env);
  int sum = 0;
  auto op = exec::subscribe(iterate, sum_receiver<Env>{sum, env});
  stdexec::start(op);
  CHECK(sum == (42 + 43 + 44 + 1));
}

#endif // STDEXEC_HAS_STD_RANGES()
