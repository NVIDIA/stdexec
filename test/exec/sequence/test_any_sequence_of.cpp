/*
 * Copyright (c) 2023 NVIDIA Corporation
 * Copyright (c) 2023 Maikel Nadolski
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


#include "exec/sequence/any_sequence_of.hpp"
#include "exec/sequence/empty_sequence.hpp"

#include <catch2/catch.hpp>

template <class Receiver>
struct ignore_all_item_rcvr {
  using is_receiver = void;
  Receiver rcvr;

  friend stdexec::env_of_t<Receiver>
    tag_invoke(stdexec::get_env_t, const ignore_all_item_rcvr& self) noexcept {
    return stdexec::get_env(self.rcvr);
  }

  template <class... As>
  friend void tag_invoke(stdexec::set_value_t, ignore_all_item_rcvr&& self, As&&...) noexcept {
    stdexec::set_value(static_cast<Receiver&&>(self.rcvr));
  }

  friend void tag_invoke(stdexec::set_stopped_t, ignore_all_item_rcvr&& self) noexcept {
    stdexec::set_value(static_cast<Receiver&&>(self.rcvr));
  }

  template <class E>
  friend void tag_invoke(stdexec::set_error_t, ignore_all_item_rcvr&& self, E&&) noexcept {
    stdexec::set_value(static_cast<Receiver&&>(self.rcvr));
  }
};

template <class Item>
struct ignore_all_sender {
  using is_sender = void;
  using completion_signatures = stdexec::completion_signatures<stdexec::set_value_t()>;

  Item item_;

  template <
    stdexec::__decays_to<ignore_all_sender> Self,
    stdexec::receiver_of<completion_signatures> Receiver>
  friend auto tag_invoke(stdexec::connect_t, Self&& self, Receiver rcvr) noexcept {
    return stdexec::connect(
      static_cast<Self&&>(self).item_,
      ignore_all_item_rcvr<Receiver>{static_cast<Receiver&&>(rcvr)});
  }
};

struct ignore_all_receiver {
  using is_receiver = void;

  template <class Item>
  friend ignore_all_sender<stdexec::__decay_t<Item>>
    tag_invoke(exec::set_next_t, ignore_all_receiver&, Item&& item) noexcept {
    return {static_cast<Item&&>(item)};
  }

  friend void tag_invoke(stdexec::set_value_t, ignore_all_receiver&&) noexcept {
  }

  friend void tag_invoke(stdexec::set_stopped_t, ignore_all_receiver&&) noexcept {
  }

  friend void tag_invoke(stdexec::set_error_t, ignore_all_receiver&&, std::exception_ptr) noexcept {
  }

  friend stdexec::empty_env tag_invoke(stdexec::get_env_t, const ignore_all_receiver&) noexcept {
    return {};
  }
};

TEST_CASE(
  "any_sequence_of - works with empty_sequence",
  "[sequence_senders][any_sequence_of][empty_sequence]") {
  using Completions = stdexec::completion_signatures<stdexec::set_value_t(int)>;
  STATIC_REQUIRE(stdexec::constructible_from<
                 exec::any_sequence_receiver_ref<Completions>::any_sender<>,
                 decltype(exec::empty_sequence())>);
  exec::any_sequence_receiver_ref<Completions>::any_sender<> any_sequence = exec::empty_sequence();
  auto op = exec::subscribe(std::move(any_sequence), ignore_all_receiver{});
  stdexec::start(op);
}

TEST_CASE("any_sequence_of - works with just(42)", "[sequence_senders][any_sequence_of]") {
  using Completions = stdexec::completion_signatures<stdexec::set_value_t(int)>;
  STATIC_REQUIRE(stdexec::constructible_from<
                 exec::any_sequence_receiver_ref<Completions>::any_sender<>,
                 decltype(stdexec::just(42))>);
  exec::any_sequence_receiver_ref<Completions>::any_sender<> any_sequence = stdexec::just(42);
  auto op = exec::subscribe(std::move(any_sequence), ignore_all_receiver{});
  stdexec::start(op);
}

TEST_CASE("any_sequence_of - works with just()", "[sequence_senders][any_sequence_of]") {
  using CompletionsFalse = stdexec::completion_signatures<stdexec::set_value_t(int)>;
  using Completions = stdexec::completion_signatures<stdexec::set_value_t()>;
  STATIC_REQUIRE_FALSE(stdexec::constructible_from<
                       exec::any_sequence_receiver_ref<CompletionsFalse>::any_sender<>,
                       decltype(stdexec::just())>);
  STATIC_REQUIRE(stdexec::constructible_from<
                 exec::any_sequence_receiver_ref<Completions>::any_sender<>,
                 decltype(stdexec::just())>);
  exec::any_sequence_receiver_ref<Completions>::any_sender<> any_sequence = stdexec::just();
  auto op = exec::subscribe(std::move(any_sequence), ignore_all_receiver{});
  stdexec::start(op);
}

TEST_CASE("any_sequence_of - has an environment", "[sequence_senders][any_sequence_of]") {
  using Completions = stdexec::completion_signatures<stdexec::set_value_t()>;
  exec::any_sequence_receiver_ref<Completions>::any_sender<> any_sequence = stdexec::just();
  auto env = stdexec::get_env(any_sequence);
  using env_t = decltype(env);
  STATIC_REQUIRE(
    stdexec::same_as<
      env_t,
      stdexec::__t<exec::__any::__sender_env<Completions, stdexec::__types<>, stdexec::__types<>>>>);
}