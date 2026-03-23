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

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wunused-function")

namespace
{
  template <class Receiver>
  struct ignore_all_item_rcvr
  {
    using receiver_concept = STDEXEC::receiver_t;
    Receiver rcvr;

    [[nodiscard]]
    auto get_env() const noexcept -> STDEXEC::env_of_t<Receiver>
    {
      return STDEXEC::get_env(rcvr);
    }

    template <class... As>
    void set_value(As&&...) noexcept
    {
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
  struct ignore_all_sender
  {
    using sender_concept        = STDEXEC::sender_t;
    using completion_signatures = STDEXEC::completion_signatures<STDEXEC::set_value_t()>;

    Item item_;

    template <STDEXEC::__decays_to<ignore_all_sender>     Self,
              STDEXEC::receiver_of<completion_signatures> Receiver>
    STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this Self&& self, Receiver rcvr) noexcept
    {
      return STDEXEC::connect(static_cast<Self&&>(self).item_,
                              ignore_all_item_rcvr<Receiver>{static_cast<Receiver&&>(rcvr)});
    }
    STDEXEC_EXPLICIT_THIS_END(connect)
  };

  struct ignore_all_receiver
  {
    using receiver_concept = STDEXEC::receiver_t;

    template <class Item>
    auto set_next(Item&& item) noexcept -> ignore_all_sender<STDEXEC::__decay_t<Item>>
    {
      return {static_cast<Item&&>(item)};
    }

    void set_value() noexcept {}

    void set_stopped() noexcept {}

    void set_error(std::exception_ptr) noexcept {}
  };

  TEST_CASE("any_sequence_of - works with empty_sequence",
            "[sequence_senders][any_sequence_of][empty_sequence]")
  {
    using completions_t  = STDEXEC::completion_signatures<STDEXEC::set_value_t(int)>;
    using any_sequence_t = exec::any_sequence_receiver_ref<completions_t>::any_sender<>;
    STATIC_REQUIRE(std::constructible_from<any_sequence_t, decltype(exec::empty_sequence())>);
    STATIC_REQUIRE(exec::sequence_sender<any_sequence_t>);

    any_sequence_t any_sequence = exec::empty_sequence();
    auto           op           = exec::subscribe(std::move(any_sequence), ignore_all_receiver{});
    STDEXEC::start(op);
  }

  TEST_CASE("any_sequence_of - works with just(42)", "[sequence_senders][any_sequence_of]")
  {
    using completions_t      = STDEXEC::completion_signatures<STDEXEC::set_value_t(int)>;
    using any_receiver_ref_t = exec::any_sequence_receiver_ref<completions_t>;
    using any_sequence_t     = any_receiver_ref_t::any_sender<>;

    STATIC_REQUIRE(exec::sequence_sender_to<decltype(STDEXEC::just(42)), any_receiver_ref_t>);
    STATIC_REQUIRE(exec::sequence_sender<any_sequence_t>);
    STATIC_REQUIRE(std::constructible_from<any_sequence_t, decltype(STDEXEC::just(42))>);

    any_sequence_t any_sequence = STDEXEC::just(42);
    auto           op           = exec::subscribe(std::move(any_sequence), ignore_all_receiver{});
    STDEXEC::start(op);
  }

  TEST_CASE("any_sequence_of - works with just()", "[sequence_senders][any_sequence_of]")
  {
    using completions_false_t = STDEXEC::completion_signatures<STDEXEC::set_value_t(int)>;
    using completions_t       = STDEXEC::completion_signatures<STDEXEC::set_value_t()>;

    STATIC_REQUIRE_FALSE(
      std::constructible_from<exec::any_sequence_receiver_ref<completions_false_t>::any_sender<>,
                              decltype(STDEXEC::just())>);
    STATIC_REQUIRE(
      std::constructible_from<exec::any_sequence_receiver_ref<completions_t>::any_sender<>,
                              decltype(STDEXEC::just())>);
    exec::any_sequence_receiver_ref<completions_t>::any_sender<> any_sequence = STDEXEC::just();
    auto op = exec::subscribe(std::move(any_sequence), ignore_all_receiver{});
    STDEXEC::start(op);
  }

  TEST_CASE("any_sequence_of - has attributes", "[sequence_senders][any_sequence_of]")
  {
    using completions_t = STDEXEC::completion_signatures<STDEXEC::set_value_t()>;
    using queries_t =
      exec::queries<STDEXEC::inplace_stop_token(STDEXEC::get_stop_token_t) noexcept>;
    using any_seq_rcvr_t = exec::any_sequence_receiver<completions_t>;
    using any_seq_sndr_t = exec::any_sequence_sender<any_seq_rcvr_t, queries_t>;
    auto           attrs = STDEXEC::prop{STDEXEC::get_stop_token, STDEXEC::inplace_stop_token{}};
    any_seq_sndr_t any_sequence = exec::write_attrs(STDEXEC::just(), attrs);
    auto           token        = STDEXEC::get_stop_token(STDEXEC::get_env(any_sequence));

    STATIC_REQUIRE(std::same_as<decltype(token), STDEXEC::inplace_stop_token>);
    CHECK(token == STDEXEC::inplace_stop_token{});
  }
}  // namespace

STDEXEC_PRAGMA_POP()
