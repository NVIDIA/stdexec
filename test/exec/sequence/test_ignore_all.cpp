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

#include "exec/sequence/ignore_all.hpp"

#include <catch2/catch.hpp>

using namespace stdexec;
using namespace exec;

struct rcvr {
  friend void tag_invoke(set_value_t, rcvr&&) noexcept;
  friend empty_env tag_invoke(get_env_t, rcvr) noexcept;
};

TEST_CASE("ignore_all - Works with a sender", "[ignore_all]") {
  auto sndr = ignore_all(just(42));
  using Sender = decltype(sndr);
  using Sigs = completion_signatures_of_t<Sender>;
  STATIC_REQUIRE(same_as<Sigs, completion_signatures<set_value_t()>>);
  STATIC_REQUIRE_FALSE(sequence_sender<Sender>);
  CHECK(sync_wait(sndr));
}

TEST_CASE("ignore_all - Forwards stop", "[ignore_all]") {
  auto sndr = ignore_all(just_stopped());
  CHECK_FALSE(sync_wait(sndr));
}

TEST_CASE("ignore_all - Forwards errors", "[ignore_all]") {
  auto sndr = ignore_all(just_error(42));
  CHECK_THROWS(sync_wait(sndr));
}