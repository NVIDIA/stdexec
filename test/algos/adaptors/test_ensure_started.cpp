/*
 * Copyright (c) Lucian Radu Teodorescu
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

#include <catch2/catch.hpp>
#include <execution.hpp>
#include <test_common/schedulers.hpp>
#include <test_common/receivers.hpp>
#include <test_common/type_helpers.hpp>

namespace ex = std::execution;

// TODO: implement ensure_started
// TEST_CASE("ensure_started returns a sender", "[adaptors][ensure_started]") {
//   auto snd = ex::ensure_started(ex::just(19));
//   static_assert(ex::sender<decltype(snd)>);
//   (void)snd;
// }
// TEST_CASE("ensure_started returns a typed_sender", "[adaptors][ensure_started]") {
//   auto snd = ex::ensure_started(ex::just(19), 8, [](int idx, int val) {});
//   static_assert(ex::typed_sender<decltype(snd)>);
//   (void)snd;
// }
// TEST_CASE("ensure_started simple example", "[adaptors][ensure_started]") {
//   bool called{false};
//   auto snd1 = ex::just() | ex::then([&] { called = true; });
//   CHECK_FALSE(called);
//   auto snd = ex::ensure_started(std::move(snd1));
//   CHECK(called);
//   auto op = ex::connect(std::move(snd), expect_void_receiver{});
//   ex::start(op);
// }
