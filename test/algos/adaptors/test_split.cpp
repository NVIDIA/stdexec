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

// TODO: implement split
// TEST_CASE("split returns a sender", "[adaptors][split]") {
//   auto snd = ex::split(ex::just(19));
//   static_assert(ex::sender<decltype(snd)>);
//   (void)snd;
// }
// TEST_CASE("split returns a typed_sender", "[adaptors][split]") {
//   auto snd = ex::split(ex::just(19));
//   static_assert(ex::typed_sender<decltype(snd), empty_env>);
//   (void)snd;
// }
// TEST_CASE("split simple example", "[adaptors][split]") {
//   bool called{false};
//   auto snd = ex::split(ex::just(19));
//   auto op1 = ex::connect(snd, expect_value_receiver<int>{19}, empty_env{});
//   auto op2 = ex::connect(snd, expect_void_receiver<int>{19}, empty_env{});
//   ex::start(op1);
//   ex::start(op2);
//   // The receiver will ensure that the right value is produced
// }
