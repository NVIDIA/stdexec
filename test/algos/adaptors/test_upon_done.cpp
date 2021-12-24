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

// TODO: implement upon_done
// TEST_CASE("then returns a sender", "[adaptors][upon_done]") {
//   auto snd = ex::upon_done(ex::just_done(), []() {});
//   static_assert(ex::sender<decltype(snd)>);
//   (void)snd;
// }
// TEST_CASE("then returns a typed_sender", "[adaptors][upon_done]") {
//   auto snd = ex::upon_done(ex::just_done(), []() {});
//   static_assert(ex::typed_sender<decltype(snd), empty_env>);
//   (void)snd;
// }
// TEST_CASE("then simple example", "[adaptors][upon_done]") {
//   bool called{false};
//   auto snd = ex::upon_done(ex::just_done(), [&]() { called = true; });
//   auto op = ex::connect(std::move(snd), expect_error_receiver{}, empty_env{});
//   ex::start(op);
//   // The receiver checks that it's called
//   // we also check that the function was invoked
//   CHECK(called);
// }
