// /*
//  * Copyright (c) 2022 Lucian Radu Teodorescu
//  *
//  * Licensed under the Apache License Version 2.0 with LLVM Exceptions
//  * (the "License"); you may not use this file except in compliance with
//  * the License. You may obtain a copy of the License at
//  *
//  *   https://llvm.org/LICENSE.txt
//  *
//  * Unless required by applicable law or agreed to in writing, software
//  * distributed under the License is distributed on an "AS IS" BASIS,
//  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  * See the License for the specific language governing permissions and
//  * limitations under the License.
//  */

// #include <catch2/catch.hpp>
// #include <stdexec/execution.hpp>
// #include <test_common/receivers.hpp>

// namespace ex = stdexec;

// struct recv_no_set_value {
//   friend void tag_invoke(ex::set_stopped_t, recv_no_set_value) noexcept {}
//   friend void tag_invoke(ex::set_error_t, recv_no_set_value, std::exception_ptr) noexcept {}
//   friend ex::env<> tag_invoke(ex::get_env_t, const recv_no_set_value&) noexcept {
//     return {};
//   }
// };

// struct recv_set_value_except {
//   friend void tag_invoke(ex::set_value_t, recv_set_value_except) {}
//   friend void tag_invoke(ex::set_stopped_t, recv_set_value_except) noexcept {}
//   friend void tag_invoke(ex::set_error_t, recv_set_value_except, std::exception_ptr) noexcept {}
//   friend ex::env<> tag_invoke(ex::get_env_t, const recv_set_value_except&) noexcept {
//     return {};
//   }
// };

// struct recv_set_value_noexcept {
//   friend void tag_invoke(ex::set_value_t, recv_set_value_noexcept) noexcept {}
//   friend void tag_invoke(ex::set_stopped_t, recv_set_value_noexcept) noexcept {}
//   friend void tag_invoke(ex::set_error_t, recv_set_value_noexcept, std::exception_ptr) noexcept
//   {} friend ex::env<> tag_invoke(ex::get_env_t, const recv_set_value_noexcept&) noexcept {
//     return {};
//   }
// };

// struct recv_set_error_except {
//   friend void tag_invoke(ex::set_value_t, recv_set_error_except) noexcept {}
//   friend void tag_invoke(ex::set_stopped_t, recv_set_error_except) noexcept {}
//   friend void tag_invoke(ex::set_error_t, recv_set_error_except, std::exception_ptr) {
//     throw std::logic_error{"err"};
//   }
//   friend ex::env<> tag_invoke(ex::get_env_t, const recv_set_error_except&) noexcept {
//     return {};
//   }
// };
// struct recv_set_stopped_except {
//   friend void tag_invoke(ex::set_value_t, recv_set_stopped_except) noexcept {}
//   friend void tag_invoke(ex::set_stopped_t, recv_set_stopped_except) { throw
//   std::logic_error{"err"}; } friend void tag_invoke(ex::set_error_t, recv_set_stopped_except,
//   std::exception_ptr) noexcept {} friend ex::env<> tag_invoke(ex::get_env_t, const
//   recv_set_stopped_except&) noexcept {
//     return {};
//   }
// };

// struct recv_non_movable {
//   recv_non_movable() = default;
//   ~recv_non_movable() = default;
//   recv_non_movable(recv_non_movable&&) = delete;
//   recv_non_movable& operator=(recv_non_movable&&) = delete;
//   recv_non_movable(const recv_non_movable&) = default;
//   recv_non_movable& operator=(const recv_non_movable&) = default;

//   friend void tag_invoke(ex::set_value_t, recv_non_movable) noexcept {}
//   friend void tag_invoke(ex::set_stopped_t, recv_non_movable) noexcept {}
//   friend void tag_invoke(ex::set_error_t, recv_non_movable, std::exception_ptr) noexcept {}
//   friend ex::env<> tag_invoke(ex::get_env_t, const recv_non_movable&) noexcept {
//     return {};
//   }
// };

// TEST_CASE("receiver types satisfy the receiver concept", "[concepts][receiver]") {
//   using namespace empty_recv;

//   REQUIRE(ex::receiver<recv0>);
//   REQUIRE(ex::receiver<recv_int>);
//   REQUIRE(ex::receiver<recv0_ec>);
//   REQUIRE(ex::receiver<recv_int_ec>);
//   REQUIRE(ex::receiver<recv0_ec, std::error_code>);
//   REQUIRE(ex::receiver<recv_int_ec, std::error_code>);
//   REQUIRE(ex::receiver<expect_void_receiver>);
//   REQUIRE(ex::receiver<expect_void_receiver_ex>);
//   REQUIRE(ex::receiver<expect_value_receiver<ex::env<>, int>>);
//   REQUIRE(ex::receiver<expect_value_receiver<ex::env<>, double>>);
//   REQUIRE(ex::receiver<expect_stopped_receiver>);
//   REQUIRE(ex::receiver<expect_stopped_receiver_ex>);
//   REQUIRE(ex::receiver<expect_error_receiver>);
//   REQUIRE(ex::receiver<expect_error_receiver_ex>);
//   REQUIRE(ex::receiver<logging_receiver>);
// }

// TEST_CASE("receiver types satisfy the receiver_of concept", "[concepts][receiver]") {
//   using namespace empty_recv;

//   REQUIRE(ex::receiver_of<recv0>);
//   REQUIRE(ex::receiver_of<recv_int, int>);
//   REQUIRE(ex::receiver_of<recv0_ec>);
//   REQUIRE(ex::receiver_of<recv_int_ec, int>);
//   REQUIRE(ex::receiver_of<expect_void_receiver>);
//   REQUIRE(ex::receiver_of<expect_void_receiver_ex>);
//   REQUIRE(ex::receiver_of<expect_value_receiver<ex::env<>, int>, int>);
//   REQUIRE(ex::receiver_of<expect_value_receiver<ex::env<>, double>, double>);
//   REQUIRE(ex::receiver_of<expect_stopped_receiver, char>);
//   REQUIRE(ex::receiver_of<expect_stopped_receiver_ex, char>);
//   REQUIRE(ex::receiver_of<expect_error_receiver, char>);
//   REQUIRE(ex::receiver_of<expect_error_receiver_ex, char>);
//   REQUIRE(ex::receiver_of<logging_receiver>);
// }

// TEST_CASE(
//     "receiver type w/o set_value models receiver but not receiver_of", "[concepts][receiver]") {
//   REQUIRE(ex::receiver<recv_no_set_value>);
//   REQUIRE(!ex::receiver_of<recv_no_set_value>);
// }

// TEST_CASE("type with set_value noexcept is a receiver", "[concepts][receiver]") {
//   REQUIRE(ex::receiver<recv_set_value_noexcept>);
//   REQUIRE(ex::receiver_of<recv_set_value_noexcept>);
// }
// TEST_CASE("type with throwing set_value is not a receiver", "[concepts][receiver]") {
//   REQUIRE(!ex::receiver<recv_set_value_except>);
//   REQUIRE(!ex::receiver_of<recv_set_value_except>);
// }
// TEST_CASE("type with throwing set_error is not a receiver", "[concepts][receiver]") {
//   REQUIRE(!ex::receiver<recv_set_error_except>);
//   REQUIRE(!ex::receiver_of<recv_set_error_except>);
// }
// TEST_CASE("type with throwing set_stopped is not a receiver", "[concepts][receiver]") {
//   REQUIRE(!ex::receiver<recv_set_stopped_except>);
//   REQUIRE(!ex::receiver_of<recv_set_stopped_except>);
// }

// TEST_CASE("non-movable type is not a receiver", "[concepts][receiver]") {
//   REQUIRE(!ex::receiver<recv_non_movable>);
//   REQUIRE(!ex::receiver_of<recv_non_movable>);
// }
