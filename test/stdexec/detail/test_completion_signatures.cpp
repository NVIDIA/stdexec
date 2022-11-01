/*
 * Copyright (c) 2022 Lucian Radu Teodorescu
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
#include <stdexec/execution.hpp>
#include <test_common/schedulers.hpp>
namespace ex = stdexec;

using namespace std;

template <class... Vs>
using set_value_sig = ex::set_value_t(Vs...);

template <class E>
using set_error_sig = ex::set_error_t(E);

TEST_CASE(
    "set_value_sig can be used to transform value types to corresponding completion signatures",
    "[detail][completion_signatures]") {
  using set_value_f = stdexec::__q<set_value_sig>;

  using tr = stdexec::__transform<set_value_f, stdexec::__q<stdexec::__types>>;

  using res = stdexec::__minvoke<tr, int, double, string>;
  using expected = stdexec::__types<    //
      ex::set_value_t(int),    //
      ex::set_value_t(double), //
      ex::set_value_t(string)  //
      >;
  static_assert(is_same_v<res, expected>);
}

TEST_CASE(
    "set_error_sig can be used to transform error types to corresponding completion signatures",
    "[detail][completion_signatures]") {
  using set_error_f = stdexec::__q<set_error_sig>;

  using tr = stdexec::__transform<set_error_f, stdexec::__q<stdexec::__types>>;

  using res = stdexec::__minvoke<tr, exception_ptr, error_code, string>;
  using expected = stdexec::__types<           //
      ex::set_error_t(exception_ptr), //
      ex::set_error_t(error_code),    //
      ex::set_error_t(string)         //
      >;
  static_assert(is_same_v<res, expected>);
}

TEST_CASE(
    "set_error_sig can be used to transform exception_ptr", "[detail][completion_signatures]") {
  using set_error_f = stdexec::__q<set_error_sig>;

  using tr = stdexec::__transform<set_error_f, stdexec::__q<stdexec::__types>>;

  using res = stdexec::__minvoke<tr, exception_ptr>;
  using expected = stdexec::__types<          //
      ex::set_error_t(exception_ptr) //
      >;
  static_assert(is_same_v<res, expected>);
}

TEST_CASE(
    "__error_types_of_t gets the error types from sender", "[detail][completion_signatures]") {
  using snd_eptr_t = decltype(ex::just_error(exception_ptr{}));
  using snd_ec_t = decltype(ex::just_error(error_code{}));
  using snd_str_t = decltype(ex::just_error(std::string{}));
  using snd_tr_just_t = decltype(ex::transfer_just(error_scheduler{}));

  using err_types_eptr = stdexec::__error_types_of_t<snd_eptr_t>;
  using err_types_ec = stdexec::__error_types_of_t<snd_ec_t>;
  using err_types_str = stdexec::__error_types_of_t<snd_str_t>;
  using err_types_tr_just = stdexec::__error_types_of_t<snd_tr_just_t>;

  static_assert(is_same_v<err_types_eptr, variant<exception_ptr>>);
  static_assert(is_same_v<err_types_ec, variant<error_code>>);
  static_assert(is_same_v<err_types_str, variant<string>>);
  static_assert(is_same_v<err_types_tr_just, variant<exception_ptr>>);
}

TEST_CASE("__error_types_of_t can also transform error types", "[detail][completion_signatures]") {
  using snd_eptr_t = decltype(ex::just_error(exception_ptr{}));
  using snd_ec_t = decltype(ex::just_error(error_code{}));
  using snd_str_t = decltype(ex::just_error(std::string{}));

  using set_error_f = stdexec::__q<set_error_sig>;
  using tr = stdexec::__transform<set_error_f>;

  using sig_eptr = stdexec::__error_types_of_t<snd_eptr_t, ex::no_env, tr>;
  using sig_ec = stdexec::__error_types_of_t<snd_ec_t, ex::no_env, tr>;
  using sig_str = stdexec::__error_types_of_t<snd_str_t, ex::no_env, tr>;

  static_assert(is_same_v<sig_eptr, stdexec::__types<ex::set_error_t(exception_ptr)>>);
  static_assert(is_same_v<sig_ec, stdexec::__types<ex::set_error_t(error_code)>>);
  static_assert(is_same_v<sig_str, stdexec::__types<ex::set_error_t(string)>>);
}

template <typename CS, typename ExpectedValTypes>
void expect_val_types() {
  using t = typename CS::template __gather_sigs<ex::set_value_t, stdexec::__q<stdexec::__types>, stdexec::__q<stdexec::__types>>;
  static_assert(is_same_v<t, ExpectedValTypes>);
}
template <typename CS, typename ExpectedErrTypes>
void expect_err_types() {
  using t = typename CS::template __gather_sigs<ex::set_error_t, stdexec::__q<stdexec::__midentity>, stdexec::__q<stdexec::__types>>;
  static_assert(is_same_v<t, ExpectedErrTypes>);
}

TEST_CASE("make_completion_signatures can replicate the completion signatures of input senders",
    "[detail][completion_signatures]") {
  using snd_int_t = decltype(ex::just(0));
  using snd_double_char_t = decltype(ex::just(3.14, 'p'));
  using snd_eptr_t = decltype(ex::just_error(exception_ptr{}));
  using snd_ec_t = decltype(ex::just_error(error_code{}));
  using snd_stopped_t = decltype(ex::just_stopped());

  using cs_int = ex::make_completion_signatures<snd_int_t>;
  using cs_double_char = ex::make_completion_signatures<snd_double_char_t>;
  using cs_eptr = ex::make_completion_signatures<snd_eptr_t>;
  using cs_ec = ex::make_completion_signatures<snd_ec_t>;
  using cs_stopped = ex::make_completion_signatures<snd_stopped_t>;

  expect_val_types<cs_int, stdexec::__types<stdexec::__types<int>>>();
  expect_val_types<cs_double_char, stdexec::__types<stdexec::__types<double, char>>>();
  expect_val_types<cs_eptr, stdexec::__types<>>();
  expect_val_types<cs_ec, stdexec::__types<>>();
  expect_val_types<cs_stopped, stdexec::__types<>>();

  expect_err_types<cs_int, stdexec::__types<>>();
  expect_err_types<cs_double_char, stdexec::__types<>>();
  expect_err_types<cs_eptr, stdexec::__types<exception_ptr>>();
  expect_err_types<cs_ec, stdexec::__types<error_code>>();
  expect_err_types<cs_stopped, stdexec::__types<>>();
}

TEST_CASE("make_completion_signatures with no_env can replicate the completion signatures of input "
          "senders",
    "[detail][completion_signatures]") {
  using snd_int_t = decltype(ex::just(0));
  using snd_double_char_t = decltype(ex::just(3.14, 'p'));
  using snd_eptr_t = decltype(ex::just_error(exception_ptr{}));
  using snd_ec_t = decltype(ex::just_error(error_code{}));
  using snd_stopped_t = decltype(ex::just_stopped());

  using cs_int = ex::make_completion_signatures<snd_int_t, ex::no_env>;
  using cs_double_char = ex::make_completion_signatures<snd_double_char_t, ex::no_env>;
  using cs_eptr = ex::make_completion_signatures<snd_eptr_t, ex::no_env>;
  using cs_ec = ex::make_completion_signatures<snd_ec_t, ex::no_env>;
  using cs_stopped = ex::make_completion_signatures<snd_stopped_t, ex::no_env>;

  expect_val_types<cs_int, stdexec::__types<stdexec::__types<int>>>();
  expect_val_types<cs_double_char, stdexec::__types<stdexec::__types<double, char>>>();
  expect_val_types<cs_eptr, stdexec::__types<>>();
  expect_val_types<cs_ec, stdexec::__types<>>();
  expect_val_types<cs_stopped, stdexec::__types<>>();

  expect_err_types<cs_int, stdexec::__types<>>();
  expect_err_types<cs_double_char, stdexec::__types<>>();
  expect_err_types<cs_eptr, stdexec::__types<exception_ptr>>();
  expect_err_types<cs_ec, stdexec::__types<error_code>>();
  expect_err_types<cs_stopped, stdexec::__types<>>();
}

TEST_CASE("make_completion_signatures can add other error signatures",
    "[detail][completion_signatures]") {
  using snd_double_t = decltype(ex::just_error(std::exception_ptr{}));
  using cs_with_ec = ex::make_completion_signatures<snd_double_t, ex::no_env,
      ex::completion_signatures<ex::set_error_t(error_code)>>;

  expect_val_types<cs_with_ec, stdexec::__types<>>();
  expect_err_types<cs_with_ec, stdexec::__types<error_code, exception_ptr>>();
}

TEST_CASE("make_completion_signatures can add other error signatures, but will dedup them",
    "[detail][completion_signatures]") {
  using snd_double_t = decltype(ex::just(3.14));
  using cs_with_ec = ex::make_completion_signatures<snd_double_t, ex::no_env,
      ex::completion_signatures<ex::set_error_t(exception_ptr)>>;

  // exception_ptr appears only once
  expect_err_types<cs_with_ec, stdexec::__types<exception_ptr>>();
}

TEST_CASE("make_completion_signatures can add other value signatures",
    "[detail][completion_signatures]") {
  using snd_double_t = decltype(ex::just(3.14));
  using cs = ex::make_completion_signatures<snd_double_t, ex::no_env,
      ex::completion_signatures<  //
          ex::set_value_t(int),   //
          ex::set_value_t(double) //
          >>;

  // will add int, double will appear only once
  expect_val_types<cs, stdexec::__types<stdexec::__types<int>, stdexec::__types<double>>>();
}

template <class... Args>
using add_int_set_value_sig =
  ex::completion_signatures<ex::set_value_t(string, Args...)>;

template <class Err>
using optional_set_error_sig =
  ex::completion_signatures<ex::set_error_t(optional<Err>)>;

TEST_CASE("make_completion_signatures can transform value types signatures",
    "[detail][completion_signatures]") {
  using snd_double_t = decltype(ex::just(3.14));
  using cs = ex::make_completion_signatures<snd_double_t, ex::no_env,
      ex::completion_signatures<  //
          ex::set_value_t(int),   //
          ex::set_value_t(double) //
          >,                      //
      add_int_set_value_sig       //
      >;

  // will transform the original "double" into <string, double>
  // then will add the other "int" and "double"
  expect_val_types<cs, stdexec::__types<stdexec::__types<int>, stdexec::__types<double>, stdexec::__types<string, double>>>();
}

TEST_CASE("make_completion_signatures can transform error types signatures",
    "[detail][completion_signatures]") {
  using snd_double_t = decltype(ex::just_error(std::exception_ptr{}));
  using cs = ex::make_completion_signatures<snd_double_t, ex::no_env,
      ex::completion_signatures<                        //
          ex::set_error_t(error_code)                   //
          >,                                            //
      stdexec::__compl_sigs::__default_set_value,            //
      optional_set_error_sig>;

  // will transform the original "exception_ptr" into optional<exception_ptr>
  // then will add the other "error_code" as specified in the additional signatures
  expect_err_types<cs, stdexec::__types<error_code, optional<exception_ptr>>>();
}

template <template <class...> class Variant = stdexec::__types>
using my_error_types = Variant<exception_ptr>;

TEST_CASE("error_types_of_t can be used to get error types",
    "[detail][completion_signatures]") {
  using snd_t = decltype(ex::transfer_just(inline_scheduler{}, 1));
  using err_t = ex::error_types_of_t<snd_t, ex::no_env, stdexec::__types>;
  static_assert(is_same_v<err_t, stdexec::__types<>>);
}

TEST_CASE(
    "regression: error_types_of_t can be used to transform error types",
    "[detail][completion_signatures]") {
  using tr = stdexec::__transform<stdexec::__q<set_error_sig>>;

  using snd_t = decltype(ex::transfer_just(inline_scheduler{}, 1));
  using err_t =
      ex::error_types_of_t<snd_t, ex::no_env, tr::template __f>;
  static_assert(is_same_v<err_t, stdexec::__types<>>);
}
