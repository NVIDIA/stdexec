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
namespace ex = STDEXEC;

using namespace std;

namespace {

  template <class... Vs>
  using set_value_sig = ex::set_value_t(Vs...);

  template <class E>
  using set_error_sig = ex::set_error_t(E);

  TEST_CASE("test the internal __completion_signature concept", "[detail][completion_signatures]") {
    static_assert(ex::__completion_signature<ex::set_value_t()>);
    static_assert(ex::__completion_signature<ex::set_value_t(int)>);
    static_assert(ex::__completion_signature<ex::set_value_t(int, int)>);
    static_assert(!ex::__completion_signature<ex::set_value_t(int, int, ...)>);
    static_assert(!ex::__completion_signature<ex::set_value_t>);
    static_assert(!ex::__completion_signature<ex::set_value_t(int) noexcept>);

    static_assert(!ex::__completion_signature<ex::set_error_t()>);
    static_assert(ex::__completion_signature<ex::set_error_t(int)>);
    static_assert(!ex::__completion_signature<ex::set_error_t(int, int)>);
    static_assert(!ex::__completion_signature<ex::set_error_t(int, ...)>);
    static_assert(!ex::__completion_signature<ex::set_error_t(int) noexcept>);

    static_assert(ex::__completion_signature<ex::set_stopped_t()>);
    static_assert(!ex::__completion_signature<ex::set_stopped_t(int)>);
    static_assert(!ex::__completion_signature<ex::set_stopped_t(...)>);
    static_assert(!ex::__completion_signature<ex::set_stopped_t() noexcept>);
  }

  TEST_CASE(
    "set_value_sig can be used to transform value types to corresponding completion signatures",
    "[detail][completion_signatures]") {
    using set_value_f = ex::__q<set_value_sig>;

    using tr = ex::__mtransform<set_value_f, ex::__q<ex::__mlist>>;

    using res = ex::__minvoke<tr, int, double, string>;
    using expected =
      ex::__mlist<ex::set_value_t(int), ex::set_value_t(double), ex::set_value_t(string)>;
    static_assert(is_same_v<res, expected>);
  }

  TEST_CASE(
    "set_error_sig can be used to transform error types to corresponding completion signatures",
    "[detail][completion_signatures]") {
    using set_error_f = ex::__q<set_error_sig>;

    using tr = ex::__mtransform<set_error_f, ex::__q<ex::__mlist>>;

    using res = ex::__minvoke<tr, exception_ptr, error_code, string>;
    using expected = ex::__mlist<
      ex::set_error_t(exception_ptr),
      ex::set_error_t(error_code),
      ex::set_error_t(string)
    >;
    static_assert(is_same_v<res, expected>);
  }

  TEST_CASE(
    "set_error_sig can be used to transform exception_ptr",
    "[detail][completion_signatures]") {
    using set_error_f = ex::__q<set_error_sig>;

    using tr = ex::__mtransform<set_error_f, ex::__q<ex::__mlist>>;

    using res = ex::__minvoke<tr, exception_ptr>;
    using expected = ex::__mlist<ex::set_error_t(exception_ptr)>;
    static_assert(is_same_v<res, expected>);
  }

  TEST_CASE(
    "__error_types_of_t gets the error types from sender",
    "[detail][completion_signatures]") {
    using snd_eptr_t = decltype(ex::just_error(exception_ptr{}));
    using snd_ec_t = decltype(ex::just_error(error_code{}));
    using snd_str_t = decltype(ex::just_error(std::string{}));
    using snd_tr_just_t = decltype(ex::transfer_just(error_scheduler{}));

    using err_types_eptr = ex::__error_types_of_t<snd_eptr_t>;
    using err_types_ec = ex::__error_types_of_t<snd_ec_t>;
    using err_types_str = ex::__error_types_of_t<snd_str_t>;
    using err_types_tr_just = ex::__error_types_of_t<snd_tr_just_t>;

    static_assert(is_same_v<err_types_eptr, variant<exception_ptr>>);
    static_assert(is_same_v<err_types_ec, variant<error_code>>);
    static_assert(is_same_v<err_types_str, variant<string>>);
    static_assert(is_same_v<err_types_tr_just, variant<exception_ptr>>);
  }

  TEST_CASE(
    "__error_types_of_t can also transform error types",
    "[detail][completion_signatures]") {
    using snd_eptr_t = decltype(ex::just_error(exception_ptr{}));
    using snd_ec_t = decltype(ex::just_error(error_code{}));
    using snd_str_t = decltype(ex::just_error(std::string{}));

    using set_error_f = ex::__q<set_error_sig>;
    using tr = ex::__mtransform<set_error_f>;

    using sig_eptr = ex::__error_types_of_t<snd_eptr_t, ex::env<>, tr>;
    using sig_ec = ex::__error_types_of_t<snd_ec_t, ex::env<>, tr>;
    using sig_str = ex::__error_types_of_t<snd_str_t, ex::env<>, tr>;

    static_assert(is_same_v<sig_eptr, ex::__mlist<ex::set_error_t(exception_ptr)>>);
    static_assert(is_same_v<sig_ec, ex::__mlist<ex::set_error_t(error_code)>>);
    static_assert(is_same_v<sig_str, ex::__mlist<ex::set_error_t(string)>>);
  }

  template <class CS, class... Expected>
  void expect_val_types() {
    using expected_t = ex::__mmake_set<Expected...>;
    using actual_t =
      ex::__gather_completions_t<ex::set_value_t, CS, ex::__q<ex::__mlist>, ex::__q<ex::__mset>>;
    static_assert(ex::__mset_eq<actual_t, expected_t>);
  }

  template <class CS, class... Expected>
  void expect_err_types() {
    using expected_t = ex::__mmake_set<Expected...>;
    using actual_t =
      ex::__gather_completions_t<ex::set_error_t, CS, ex::__q<ex::__midentity>, ex::__q<ex::__mset>>;
    static_assert(ex::__mset_eq<actual_t, expected_t>);
  }

  TEST_CASE(
    "transform_completion_signatures_of can replicate the completion signatures of input senders",
    "[detail][completion_signatures]") {
    using snd_int_t = decltype(ex::just(0));
    using snd_double_char_t = decltype(ex::just(3.14, 'p'));
    using snd_eptr_t = decltype(ex::just_error(exception_ptr{}));
    using snd_ec_t = decltype(ex::just_error(error_code{}));
    using snd_stopped_t = decltype(ex::just_stopped());

    using cs_int = ex::transform_completion_signatures_of<snd_int_t>;
    using cs_double_char = ex::transform_completion_signatures_of<snd_double_char_t>;
    using cs_eptr = ex::transform_completion_signatures_of<snd_eptr_t>;
    using cs_ec = ex::transform_completion_signatures_of<snd_ec_t>;
    using cs_stopped = ex::transform_completion_signatures_of<snd_stopped_t>;

    expect_val_types<cs_int, ex::__mlist<int>>();
    expect_val_types<cs_double_char, ex::__mlist<double, char>>();
    expect_val_types<cs_eptr>();
    expect_val_types<cs_ec>();
    expect_val_types<cs_stopped>();

    expect_err_types<cs_int>();
    expect_err_types<cs_double_char>();
    expect_err_types<cs_eptr, exception_ptr>();
    expect_err_types<cs_ec, error_code>();
    expect_err_types<cs_stopped>();
  }

  TEST_CASE(
    "transform_completion_signatures_of with ex::env<> can replicate the completion signatures of "
    "input "
    "senders",
    "[detail][completion_signatures]") {
    using snd_int_t = decltype(ex::just(0));
    using snd_double_char_t = decltype(ex::just(3.14, 'p'));
    using snd_eptr_t = decltype(ex::just_error(exception_ptr{}));
    using snd_ec_t = decltype(ex::just_error(error_code{}));
    using snd_stopped_t = decltype(ex::just_stopped());

    using cs_int = ex::transform_completion_signatures_of<snd_int_t, ex::env<>>;
    using cs_double_char = ex::transform_completion_signatures_of<snd_double_char_t, ex::env<>>;
    using cs_eptr = ex::transform_completion_signatures_of<snd_eptr_t, ex::env<>>;
    using cs_ec = ex::transform_completion_signatures_of<snd_ec_t, ex::env<>>;
    using cs_stopped = ex::transform_completion_signatures_of<snd_stopped_t, ex::env<>>;

    expect_val_types<cs_int, ex::__mlist<int>>();
    expect_val_types<cs_double_char, ex::__mlist<double, char>>();
    expect_val_types<cs_eptr>();
    expect_val_types<cs_ec>();
    expect_val_types<cs_stopped>();

    expect_err_types<cs_int>();
    expect_err_types<cs_double_char>();
    expect_err_types<cs_eptr, exception_ptr>();
    expect_err_types<cs_ec, error_code>();
    expect_err_types<cs_stopped>();
  }

  TEST_CASE(
    "transform_completion_signatures_of can add other error signatures",
    "[detail][completion_signatures]") {
    using snd_double_t = decltype(ex::just_error(std::exception_ptr{}));
    using cs_with_ec = ex::transform_completion_signatures_of<
      snd_double_t,
      ex::env<>,
      ex::completion_signatures<ex::set_error_t(error_code)>
    >;

    expect_val_types<cs_with_ec>();
    expect_err_types<cs_with_ec, error_code, exception_ptr>();
  }

  TEST_CASE(
    "transform_completion_signatures_of can add other error signatures, but will dedup them",
    "[detail][completion_signatures]") {
    using snd_double_t = decltype(ex::just(3.14));
    using cs_with_ec = ex::transform_completion_signatures_of<
      snd_double_t,
      ex::env<>,
      ex::completion_signatures<ex::set_error_t(exception_ptr)>
    >;

    // exception_ptr appears only once
    expect_err_types<cs_with_ec, exception_ptr>();
  }

  TEST_CASE(
    "transform_completion_signatures_of can add other value signatures",
    "[detail][completion_signatures]") {
    using snd_double_t = decltype(ex::just(3.14));
    using cs = ex::transform_completion_signatures_of<
      snd_double_t,
      ex::env<>,
      ex::completion_signatures<ex::set_value_t(int), ex::set_value_t(double)>
    >;

    // will add int, double will appear only once
    expect_val_types<cs, ex::__mlist<int>, ex::__mlist<double>>();
  }

  template <class... Args>
  using add_int_set_value_sig = ex::completion_signatures<ex::set_value_t(string, Args...)>;

  template <class Err>
  using optional_set_error_sig = ex::completion_signatures<ex::set_error_t(optional<Err>)>;

  TEST_CASE(
    "transform_completion_signatures_of can transform value types signatures",
    "[detail][completion_signatures]") {
    using snd_double_t = decltype(ex::just(3.14));
    using cs = ex::transform_completion_signatures_of<
      snd_double_t,
      ex::env<>,
      ex::completion_signatures<ex::set_value_t(int), ex::set_value_t(double)>,
      add_int_set_value_sig
    >;

    // will transform the original "double" into <string, double>
    // then will add the other "int" and "double"
    expect_val_types<cs, ex::__mlist<int>, ex::__mlist<double>, ex::__mlist<string, double>>();
  }

  TEST_CASE(
    "transform_completion_signatures_of can transform error types signatures",
    "[detail][completion_signatures]") {
    using snd_eptr_t = decltype(ex::just_error(std::exception_ptr{}));
    using cs = ex::transform_completion_signatures_of<
      snd_eptr_t,
      ex::env<>,
      ex::completion_signatures<ex::set_error_t(error_code)>,
      ex::__cmplsigs::__default_set_value,
      optional_set_error_sig
    >;

    // will transform the original "exception_ptr" into optional<exception_ptr>
    // then will add the other "error_code" as specified in the additional signatures
    expect_err_types<cs, error_code, optional<exception_ptr>>();
  }

  TEST_CASE("error_types_of_t can be used to get error types", "[detail][completion_signatures]") {
    using snd_t = decltype(ex::when_all(
      ex::just_error(std::error_code{}), ex::just_error(std::exception_ptr{})));
    using actual_t = ex::error_types_of_t<snd_t, ex::env<>, ex::__mset>;
    using expected_t = ex::__mset<std::error_code, std::exception_ptr>;
    static_assert(ex::__mset_eq<actual_t, expected_t>);
  }

  TEST_CASE(
    "regression: error_types_of_t can be used to transform error types",
    "[detail][completion_signatures]") {
    using tr = ex::__mtransform<ex::__q<optional>, ex::__q<ex::__mset>>;

    using snd_t = decltype(ex::when_all(
      ex::just_error(std::error_code{}), ex::just_error(std::exception_ptr{})));
    using actual_t = ex::error_types_of_t<snd_t, ex::env<>, tr::__f>;
    using expected_t = ex::__mset<optional<std::error_code>, optional<std::exception_ptr>>;
    static_assert(ex::__mset_eq<actual_t, expected_t>);
  }
} // namespace
