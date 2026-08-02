/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *                         Copyright (c) 2025 Robert Leahy. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <catch2/catch_all.hpp>
#include <stdexec/execution.hpp>

#include <cstddef>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <variant>

#include <test_common/receivers.hpp>
#include <test_common/type_helpers.hpp>

namespace
{

  template <typename Completion>
  concept stored_completion = requires { typename std::remove_cvref_t<Completion>::tag_type; };

  template <typename Completion, typename Tag>
  concept completion_with_tag =
    stored_completion<Completion>
    && std::is_same_v<typename std::remove_cvref_t<Completion>::tag_type, Tag>;

  template <typename... Fns>
  struct overloaded : Fns...
  {
    using Fns::operator()...;
  };

  template <typename... Fns>
  overloaded(Fns...) -> overloaded<Fns...>;

  template <typename T>
  concept tuple_size_defined_for = requires { typename std::tuple_size<T>::type; };

  template <typename T>
  concept tuple_element_defined_for = requires { typename std::tuple_element<0, T>::type; };

  struct monostate_may_throw
  {
    void operator()(std::monostate) const {}

    template <typename Completion>
      requires completion_with_tag<Completion, ::STDEXEC::set_value_t>
    void operator()(Completion&&) const noexcept
    {}
  };

  static_assert(std::is_same_v<::STDEXEC::set_value_t(int), ::STDEXEC::set_value_t(int const)>);
  static_assert(std::is_same_v<::STDEXEC::set_value_t(int*), ::STDEXEC::set_value_t(int[])>);
  static_assert(
    std::is_same_v<::STDEXEC::set_value_t(int (*)(int)), ::STDEXEC::set_value_t(int(int))>);

  TEST_CASE("A single stored completion exposes its tag and arguments",
            "[storage_for_completion_signatures]")
  {
    using completion =
      ::STDEXEC::storage_for_completion_signature<::STDEXEC::set_value_t(int, int&)>;

    static_assert(!std::derived_from<completion, std::tuple<int, int&>>);
    static_assert(!tuple_size_defined_for<completion>);
    static_assert(!tuple_element_defined_for<completion>);
    static_assert(std::is_same_v<completion::tag_type, ::STDEXEC::set_value_t>);
    static_assert(std::is_same_v<completion::signature_type, ::STDEXEC::set_value_t(int, int&)>);
    static_assert(
      std::is_same_v<completion::__normalized_signature_t, ::STDEXEC::set_value_t(int&&, int&)>);
    static_assert(std::is_constructible_v<completion, ::STDEXEC::set_value_t, int, int&>);
    static_assert(!std::is_constructible_v<completion, int, int&>);
    static_assert(std::is_same_v<decltype(completion::tag()), ::STDEXEC::set_value_t>);
    static_assert(noexcept(completion::tag()));
    static_assert(std::is_same_v<decltype(std::declval<completion&>().forward_arguments()),
                                 std::tuple<int&, int&>>);
    static_assert(std::is_same_v<decltype(std::declval<completion const &>().forward_arguments()),
                                 std::tuple<int const &, int&>>);
    static_assert(std::is_same_v<decltype(std::declval<completion&&>().forward_arguments()),
                                 std::tuple<int&&, int&>>);
    static_assert(std::is_same_v<decltype(std::declval<completion const &&>().forward_arguments()),
                                 std::tuple<int const &&, int&>>);
    static_assert(std::is_same_v<decltype(std::declval<completion&>().__forward_arguments()),
                                 ::STDEXEC::__tuple<int&, int&>>);
    static_assert(std::is_same_v<decltype(std::declval<completion const &>().__forward_arguments()),
                                 ::STDEXEC::__tuple<int const &, int&>>);
    static_assert(std::is_same_v<decltype(std::declval<completion&&>().__forward_arguments()),
                                 ::STDEXEC::__tuple<int&&, int&>>);
    static_assert(
      std::is_same_v<decltype(std::declval<completion const &&>().__forward_arguments()),
                     ::STDEXEC::__tuple<int const &&, int&>>);
    using internal_arguments = decltype(std::declval<completion&&>().__forward_arguments());
    static_assert(
      std::is_same_v<decltype(::STDEXEC::__get<0>(std::declval<internal_arguments&&>())), int&&>);
    static_assert(
      std::is_same_v<decltype(::STDEXEC::__get<1>(std::declval<internal_arguments&&>())), int&>);

    int        referenced = 42;
    completion c{::STDEXEC::set_value, 13, referenced};
    auto       args = c.forward_arguments();
    CHECK(std::get<0>(args) == 13);
    CHECK(&std::get<1>(args) == &referenced);

    auto const & cc      = c;
    auto         cc_args = cc.forward_arguments();
    std::get<1>(cc_args) = 99;
    CHECK(referenced == 99);

    ::STDEXEC::__apply(
      [](auto&& first, auto&& second)
      {
        static_assert(std::is_same_v<decltype(first), int&&>);
        static_assert(std::is_same_v<decltype(second), int&>);
      },
      std::move(c).__forward_arguments());

    int                                                                       other = 7;
    ::STDEXEC::storage_for_completion_signature<::STDEXEC::set_value_t(int&)> c2{
      ::STDEXEC::set_value,
      other};
    bool invoked = false;
    std::apply(
      [&](auto&&... values)
      {
        CHECK(!invoked);
        invoked = true;
        CHECK(sizeof...(values) == 3);
      },
      std::tuple_cat(std::move(c).forward_arguments(), c2.forward_arguments()));
    CHECK(invoked);
  }

  TEST_CASE("An internal argument tuple forwards more than eight values",
            "[storage_for_completion_signatures]")
  {
    using completion = ::STDEXEC::storage_for_completion_signature<
      ::STDEXEC::set_value_t(int, int, int, int, int, int, int, int, int)>;

    completion c{::STDEXEC::set_value, 0, 1, 2, 3, 4, 5, 6, 7, 8};
    ::STDEXEC::__apply(
      []<typename... _Ts>(_Ts&&... values)
      {
        static_assert(sizeof...(_Ts) == 9);
        static_assert((std::is_same_v<_Ts, int> && ...));
        CHECK((values + ...) == 36);
      },
      std::move(c).__forward_arguments());
  }

  template <typename Storage, typename Value>
  concept can_arrive_set_value_with = requires(Storage& s, Value&& value) {
    s.arrive(::STDEXEC::set_value, static_cast<Value&&>(value));
  };

  template <typename Storage, typename Error>
  concept can_arrive_set_error_with = requires(Storage& s, Error&& error) {
    s.arrive(::STDEXEC::set_error, static_cast<Error&&>(error));
  };

  struct copyable_value
  {
    copyable_value()                                = default;
    copyable_value(copyable_value const &) noexcept = default;
    copyable_value(copyable_value&&) noexcept       = default;
  };

  using copyable_value_storage = ::STDEXEC::storage_for_completion_signatures<
    ::STDEXEC::completion_signatures<::STDEXEC::set_value_t(copyable_value)>>;

  static_assert(!can_arrive_set_value_with<copyable_value_storage, copyable_value&>);
  static_assert(can_arrive_set_value_with<copyable_value_storage, copyable_value>);

  TEST_CASE("Storing no completion signatures works", "[storage_for_completion_signatures]")
  {
    ::STDEXEC::storage_for_completion_signatures<::STDEXEC::completion_signatures<>> storage;
    static_assert(std::is_empty_v<decltype(storage)>);
    static_assert(
      std::is_same_v<decltype(storage)::completion_signatures, ::STDEXEC::completion_signatures<>>);
    static_assert(noexcept(storage.has_completion()));
    CHECK(!storage.has_completion());
    bool visited_empty = false;
    ::STDEXEC::visit_stored_completion([&] { visited_empty = true; }, std::move(storage));
    CHECK(visited_empty);
    struct receiver
    {
      using receiver_concept = ::STDEXEC::receiver_t;
      explicit receiver(bool& moved) noexcept
        : moved_(&moved)
      {}
      receiver(receiver&& other) noexcept
        : moved_(std::exchange(other.moved_, nullptr))
      {
        *moved_ = true;
      }
      void set_value() noexcept
      {
        FAIL("Unexpected value");
      }
      bool* moved_;
    };
    bool moved = false;
    CHECK(!std::move(storage).complete(receiver{moved}));
    CHECK(!moved);
  }

  TEST_CASE("Storing simple completion signatures and then visiting them works",
            "[storage_for_completion_signatures]")
  {
    using completion_signatures =
      ::STDEXEC::completion_signatures<::STDEXEC::set_value_t(int),
                                       ::STDEXEC::set_stopped_t(),
                                       ::STDEXEC::set_error_t(std::error_code)>;
    using storage = ::STDEXEC::storage_for_completion_signatures<completion_signatures>;
    static_assert(set_equivalent<completion_signatures, storage::completion_signatures>);
    {
      storage s;
      CHECK(!s.has_completion());
      bool visited_empty = false;
      ::STDEXEC::visit_stored_completion(overloaded{[&] { visited_empty = true; },
                                                    [](auto&&) { FAIL("Unexpected completion"); }},
                                         std::move(s));
      CHECK(visited_empty);
    }
    {
      storage s;
      CHECK(!s.has_completion());
      static_assert(noexcept(s.arrive(::STDEXEC::set_stopped)));
      s.arrive(::STDEXEC::set_stopped);
      CHECK(s.has_completion());
      static_assert(noexcept(
        ::STDEXEC::visit_stored_completion(overloaded{[]() noexcept {}, [](auto&&) noexcept {}},
                                           std::move(s))));
      static_assert(!noexcept(
        ::STDEXEC::visit_stored_completion(overloaded{[] {}, [](auto&&) {}}, std::move(s))));
      bool invoked = false;
      ::STDEXEC::visit_stored_completion(
        overloaded{
          [] { FAIL("Unexpected completion"); },
          [&](auto&& completion)
          {
            CHECK(!invoked);
            invoked = true;
            if constexpr (completion_with_tag<decltype(completion), ::STDEXEC::set_stopped_t>)
            {
              CHECK(std::tuple_size_v<decltype(completion.forward_arguments())> == 0);
            }
            else
            {
              FAIL("Unexpected completion");
            }
          }},
        std::move(s));
      CHECK(invoked);
    }
    {
      storage s;
      CHECK(!s.has_completion());
      static_assert(noexcept(s.arrive(::STDEXEC::set_value, 5)));
      s.arrive(::STDEXEC::set_value, 5);
      CHECK(s.has_completion());
      static_assert(!noexcept(
        ::STDEXEC::visit_stored_completion(overloaded{[] {}, [](auto&&) {}}, std::move(s))));
      bool invoked = false;
      ::STDEXEC::visit_stored_completion(
        overloaded{
          [] { FAIL("Unexpected completion"); },
          [&](auto&& completion)
          {
            CHECK(!invoked);
            invoked = true;
            if constexpr (completion_with_tag<decltype(completion), ::STDEXEC::set_value_t>)
            {
              auto args = static_cast<decltype(completion)&&>(completion).forward_arguments();
              CHECK(std::get<0>(args) == 5);
            }
            else
            {
              FAIL("Unexpected completion");
            }
          }},
        std::move(s));
      CHECK(invoked);
    }
    {
      storage s;
      CHECK(!s.has_completion());
      static_assert(noexcept(s.arrive(::STDEXEC::set_error, std::error_code{})));
      s.arrive(::STDEXEC::set_error, make_error_code(std::errc::no_such_file_or_directory));
      CHECK(s.has_completion());
      static_assert(!noexcept(
        ::STDEXEC::visit_stored_completion(overloaded{[] {}, [](auto&&) {}}, std::move(s))));
      bool invoked = false;
      ::STDEXEC::visit_stored_completion(
        overloaded{
          [] { FAIL("Unexpected completion"); },
          [&](auto&& completion)
          {
            CHECK(!invoked);
            invoked = true;
            if constexpr (completion_with_tag<decltype(completion), ::STDEXEC::set_error_t>)
            {
              auto args = static_cast<decltype(completion)&&>(completion).forward_arguments();
              CHECK(std::get<0>(args) == make_error_code(std::errc::no_such_file_or_directory));
            }
            else
            {
              FAIL("Unexpected completion");
            }
          }},
        std::move(s));
      CHECK(invoked);
    }
  }

  TEST_CASE("Stored completions can be visited as completion objects",
            "[storage_for_completion_signatures]")
  {
    using storage = ::STDEXEC::storage_for_completion_signatures<
      ::STDEXEC::completion_signatures<::STDEXEC::set_value_t(int, int&)>>;

    int     referenced = 42;
    storage s;
    s.arrive(::STDEXEC::set_value, 13, referenced);

    bool invoked = false;
    ::STDEXEC::visit_stored_completion(
      overloaded{[] { FAIL("Unexpected completion"); },
                 [&](auto&& completion)
                 {
                   CHECK(!invoked);
                   invoked = true;
                   if constexpr (completion_with_tag<decltype(completion), ::STDEXEC::set_value_t>)
                   {
                     auto args =
                       static_cast<decltype(completion)&&>(completion).forward_arguments();
                     CHECK(std::get<0>(args) == 13);
                     CHECK(&std::get<1>(args) == &referenced);
                   }
                   else
                   {
                     FAIL("Unexpected completion");
                   }
                 }},
      s);
    CHECK(invoked);

    auto const & cs = s;
    ::STDEXEC::visit_stored_completion(
      overloaded{[] { FAIL("Unexpected completion"); },
                 [&](auto&& completion)
                 {
                   if constexpr (completion_with_tag<decltype(completion), ::STDEXEC::set_value_t>)
                   {
                     static_assert(std::is_const_v<std::remove_reference_t<decltype(completion)>>);
                     auto args         = completion.forward_arguments();
                     std::get<1>(args) = 99;
                   }
                   else
                   {
                     FAIL("Unexpected completion");
                   }
                 }},
      cs);
    CHECK(referenced == 99);
  }

  TEST_CASE("Multiple stored completions can be visited together",
            "[storage_for_completion_signatures]")
  {
    using first_storage = ::STDEXEC::storage_for_completion_signatures<
      ::STDEXEC::completion_signatures<::STDEXEC::set_value_t(int)>>;
    using second_storage = ::STDEXEC::storage_for_completion_signatures<
      ::STDEXEC::completion_signatures<::STDEXEC::set_error_t(std::exception_ptr)>>;
    using third_storage = ::STDEXEC::storage_for_completion_signatures<
      ::STDEXEC::completion_signatures<::STDEXEC::set_value_t(int&)>>;

    int            referenced = 42;
    first_storage  first;
    second_storage second;
    third_storage  third;
    first.arrive(::STDEXEC::set_value, 13);
    second.arrive(::STDEXEC::set_error, std::exception_ptr{});
    third.arrive(::STDEXEC::set_value, referenced);

    bool invoked = false;
    ::STDEXEC::visit_stored_completion(
      overloaded{
        [] { FAIL("Unexpected completion"); },
        [&](auto&& first_completion, auto&& second_completion, auto&& third_completion)
        {
          CHECK(!invoked);
          invoked = true;
          if constexpr (completion_with_tag<decltype(first_completion), ::STDEXEC::set_value_t>
                        && completion_with_tag<decltype(second_completion), ::STDEXEC::set_error_t>
                        && completion_with_tag<decltype(third_completion), ::STDEXEC::set_value_t>)
          {
            auto args = std::tuple_cat(
              static_cast<decltype(first_completion)&&>(first_completion).forward_arguments(),
              static_cast<decltype(second_completion)&&>(second_completion).forward_arguments(),
              static_cast<decltype(third_completion)&&>(third_completion).forward_arguments());
            CHECK(std::get<0>(args) == 13);
            CHECK(&std::get<2>(args) == &referenced);
          }
          else
          {
            FAIL("Unexpected completion");
          }
        }},
      std::move(first),
      std::move(second),
      std::move(third));
    CHECK(invoked);
  }

  TEST_CASE("Stored completion visitation computes noexcept", "[storage_for_completion_signatures]")
  {
    using storage = ::STDEXEC::storage_for_completion_signatures<
      ::STDEXEC::completion_signatures<::STDEXEC::set_value_t(int)>>;

    storage s;
    static_assert(noexcept(
      ::STDEXEC::visit_stored_completion(overloaded{[]() noexcept {}, [](auto&&) noexcept {}}, s)));
    static_assert(
      !noexcept(::STDEXEC::visit_stored_completion(overloaded{[] {}, [](auto&&) noexcept {}}, s)));
    static_assert(!noexcept(
      ::STDEXEC::visit_stored_completion(overloaded{[]() noexcept {}, [](auto&&) {}}, s)));

    using empty_storage =
      ::STDEXEC::storage_for_completion_signatures<::STDEXEC::completion_signatures<>>;
    empty_storage empty;
    static_assert(noexcept(::STDEXEC::visit_stored_completion([]() noexcept {}, empty)));
  }

  TEST_CASE("Stored completion visitation collapses empty storage to nullary invocation",
            "[storage_for_completion_signatures]")
  {
    using storage = ::STDEXEC::storage_for_completion_signatures<
      ::STDEXEC::completion_signatures<::STDEXEC::set_value_t(int)>>;

    storage first;
    storage second;
    second.arrive(::STDEXEC::set_value, 13);

    bool visited_empty = false;
    ::STDEXEC::visit_stored_completion(overloaded{[&] { visited_empty = true; },
                                                  [](auto&&) { FAIL("Unexpected completion"); }},
                                       first);
    CHECK(visited_empty);

    bool visited_empty_from_mixed = false;
    ::STDEXEC::visit_stored_completion(overloaded{[&] { visited_empty_from_mixed = true; },
                                                  [](auto&&...) { FAIL("Unexpected completion"); }},
                                       second,
                                       first);
    CHECK(visited_empty_from_mixed);
  }

  TEST_CASE("Stored completion visitation returns visitor results",
            "[storage_for_completion_signatures]")
  {
    using storage = ::STDEXEC::storage_for_completion_signatures<
      ::STDEXEC::completion_signatures<::STDEXEC::set_value_t(int)>>;

    storage empty;
    storage value;
    value.arrive(::STDEXEC::set_value, 42);

    auto const classify = overloaded{
      []() noexcept -> int { return 0; },
      [](auto&& completion) noexcept -> int
      {
        if constexpr (completion_with_tag<decltype(completion), ::STDEXEC::set_value_t>)
        {
          return std::get<0>(static_cast<decltype(completion)&&>(completion).forward_arguments());
        }
        else
        {
          return -1;
        }
      }};

    static_assert(
      std::is_same_v<decltype(::STDEXEC::visit_stored_completion(classify, empty)), int>);
    static_assert(noexcept(::STDEXEC::visit_stored_completion(classify, empty)));
    CHECK(::STDEXEC::visit_stored_completion(classify, empty) == 0);
    CHECK(::STDEXEC::visit_stored_completion(classify, value) == 42);

    static_assert(
      std::is_same_v<decltype(::STDEXEC::visit_stored_completion<long>(classify, value)), long>);
    static_assert(noexcept(::STDEXEC::visit_stored_completion<long>(classify, value)));
    CHECK(::STDEXEC::visit_stored_completion<long>(classify, value) == 42L);
  }

  TEST_CASE("Storing simple completion signatures and then completing a receiver therewith works",
            "[storage_for_completion_signatures]")
  {
    using storage_type = ::STDEXEC::storage_for_completion_signatures<
      ::STDEXEC::completion_signatures<::STDEXEC::set_value_t(),
                                       ::STDEXEC::set_stopped_t(),
                                       ::STDEXEC::set_error_t(std::exception_ptr)>>;
    {
      storage_type storage;
      storage.arrive(::STDEXEC::set_value);
      CHECK(std::move(storage).complete(expect_void_receiver{}));
    }
    {
      std::optional<storage_type> storage(std::in_place);
      storage->arrive(::STDEXEC::set_error, std::make_exception_ptr(std::logic_error("TEST")));
      std::exception_ptr ex;
      struct receiver
      {
        using receiver_concept = ::STDEXEC::receiver_t;
        void set_value() noexcept
        {
          FAIL("Unexpected value invocation");
        }
        void set_stopped() noexcept
        {
          FAIL("Unexpected stopped invocation");
        }
        void set_error(std::exception_ptr&& ex) noexcept
        {
          //  This ensures that the exception_ptr is moved onto the stack
          CHECK(storage_);
          storage_.reset();
          CHECK(!ex_);
          ex_ = std::move(ex);
        }
        std::optional<storage_type>& storage_;
        std::exception_ptr&          ex_;
      };
      CHECK(std::move(*storage).complete(receiver{storage, ex}));
      REQUIRE(ex);
      bool threw = false;
      try
      {
        std::rethrow_exception(std::move(ex));
      }
      catch (std::logic_error const & ex)
      {
        threw = true;
        CHECK(ex.what() == std::string_view("TEST"));
      }
      CHECK(threw);
    }
  }

  TEST_CASE("When moving a stored value onto the stack throws complete reports std::exception_ptr",
            "[storage_for_completion_signatures]")
  {
    struct throws_on_second_move
    {
      explicit throws_on_second_move(int& moves) noexcept
        : moves_(&moves)
      {}
      throws_on_second_move(throws_on_second_move&& other)
      {
        moves_ = other.moves_;
        ++*moves_;
        if (*moves_ == 2)
        {
          throw std::runtime_error("Throwing as requested");
        }
      }
      int* moves_;
    };

    using storage = ::STDEXEC::storage_for_completion_signatures<
      ::STDEXEC::completion_signatures<::STDEXEC::set_value_t(throws_on_second_move)>>;

    struct receiver
    {
      using receiver_concept = ::STDEXEC::receiver_t;
      void set_value(throws_on_second_move&&) noexcept
      {
        FAIL("Unexpected value invocation");
      }
      void set_error(std::exception_ptr&& ex) noexcept
      {
        CHECK(!ex_);
        ex_ = std::move(ex);
      }
      std::exception_ptr& ex_;
    };

    int     moves = 0;
    storage s;
    s.arrive(::STDEXEC::set_value, throws_on_second_move{moves});
    CHECK(moves == 1);

    std::exception_ptr ex;
    CHECK(std::move(s).complete(receiver{ex}));
    REQUIRE(ex);
    bool threw = false;
    try
    {
      std::rethrow_exception(std::move(ex));
    }
    catch (std::runtime_error const & ex)
    {
      threw = true;
      CHECK(ex.what() == std::string_view("Throwing as requested"));
    }
    CHECK(threw);
  }

  TEST_CASE("When storing a completion signature would throw it is simply coalesced to "
            "std::exception_ptr",
            "[storage_for_completion_signatures]")
  {
    struct maybe_throws_on_move
    {
      maybe_throws_on_move() = default;
      maybe_throws_on_move(maybe_throws_on_move&& other)
      {
        if (other.throws)
        {
          throw std::runtime_error("Throwing as requested");
        }
      }
      bool throws{false};
    };
    {
      using storage = ::STDEXEC::storage_for_completion_signatures<
        ::STDEXEC::completion_signatures<::STDEXEC::set_value_t(maybe_throws_on_move)>>;
      static_assert(storage::nothrow_arrive);
      static_assert(set_equivalent<
                    ::STDEXEC::completion_signatures<::STDEXEC::set_value_t(maybe_throws_on_move),
                                                     ::STDEXEC::set_error_t(std::exception_ptr)>,
                    storage::completion_signatures>);
      {
        storage s;
        static_assert(noexcept(s.arrive(::STDEXEC::set_value, maybe_throws_on_move{})));
        maybe_throws_on_move obj;
        obj.throws = true;
        s.arrive(::STDEXEC::set_value, std::move(obj));
        bool invoked = false;
        ::STDEXEC::visit_stored_completion(
          overloaded{
            [] { FAIL("Unexpected completion"); },
            [&](auto&& completion)
            {
              CHECK(!invoked);
              invoked = true;
              if constexpr (completion_with_tag<decltype(completion), ::STDEXEC::set_error_t>)
              {
                auto args = static_cast<decltype(completion)&&>(completion).forward_arguments();
                REQUIRE(std::get<0>(args));
                //  TODO?
              }
              else
              {
                FAIL("Unexpected completion");
              }
            }},
          std::move(s));
        CHECK(invoked);
      }
      {
        storage s;
        s.arrive(::STDEXEC::set_value, maybe_throws_on_move{});
        bool invoked = false;
        ::STDEXEC::visit_stored_completion(
          overloaded{
            [] { FAIL("Unexpected completion"); },
            [&](auto&& completion)
            {
              CHECK(!invoked);
              invoked = true;
              if constexpr (!completion_with_tag<decltype(completion), ::STDEXEC::set_value_t>)
              {
                FAIL("Unexpected completion");
              }
            }},
          std::move(s));
        CHECK(invoked);
      }
    }
    //  Important that the below cases don't add the std::exception_ptr completion
    //  since propagating a reference can't throw
    {
      using signatures =
        ::STDEXEC::completion_signatures<::STDEXEC::set_value_t(maybe_throws_on_move&)>;
      using storage = ::STDEXEC::storage_for_completion_signatures<signatures>;
      static_assert(std::is_same_v<storage::completion_signatures, signatures>);
      maybe_throws_on_move obj;
      storage              s;
      s.arrive(::STDEXEC::set_value, obj);
      bool invoked = false;
      ::STDEXEC::visit_stored_completion(
        overloaded{
          [] { FAIL("Unexpected completion"); },
          [&](auto&& completion)
          {
            CHECK(!invoked);
            invoked = true;
            if constexpr (completion_with_tag<decltype(completion), ::STDEXEC::set_value_t>)
            {
              auto args = static_cast<decltype(completion)&&>(completion).forward_arguments();
              CHECK(&obj == &std::get<0>(args));
            }
            else
            {
              FAIL("Unexpected completion");
            }
          }},
        std::move(s));
      CHECK(invoked);
    }
    {
      using signatures =
        ::STDEXEC::completion_signatures<::STDEXEC::set_value_t(maybe_throws_on_move&&)>;
      using storage = ::STDEXEC::storage_for_completion_signatures<signatures>;
      static_assert(std::is_same_v<storage::completion_signatures, signatures>);
      maybe_throws_on_move obj;
      storage              s;
      s.arrive(::STDEXEC::set_value, std::move(obj));
      bool invoked = false;
      ::STDEXEC::visit_stored_completion(
        overloaded{
          [] { FAIL("Unexpected completion"); },
          [&](auto&& completion)
          {
            CHECK(!invoked);
            invoked = true;
            if constexpr (completion_with_tag<decltype(completion), ::STDEXEC::set_value_t>)
            {
              auto args = static_cast<decltype(completion)&&>(completion).forward_arguments();
              CHECK(&obj == &std::get<0>(args));
            }
            else
            {
              FAIL("Unexpected completion");
            }
          }},
        std::move(s));
      CHECK(invoked);
    }
  }

  TEST_CASE("Storage can propagate persistence exceptions instead of coalescing them",
            "[storage_for_completion_signatures]")
  {
    struct maybe_throws_on_move
    {
      maybe_throws_on_move() = default;
      maybe_throws_on_move(maybe_throws_on_move&& other)
      {
        if (other.throws)
        {
          throw std::runtime_error("Throwing as requested");
        }
        other.moved = true;
      }
      bool throws{false};
      bool moved{false};
    };

    using signatures =
      ::STDEXEC::completion_signatures<::STDEXEC::set_value_t(maybe_throws_on_move)>;
    using storage = ::STDEXEC::storage_for_completion_signatures<
      signatures,
      ::STDEXEC::storage_for_completion_signatures_error_policy::propagate>;

    static_assert(!storage::nothrow_arrive);
    static_assert(std::is_same_v<storage::completion_signatures, signatures>);
    static_assert(!noexcept(
      std::declval<storage&>().arrive(::STDEXEC::set_value, std::declval<maybe_throws_on_move>())));

    storage              s;
    maybe_throws_on_move obj;
    obj.throws = true;
    CHECK_THROWS_AS(s.arrive(::STDEXEC::set_value, std::move(obj)), std::runtime_error);
    CHECK(!obj.moved);
    bool visited_empty = false;
    ::STDEXEC::visit_stored_completion(overloaded{[&] { visited_empty = true; },
                                                  [](auto&&) { FAIL("Unexpected completion"); }},
                                       std::move(s));
    CHECK(visited_empty);
  }

  TEST_CASE("Propagating storage remains noexcept when persistence is noexcept",
            "[storage_for_completion_signatures]")
  {
    using signatures = ::STDEXEC::completion_signatures<::STDEXEC::set_value_t(int)>;
    using storage    = ::STDEXEC::storage_for_completion_signatures<
         signatures,
         ::STDEXEC::storage_for_completion_signatures_error_policy::propagate>;

    static_assert(storage::nothrow_arrive);
    static_assert(std::is_same_v<storage::completion_signatures, signatures>);
    storage s;
    static_assert(noexcept(s.arrive(::STDEXEC::set_value, 42)));
    s.arrive(::STDEXEC::set_value, 42);

    bool invoked = false;
    ::STDEXEC::visit_stored_completion(
      overloaded{[]() noexcept { FAIL("Unexpected completion"); },
                 [&](auto&& completion) noexcept
                 {
                   CHECK(!invoked);
                   invoked = true;
                   if constexpr (completion_with_tag<decltype(completion), ::STDEXEC::set_value_t>)
                   {
                     auto args =
                       static_cast<decltype(completion)&&>(completion).forward_arguments();
                     CHECK(std::get<0>(args) == 42);
                   }
                   else
                   {
                     FAIL("Unexpected completion");
                   }
                 }},
      std::move(s));
    CHECK(invoked);
  }

  TEST_CASE("Storage deduplicates exact duplicate completion signatures",
            "[storage_for_completion_signatures]")
  {
    using duplicate_signatures =
      ::STDEXEC::completion_signatures<::STDEXEC::set_value_t(int), ::STDEXEC::set_value_t(int)>;
    using deduplicated_signatures = ::STDEXEC::completion_signatures<::STDEXEC::set_value_t(int)>;

    using internalizing_storage =
      ::STDEXEC::storage_for_completion_signatures<duplicate_signatures>;
    using propagating_storage = ::STDEXEC::storage_for_completion_signatures<
      duplicate_signatures,
      ::STDEXEC::storage_for_completion_signatures_error_policy::propagate>;

    static_assert(
      std::is_same_v<internalizing_storage::completion_signatures, deduplicated_signatures>);
    static_assert(
      std::is_same_v<propagating_storage::completion_signatures, deduplicated_signatures>);

    propagating_storage storage;
    storage.arrive(::STDEXEC::set_value, 42);

    bool invoked = false;
    ::STDEXEC::visit_stored_completion(
      overloaded{[] { FAIL("Unexpected empty storage"); },
                 [&](auto&& completion)
                 {
                   CHECK(!invoked);
                   invoked = true;
                   if constexpr (completion_with_tag<decltype(completion), ::STDEXEC::set_value_t>)
                   {
                     CHECK(std::get<0>(
                             static_cast<decltype(completion)&&>(completion).forward_arguments())
                           == 42);
                   }
                   else
                   {
                     FAIL("Unexpected completion");
                   }
                 }},
      std::move(storage));
    CHECK(invoked);
  }

  TEST_CASE("Internalized storage errors are not external arrivals",
            "[storage_for_completion_signatures]")
  {
    struct maybe_throws_on_move
    {
      maybe_throws_on_move() = default;
      maybe_throws_on_move(maybe_throws_on_move&&) {}
    };

    using storage = ::STDEXEC::storage_for_completion_signatures<
      ::STDEXEC::completion_signatures<::STDEXEC::set_value_t(maybe_throws_on_move)>>;

    static_assert(
      set_equivalent<storage::completion_signatures,
                     ::STDEXEC::completion_signatures<::STDEXEC::set_value_t(maybe_throws_on_move),
                                                      ::STDEXEC::set_error_t(std::exception_ptr)>>);
    static_assert(!can_arrive_set_error_with<storage, std::exception_ptr>);
  }

  TEST_CASE("Storage computes completion signatures through a consteval function",
            "[storage_for_completion_signatures]")
  {
    {
      using storage = ::STDEXEC::storage_for_completion_signatures<
        ::STDEXEC::completion_signatures<::STDEXEC::set_value_t(int), ::STDEXEC::set_value_t(int)>>;

      static_assert(std::is_same_v<decltype(storage::get_completion_signatures()),
                                   storage::completion_signatures>);
    }
    {
      struct maybe_throws_on_move
      {
        maybe_throws_on_move() = default;
        maybe_throws_on_move(maybe_throws_on_move&&) {}
      };

      using storage = ::STDEXEC::storage_for_completion_signatures<
        ::STDEXEC::completion_signatures<::STDEXEC::set_value_t(maybe_throws_on_move)>>;

      static_assert(set_equivalent<decltype(storage::get_completion_signatures()),
                                   storage::completion_signatures>);
    }
  }

  TEST_CASE("Storage rejects ambiguous or non-persistable completion signatures",
            "[storage_for_completion_signatures]")
  {
#if STDEXEC_NO_STDCPP_CONSTEXPR_EXCEPTIONS()
    {
      using storage = ::STDEXEC::storage_for_completion_signatures<
        ::STDEXEC::completion_signatures<::STDEXEC::set_value_t(int),
                                         ::STDEXEC::set_value_t(int&&)>>;

      static_assert(::STDEXEC::__merror<decltype(storage::get_completion_signatures())>);
    }
    {
      struct not_move_constructible
      {
        not_move_constructible()                         = default;
        not_move_constructible(not_move_constructible&&) = delete;
      };

      using storage = ::STDEXEC::storage_for_completion_signatures<
        ::STDEXEC::completion_signatures<::STDEXEC::set_value_t(not_move_constructible)>>;

      static_assert(::STDEXEC::__merror<decltype(storage::get_completion_signatures())>);
    }
#endif
  }

}  // unnamed namespace
