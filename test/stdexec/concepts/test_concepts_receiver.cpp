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
#include <catch2/catch_all.hpp>

#include <stdexec/execution.hpp>

#include <test_common/receivers.hpp>

#include <concepts>
#include <cstddef>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace ex = STDEXEC;

struct recv_no_set_value
{
  using receiver_concept = STDEXEC::receiver_tag;

  void set_stopped() noexcept {}
  void set_error(std::exception_ptr) noexcept {}

  [[nodiscard]]
  ex::env<> get_env() const noexcept
  {
    return {};
  }
};

struct recv_set_value_except
{
  using receiver_concept = STDEXEC::receiver_tag;

  void set_value() {}
  void set_stopped() noexcept {}
  void set_error(std::exception_ptr) noexcept {}

  [[nodiscard]]
  ex::env<> get_env() const noexcept
  {
    return {};
  }
};

struct recv_set_value_noexcept
{
  using receiver_concept = STDEXEC::receiver_tag;

  void set_value() noexcept {}
  void set_stopped() noexcept {}
  void set_error(std::exception_ptr) noexcept {}

  [[nodiscard]]
  ex::env<> get_env() const noexcept
  {
    return {};
  }
};

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
struct recv_set_error_except
{
  using receiver_concept = STDEXEC::receiver_tag;

  void set_value() noexcept {}
  void set_stopped() noexcept {}
  void set_error(std::exception_ptr)
  {
    STDEXEC_THROW(std::logic_error{"err"});
  }

  [[nodiscard]]
  ex::env<> get_env() const noexcept
  {
    return {};
  }
};

struct recv_set_stopped_except
{
  using receiver_concept = STDEXEC::receiver_tag;

  void set_value() noexcept {}
  void set_stopped()
  {
    STDEXEC_THROW(std::logic_error{"err"});
  }
  void set_error(std::exception_ptr) noexcept {}

  [[nodiscard]]
  ex::env<> get_env() const noexcept
  {
    return {};
  }
};
#endif

struct recv_non_movable
{
  using receiver_concept = STDEXEC::receiver_tag;

  recv_non_movable()                                    = default;
  ~recv_non_movable()                                   = default;
  recv_non_movable(recv_non_movable&&)                  = delete;
  recv_non_movable& operator=(recv_non_movable&&)       = delete;
  recv_non_movable(recv_non_movable const &)            = default;
  recv_non_movable& operator=(recv_non_movable const &) = default;

  void set_value() noexcept {}
  void set_stopped() noexcept {}
  void set_error(std::exception_ptr) noexcept {}

  [[nodiscard]]
  ex::env<> get_env() const noexcept
  {
    return {};
  }
};

template <class _ParentOp, class _Env>
struct test_basic_receiver
{
  using receiver_concept = ex::receiver_tag;

  template <class _ChildOp>
  static test_basic_receiver make_receiver_for(_ChildOp *__child) noexcept
  {
    // `_ChildOp` is the first non-static data member of `_ParentOp`.
    auto const address = reinterpret_cast<void *>(__child);
    // NOTE: Actually, it's still UB. Under the current standard, pointer conversion/arithmetic cannot achieve
    // this. A truly well-defined approach will likely have to wait until proposals like P3407 are adopted into the
    // standard.
    return test_basic_receiver{reinterpret_cast<_ParentOp *>(address)};
  }

  template <class... _Args>
  void set_value(_Args &&...__args) noexcept
  {
    ex::set_value(std::move(__parent->get_receiver()), static_cast<_Args &&>(__args)...);
  }

  template <class _Error>
  void set_error(_Error &&__error) noexcept
  {
    ex::set_error(std::move(__parent->get_receiver()), static_cast<_Error &&>(__error));
  }

  void set_stopped() noexcept
  {
    ex::set_stopped(std::move(__parent->get_receiver()));
  }

  _Env get_env() const noexcept
  {
    return ex::get_env(__parent->get_receiver());
  }

  _ParentOp *__parent;
};

template <class _Sender, class _Receiver>
struct test_basic_state
{
  using operation_state_concept = ex::operation_state_tag;

  using receiver_t = test_basic_receiver<test_basic_state, ex::env_of_t<_Receiver>>;
  using child_op_t = ex::connect_result_t<_Sender, receiver_t>;

  test_basic_state(_Sender __sender, _Receiver __receiver)
    : receiver(std::move(__receiver))
  {
    // The child operation state should be created after `receiver` constructed
    ::new (static_cast<void *>(storage))
      child_op_t(ex::connect(std::move(__sender), receiver_t{this}));
  }

  void start() & noexcept
  {
    ex::start(*::std::launder(reinterpret_cast<child_op_t *>(storage)));
  }

  _Receiver &get_receiver() noexcept
  {
    return receiver;
  }

  alignas(child_op_t) std::byte storage[sizeof(child_op_t)];
  _Receiver receiver;
};

template <class _Sender, class _Receiver>
  requires ex::inlinable_receiver<_Receiver, test_basic_state<_Sender, _Receiver>>
struct test_basic_state<_Sender, _Receiver>
{
  using operation_state_concept = ex::operation_state_tag;

  using receiver_t = test_basic_receiver<test_basic_state, ex::env_of_t<_Receiver>>;
  using child_op_t = ex::connect_result_t<_Sender, receiver_t>;

  test_basic_state(_Sender __sender, _Receiver const &)
    : child_op(ex::connect(std::move(__sender), receiver_t{this}))
  {}

  void start() & noexcept
  {
    ex::start(child_op);
  }

  _Receiver get_receiver() noexcept
  {
    return _Receiver::make_receiver_for(this);
  }

  child_op_t child_op;
};

template <class _Tag, class _Child>
struct test_basic_sender
{
  using sender_concept = ex::sender_tag;

  _Tag   tag;
  _Child child;

  template <class, class... _Env>
    requires ex::sender_in<_Child, _Env...>
  static consteval auto get_completion_signatures()
  {
    return ex::get_completion_signatures<_Child, _Env...>();
  }

  template <class _Receiver>
  auto connect(_Receiver __receiver)
  {
    return test_basic_state<_Child, _Receiver>{std::move(child), std::move(__receiver)};
  }
};

struct test_adaptor : ex::sender_adaptor_closure<test_adaptor>
{
  template <class _Sender>
  auto operator()(_Sender __sender)
  {
    return test_basic_sender<test_adaptor, _Sender>{*this, std::move(__sender)};
  }
};

struct inlinable_state
{};

struct non_inlinable_state
{};

struct normal_receiver
{
  using receiver_concept = ex::receiver_tag;

  static void set_value(auto const &...) noexcept {}
  static void set_error(auto const &) noexcept {}
  static void set_stopped() noexcept {}
};

struct inlinable_receiver
{
  using receiver_concept = ex::receiver_tag;

  static inlinable_receiver make_receiver_for(inlinable_state *) noexcept
  {
    return {};
  }

  static normal_receiver make_receiver_for(non_inlinable_state *) noexcept
  {
    return {};
  }

  template <class _Child, class _Receiver>
  static inlinable_receiver make_receiver_for(test_basic_state<_Child, _Receiver> *) noexcept
  {
    return {};
  }

  static void set_value(auto const &...) noexcept {}
  static void set_error(auto const &) noexcept {}
  static void set_stopped() noexcept {}
};

static_assert(ex::receiver<inlinable_receiver>);

TEST_CASE("receiver types satisfy the receiver concept", "[concepts][receiver]")
{
  using namespace empty_recv;

  REQUIRE(ex::receiver<recv0>);
  REQUIRE(ex::receiver<recv_int>);
  REQUIRE(ex::receiver<recv0_ec>);
  REQUIRE(ex::receiver<recv_int_ec>);
  REQUIRE(ex::receiver<expect_void_receiver<>>);
  REQUIRE(ex::receiver<expect_void_receiver_ex>);
  REQUIRE(ex::receiver<expect_value_receiver<ex::env<>, int>>);
  REQUIRE(ex::receiver<expect_value_receiver<ex::env<>, double>>);
  REQUIRE(ex::receiver<expect_stopped_receiver<>>);
  REQUIRE(ex::receiver<expect_stopped_receiver_ex<>>);
  REQUIRE(ex::receiver<expect_error_receiver<>>);
  REQUIRE(ex::receiver<expect_error_receiver_ex<std::error_code>>);
  REQUIRE(ex::receiver<logging_receiver>);
}

TEST_CASE("receiver types satisfy the receiver_of concept", "[concepts][receiver]")
{
  using namespace empty_recv;

  REQUIRE(ex::receiver_of<recv_int, ex::completion_signatures<ex::set_value_t(int)>>);
  REQUIRE(ex::receiver_of<recv0_ec, ex::completion_signatures<ex::set_error_t(std::error_code)>>);
  REQUIRE(
    ex::receiver_of<recv_int_ec, ex::completion_signatures<ex::set_error_t(std::error_code)>>);
  REQUIRE(ex::receiver_of<recv_int_ec, ex::completion_signatures<ex::set_value_t(int)>>);
  REQUIRE(ex::receiver_of<expect_value_receiver<ex::env<>, int>,
                          ex::completion_signatures<ex::set_value_t(int)>>);
  REQUIRE(ex::receiver_of<expect_value_receiver<ex::env<>, double>,
                          ex::completion_signatures<ex::set_value_t(double)>>);
  REQUIRE(
    ex::receiver_of<expect_stopped_receiver<>, ex::completion_signatures<ex::set_value_t(char)>>);
  REQUIRE(ex::receiver_of<expect_stopped_receiver_ex<>,
                          ex::completion_signatures<ex::set_value_t(char)>>);
  REQUIRE(
    ex::receiver_of<expect_error_receiver<>, ex::completion_signatures<ex::set_value_t(char)>>);
  REQUIRE(ex::receiver_of<expect_error_receiver_ex<std::error_code>,
                          ex::completion_signatures<ex::set_value_t(char)>>);
  REQUIRE(ex::receiver_of<logging_receiver, ex::completion_signatures<ex::set_value_t(char)>>);
}

TEST_CASE("receiver type w/o set_value models receiver but not receiver_of", "[concepts][receiver]")
{
  REQUIRE(ex::receiver<recv_no_set_value>);
  REQUIRE(!ex::receiver_of<recv_no_set_value, ex::completion_signatures<ex::set_value_t()>>);
}

TEST_CASE("type with set_value noexcept is a receiver", "[concepts][receiver]")
{
  REQUIRE(ex::receiver<recv_set_value_noexcept>);
  REQUIRE(ex::receiver_of<recv_set_value_noexcept, ex::completion_signatures<ex::set_value_t()>>);
}

TEST_CASE("non-movable type is not a receiver", "[concepts][receiver]")
{
  REQUIRE(!ex::receiver<recv_non_movable>);
  REQUIRE(!ex::receiver_of<recv_non_movable, ex::completion_signatures<ex::set_value_t()>>);
}

TEST_CASE("inlinable_receiver", "[concepts][receiver]")
{
  static_assert(!ex::inlinable_receiver<inlinable_receiver, int>);
  static_assert(!ex::inlinable_receiver<inlinable_receiver, non_inlinable_state>);
  static_assert(ex::inlinable_receiver<inlinable_receiver, inlinable_state>);

  auto sender             = ex::just(42) | test_adaptor{};
  using normal_state      = ex::connect_result_t<decltype(sender), normal_receiver>;
  using inlinable_state_t = ex::connect_result_t<decltype(sender), inlinable_receiver>;
  static_assert(ex::operation_state<normal_state>);
  static_assert(ex::operation_state<inlinable_state_t>);

  auto nested_sender           = sender | test_adaptor{};
  using nested_normal_state    = ex::connect_result_t<decltype(nested_sender), normal_receiver>;
  using nested_inlinable_state = ex::connect_result_t<decltype(nested_sender), inlinable_receiver>;
  static_assert(sizeof(normal_state) == sizeof(nested_normal_state));
  static_assert(sizeof(inlinable_state_t) == sizeof(nested_inlinable_state));

  auto [result] = ex::sync_wait(nested_sender).value();
  REQUIRE(result == 42);

  auto nested_4_sender = ex::just(67) | test_adaptor{} | test_adaptor{} | test_adaptor{}
                       | test_adaptor{};
  using nested_4_normal_state = ex::connect_result_t<decltype(nested_4_sender), normal_receiver>;
  using nested_4_inlinable_state =
    ex::connect_result_t<decltype(nested_4_sender), inlinable_receiver>;
  static_assert(sizeof(normal_state) == sizeof(nested_4_normal_state));
  static_assert(sizeof(inlinable_state_t) == sizeof(nested_4_inlinable_state));

  auto [result1] = ex::sync_wait(nested_4_sender).value();
  REQUIRE(result1 == 67);
}
