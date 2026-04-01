/*
 * Copyright (c) 2026 Ian Petersen
 * Copyright (c) 2026 NVIDIA Corporation
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

#include "test_common/receivers.hpp"
#include "test_common/type_helpers.hpp"
#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>

#include <type_traits>

namespace ex = STDEXEC;

namespace
{
  template <class T>
  struct ready_awaitable
  {
   public:
    explicit constexpr ready_awaitable(T&& t) noexcept
      : t_(std::move(t))
    {}

    explicit constexpr ready_awaitable(T const & t)
      : t_(t)
    {}

    static constexpr bool await_ready() noexcept
    {
      return true;
    }

    static void await_suspend(std::coroutine_handle<>) noexcept
    {
      FAIL_CHECK("this awaitable should never suspend");
    }

    constexpr T await_resume() noexcept
    {
      return std::move(t_);
    }

    ready_awaitable& base() noexcept
    {
      return *this;
    }

   private:
    T t_;
  };

  template <>
  struct ready_awaitable<void>
  {
    static constexpr bool await_ready() noexcept
    {
      return true;
    }

    static void await_suspend(std::coroutine_handle<>) noexcept
    {
      FAIL_CHECK("this awaitable should never suspend");
    }

    static constexpr void await_resume() noexcept {}

    ready_awaitable& base() noexcept
    {
      return *this;
    }
  };

  template <class Awaitable>
  struct awaitable_ref
  {
    Awaitable* awaitable_;

    explicit constexpr awaitable_ref(Awaitable& awaitable) noexcept
      : awaitable_(&awaitable)
    {}

    constexpr auto await_ready() const noexcept(noexcept(awaitable_->await_ready()))
      requires requires(Awaitable& a) {
        { a.await_ready() };
      }
    {
      return awaitable_->await_ready();
    }

    template <class Promise>
    constexpr auto await_suspend(std::coroutine_handle<Promise> coro) const
      noexcept(noexcept(awaitable_->await_suspend(coro)))
      requires requires(Awaitable& a) {
        { a.await_suspend(coro) };
      }
    {
      return awaitable_->await_suspend(coro);
    }

    constexpr decltype(auto) await_resume() const noexcept(noexcept(awaitable_->await_resume()))
      requires requires(Awaitable& a) {
        { a.await_resume() };
      }
    {
      return awaitable_->await_resume();
    }

    template <class Promise>
      requires requires(Awaitable& a, Promise& p) {
        { a.as_awaitable(p) } -> ex::__awaitable<Promise>;
      }
    constexpr auto as_awaitable(Promise& p) const noexcept(noexcept(awaitable_->as_awaitable(p)))
    {
      return awaitable_->as_awaitable(p);
    }

    constexpr auto operator co_await() const noexcept(noexcept(awaitable_->operator co_await()))
      requires requires(Awaitable& a) {
        { a.operator co_await() };
      }
    {
      return awaitable_->operator co_await();
    }
  };

  template <class Awaitable>
  awaitable_ref(Awaitable&) -> awaitable_ref<Awaitable>;

  template <class Awaitable>
    requires requires(Awaitable& a) {
      { operator co_await(a) };
    }
  constexpr auto operator co_await(awaitable_ref<Awaitable> ref)
    noexcept(noexcept(operator co_await(*ref.awaitable_)))
  {
    return operator co_await(*ref.awaitable_);
  }

  template <class T>
  struct suspending_awaitable : ready_awaitable<T>
  {
    using ready_awaitable<T>::ready_awaitable;

    suspending_awaitable(suspending_awaitable&&) = delete;

    ~suspending_awaitable() = default;

    constexpr void resume_parent() const noexcept
    {
      parent_.resume();
    }

    static constexpr bool await_ready() noexcept
    {
      return false;
    }

    constexpr void await_suspend(std::coroutine_handle<> coro) noexcept
    {
      parent_ = coro;
    }

    suspending_awaitable& base() noexcept
    {
      return *this;
    }

   private:
    std::coroutine_handle<> parent_;
  };

  template <class T>
  struct conditionally_suspending_awaitable : suspending_awaitable<T>
  {
    template <class... U>
      requires(sizeof...(U) == 0 && std::same_as<T, void>)
             || (sizeof...(U) == 1 && !std::same_as<T, void>)
    explicit constexpr conditionally_suspending_awaitable(bool suspend, U&&... u) noexcept
      : suspending_awaitable<T>(std::forward<U>(u)...)
      , suspend_(suspend)
    {}

    constexpr bool await_suspend(std::coroutine_handle<> coro) noexcept
    {
      if (suspend_)
      {
        suspending_awaitable<T>::await_suspend(coro);
      }

      return suspend_;
    }

    conditionally_suspending_awaitable& base() noexcept
    {
      return *this;
    }

   private:
    bool suspend_;
  };

  template <class T>
  struct symmetrically_suspending_awaitable : conditionally_suspending_awaitable<T>
  {
    using conditionally_suspending_awaitable<T>::conditionally_suspending_awaitable;

    constexpr std::coroutine_handle<> await_suspend(std::coroutine_handle<> coro) noexcept
    {
      if (conditionally_suspending_awaitable<T>::await_suspend(coro))
      {
        return std::noop_coroutine();
      }
      else
      {
        return coro;
      }
    }

    symmetrically_suspending_awaitable& base() noexcept
    {
      return *this;
    }
  };

  template <class Awaitable>
  struct with_as_awaitable
  {
    template <class... T>
      requires std::constructible_from<Awaitable, T...>
    explicit(sizeof...(T) == 1) with_as_awaitable(T&&... t)
      noexcept(std::is_nothrow_constructible_v<Awaitable, T...>)
      : awaitable_(std::forward<T>(t)...)
    {}

    template <class Promise>
    awaitable_ref<Awaitable> as_awaitable(Promise&) noexcept
    {
      return awaitable_ref(awaitable_);
    }

    auto& base() noexcept
    {
      return awaitable_.base();
    }

   private:
    Awaitable awaitable_;
  };

  template <class Awaitable>
  struct with_member_co_await
  {
    template <class... T>
      requires std::constructible_from<Awaitable, T...>
    explicit(sizeof...(T) == 1) with_member_co_await(T&&... t)
      noexcept(std::is_nothrow_constructible_v<Awaitable, T...>)
      : awaitable_(std::forward<T>(t)...)
    {}

    constexpr awaitable_ref<Awaitable> operator co_await() noexcept
    {
      return awaitable_ref(awaitable_);
    }

    auto& base() noexcept
    {
      return awaitable_.base();
    }

   private:
    Awaitable awaitable_;
  };

  template <class Awaitable>
  struct with_friend_co_await
  {
    template <class... T>
      requires std::constructible_from<Awaitable, T...>
    explicit(sizeof...(T) == 1) with_friend_co_await(T&&... t)
      noexcept(std::is_nothrow_constructible_v<Awaitable, T...>)
      : awaitable_(std::forward<T>(t)...)
    {}

    auto& base() noexcept
    {
      return awaitable_.base();
    }

   private:
    Awaitable awaitable_;

    template <class Self>
      requires std::same_as<std::remove_cvref_t<Self>, with_friend_co_await>
    friend constexpr awaitable_ref<Awaitable> operator co_await(Self&& self) noexcept
    {
      return awaitable_ref(self.awaitable_);
    }
  };

  TEST_CASE("can connect and start a ready_awaitable<int>", "[cpo][cpo_connect_awaitable]")
  {
    auto test = [](auto awaitable) noexcept
    {
      auto op = ex::connect(std::move(awaitable), expect_value_receiver{42});
      op.start();
    };

    test(ready_awaitable{42});
    test(with_as_awaitable<ready_awaitable<int>>{42});
    test(with_member_co_await<ready_awaitable<int>>{42});
    test(with_friend_co_await<ready_awaitable<int>>{42});
    test(with_as_awaitable<with_member_co_await<ready_awaitable<int>>>{42});
    test(with_as_awaitable<with_friend_co_await<ready_awaitable<int>>>{42});
  }

  TEST_CASE("can connect and start a ready_awaitable<void>", "[cpo][cpo_connect_awaitable]")
  {
    auto test = [](auto awaitable) noexcept
    {
      auto op = ex::connect(std::move(awaitable), expect_void_receiver{});
      op.start();
    };

    test(ready_awaitable<void>{});
    test(with_as_awaitable<ready_awaitable<void>>{});
    test(with_member_co_await<ready_awaitable<void>>{});
    test(with_friend_co_await<ready_awaitable<void>>{});
    test(with_as_awaitable<with_member_co_await<ready_awaitable<void>>>{});
    test(with_as_awaitable<with_friend_co_await<ready_awaitable<void>>>{});
  }

  TEST_CASE("can connect and start a suspending_awaitable", "[cpo][cpo_connect_awaitable]")
  {
    auto test = [](auto awaitable, auto... values) noexcept
    {
      auto op = ex::connect(awaitable_ref(awaitable), expect_value_receiver{std::move(values)...});
      op.start();
      awaitable.base().resume_parent();
    };

    test(suspending_awaitable<int>{42}, 42);
    test(with_as_awaitable<suspending_awaitable<int>>{42}, 42);
    test(with_member_co_await<suspending_awaitable<int>>{42}, 42);
    test(with_friend_co_await<suspending_awaitable<int>>{42}, 42);
    test(with_as_awaitable<with_member_co_await<suspending_awaitable<int>>>{42}, 42);
    test(with_as_awaitable<with_friend_co_await<suspending_awaitable<int>>>{42}, 42);

    test(suspending_awaitable<void>{});
    test(with_as_awaitable<suspending_awaitable<void>>{});
    test(with_member_co_await<suspending_awaitable<void>>{});
    test(with_friend_co_await<suspending_awaitable<void>>{});
    test(with_as_awaitable<with_member_co_await<suspending_awaitable<void>>>{});
    test(with_as_awaitable<with_friend_co_await<suspending_awaitable<void>>>{});
  }

  TEST_CASE("can connect and start a conditionally_suspending_awaitable",
            "[cpo][cpo_connect_awaitable]")
  {
    {
      auto test = [](auto awaitable, auto... values) noexcept
      {
        auto op = ex::connect(awaitable_ref(awaitable),
                              expect_value_receiver{std::move(values)...});
        op.start();
        awaitable.base().resume_parent();
      };

      test(conditionally_suspending_awaitable<int>(true, 42), 42);
      test(with_as_awaitable<conditionally_suspending_awaitable<int>>(true, 42), 42);
      test(with_member_co_await<conditionally_suspending_awaitable<int>>(true, 42), 42);
      test(with_friend_co_await<conditionally_suspending_awaitable<int>>(true, 42), 42);
      test(with_as_awaitable<with_member_co_await<conditionally_suspending_awaitable<int>>>(true,
                                                                                            42),
           42);
      test(with_as_awaitable<with_friend_co_await<conditionally_suspending_awaitable<int>>>(true,
                                                                                            42),
           42);

      test(conditionally_suspending_awaitable<void>(true));
      test(with_as_awaitable<conditionally_suspending_awaitable<void>>(true));
      test(with_member_co_await<conditionally_suspending_awaitable<void>>(true));
      test(with_friend_co_await<conditionally_suspending_awaitable<void>>(true));
      test(with_as_awaitable<with_member_co_await<conditionally_suspending_awaitable<void>>>(true));
      test(with_as_awaitable<with_friend_co_await<conditionally_suspending_awaitable<void>>>(true));
    }

    {
      auto test = [](auto awaitable, auto... values) noexcept
      {
        auto op = ex::connect(awaitable_ref(awaitable),
                              expect_value_receiver{std::move(values)...});
        op.start();
      };

      test(conditionally_suspending_awaitable<int>(false, 42), 42);
      test(with_as_awaitable<conditionally_suspending_awaitable<int>>(false, 42), 42);
      test(with_member_co_await<conditionally_suspending_awaitable<int>>(false, 42), 42);
      test(with_friend_co_await<conditionally_suspending_awaitable<int>>(false, 42), 42);
      test(with_as_awaitable<with_member_co_await<conditionally_suspending_awaitable<int>>>(false,
                                                                                            42),
           42);
      test(with_as_awaitable<with_friend_co_await<conditionally_suspending_awaitable<int>>>(false,
                                                                                            42),
           42);

      test(conditionally_suspending_awaitable<void>(false));
      test(with_as_awaitable<conditionally_suspending_awaitable<void>>(false));
      test(with_member_co_await<conditionally_suspending_awaitable<void>>(false));
      test(with_friend_co_await<conditionally_suspending_awaitable<void>>(false));
      test(
        with_as_awaitable<with_member_co_await<conditionally_suspending_awaitable<void>>>(false));
      test(
        with_as_awaitable<with_friend_co_await<conditionally_suspending_awaitable<void>>>(false));
    }
  }

  TEST_CASE("can connect and start a symmetrically_suspending_awaitable",
            "[cpo][cpo_connect_awaitable]")
  {
    {
      auto test = [](auto awaitable, auto... values) noexcept
      {
        auto op = ex::connect(awaitable_ref(awaitable),
                              expect_value_receiver{std::move(values)...});
        op.start();
        awaitable.base().resume_parent();
      };

      test(symmetrically_suspending_awaitable<int>(true, 42), 42);
      test(with_as_awaitable<symmetrically_suspending_awaitable<int>>(true, 42), 42);
      test(with_member_co_await<symmetrically_suspending_awaitable<int>>(true, 42), 42);
      test(with_friend_co_await<symmetrically_suspending_awaitable<int>>(true, 42), 42);
      test(with_as_awaitable<with_member_co_await<symmetrically_suspending_awaitable<int>>>(true,
                                                                                            42),
           42);
      test(with_as_awaitable<with_friend_co_await<symmetrically_suspending_awaitable<int>>>(true,
                                                                                            42),
           42);

      test(symmetrically_suspending_awaitable<void>(true));
      test(with_as_awaitable<symmetrically_suspending_awaitable<void>>(true));
      test(with_member_co_await<symmetrically_suspending_awaitable<void>>(true));
      test(with_friend_co_await<symmetrically_suspending_awaitable<void>>(true));
      test(with_as_awaitable<with_member_co_await<symmetrically_suspending_awaitable<void>>>(true));
      test(with_as_awaitable<with_friend_co_await<symmetrically_suspending_awaitable<void>>>(true));
    }

    {
      auto test = [](auto awaitable, auto... values) noexcept
      {
        auto op = ex::connect(awaitable_ref(awaitable),
                              expect_value_receiver{std::move(values)...});
        op.start();
      };

      test(symmetrically_suspending_awaitable<int>(false, 42), 42);
      test(with_as_awaitable<symmetrically_suspending_awaitable<int>>(false, 42), 42);
      test(with_member_co_await<symmetrically_suspending_awaitable<int>>(false, 42), 42);
      test(with_friend_co_await<symmetrically_suspending_awaitable<int>>(false, 42), 42);
      test(with_as_awaitable<with_member_co_await<symmetrically_suspending_awaitable<int>>>(false,
                                                                                            42),
           42);
      test(with_as_awaitable<with_friend_co_await<symmetrically_suspending_awaitable<int>>>(false,
                                                                                            42),
           42);

      test(symmetrically_suspending_awaitable<void>(false));
      test(with_as_awaitable<symmetrically_suspending_awaitable<void>>(false));
      test(with_member_co_await<symmetrically_suspending_awaitable<void>>(false));
      test(with_friend_co_await<symmetrically_suspending_awaitable<void>>(false));
      test(
        with_as_awaitable<with_member_co_await<symmetrically_suspending_awaitable<void>>>(false));
      test(
        with_as_awaitable<with_friend_co_await<symmetrically_suspending_awaitable<void>>>(false));
    }
  }
}  // namespace
