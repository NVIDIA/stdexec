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

#include <exec/function.hpp>

#include <catch2/catch_all.hpp>

#include <stdexec/execution.hpp>

#include <memory>

namespace ex = STDEXEC;

namespace
{
  template <class Channel, class Domain>
  struct domain_sender_t
  {
    template <class... Values>
    class sender
    {
      struct attrs
      {
        template <class... Env>
        constexpr Domain query(ex::get_completion_domain_t<Channel>, Env const &...) const noexcept
        {
          return {};
        }
      };

      template <class Receiver>
      struct opstate
      {
        using operation_state_concept = ex::operation_state_tag;

        void start() & noexcept
        {
          ex::__apply(Channel(), std::move(values), std::move(rcvr));
        }

        Receiver               rcvr;
        ex::__tuple<Values...> values;
      };

      ex::__tuple<Values...> values_;

     public:
      using sender_concept = ex::sender_tag;

      template <class S>
      static consteval auto get_completion_signatures() noexcept  //
        -> ex::completion_signatures<Channel(Values...)>
      {
        return {};
      }

      constexpr attrs get_env() const noexcept
      {
        return {};
      }

      constexpr explicit sender(Values... values) noexcept
        : values_(values...)
      {}

      template <class Receiver>
      opstate<Receiver> connect(Receiver rcvr) && noexcept
      {
        return opstate<Receiver>(std::move(rcvr), std::move(values_));
      }
    };

    template <class... Values>
    constexpr sender<Values...> operator()(Values... values) const noexcept
    {
      return sender<Values...>(std::move(values)...);
    }
  };

  template <auto Channel, class Domain>
  inline constexpr domain_sender_t<std::remove_cvref_t<decltype(Channel)>, Domain> domain_sender{};

  TEST_CASE("exec::function is constructible", "[types][function]")
  {
    SECTION("void()")
    {
      exec::function<void()> sndr([]() noexcept { return ex::just(); });
      STATIC_REQUIRE(STDEXEC::sender<decltype(sndr)>);
    }

    SECTION("int()")
    {
      exec::function<int()> sndr([]() noexcept { return ex::just(42); });
      STATIC_REQUIRE(STDEXEC::sender<decltype(sndr)>);
    }

    SECTION("void(int, double&)")
    {
      double                              d = 4.;
      exec::function<void(int, double &)> sndr(5,
                                               d,
                                               [](int, double &) noexcept { return ex::just(); });
      STATIC_REQUIRE(STDEXEC::sender<decltype(sndr)>);
    }

    SECTION("void() noexcept")
    {
      exec::function<void() noexcept> sndr([]() noexcept { return ex::just(); });
      STATIC_REQUIRE(STDEXEC::sender<decltype(sndr)>);
    }

    SECTION("int() noexcept")
    {
      exec::function<int() noexcept> sndr([]() noexcept { return ex::just(42); });
      STATIC_REQUIRE(STDEXEC::sender<decltype(sndr)>);
    }

    SECTION("sender_tag() with only set_value_t(int)")
    {
      exec::function<ex::sender_tag(), ex::completion_signatures<ex::set_value_t(int)>> sndr(
        []() noexcept { return ex::just(42); });
      STATIC_REQUIRE(STDEXEC::sender<decltype(sndr)>);
    }

    SECTION("sender_tag() with only set_stopped_t()")
    {
      exec::function<ex::sender_tag(), ex::completion_signatures<ex::set_stopped_t()>> sndr(
        []() noexcept { return ex::just_stopped(); });
      STATIC_REQUIRE(STDEXEC::sender<decltype(sndr)>);
    }

    SECTION("void() with trivial custom environment")
    {
      exec::function<void(), exec::queries<>> sndr([]() noexcept { return ex::just(); });
      STATIC_REQUIRE(STDEXEC::sender<decltype(sndr)>);
    }

    SECTION("sender_tag(int) with only set_value_t() and trivial environment")
    {
      exec::function<ex::sender_tag(int),
                     ex::completion_signatures<ex::set_value_t()>,
                     exec::queries<>>
        sndr(5, [](int) noexcept { return ex::just(); });
      STATIC_REQUIRE(STDEXEC::sender<decltype(sndr)>);
    }

    struct domain
    {};

    SECTION("void() with attrs but no queries")
    {
      exec::function<void(), exec::attrs<domain(ex::get_completion_domain_t<ex::set_value_t>)>>
        sndr(domain_sender<ex::set_value, domain>);

      STATIC_REQUIRE(ex::sender<decltype(sndr)>);
    }

    SECTION("void() noexcept with attrs but no queries")
    {
      exec::function<void() noexcept,
                     exec::attrs<domain(ex::get_completion_domain_t<ex::set_value_t>)>>
        sndr(domain_sender<ex::set_value, domain>);

      STATIC_REQUIRE(ex::sender<decltype(sndr)>);
    }

    SECTION("sender_tag(int) with set_value_t(int) and attrs but no queries")
    {
      exec::function<ex::sender_tag(int),
                     ex::completion_signatures<ex::set_value_t(int)>,
                     exec::attrs<domain(ex::get_completion_domain_t<ex::set_value_t>)>>
        sndr(42, domain_sender<ex::set_value, domain>);

      STATIC_REQUIRE(ex::sender<decltype(sndr)>);
    }

    SECTION("sender_tag(int) with set_error_t(int), attrs, and trivial queries")
    {
      exec::function<ex::sender_tag(int),
                     ex::completion_signatures<ex::set_error_t(int)>,
                     exec::queries<>,
                     exec::attrs<domain(ex::get_completion_domain_t<ex::set_error_t>)>>
        sndr(42, domain_sender<ex::set_error, domain>);

      STATIC_REQUIRE(ex::sender<decltype(sndr)>);
    }
  }

  TEST_CASE("exec::function is connectable", "[types][function]")
  {
    SECTION("int() noexcept from just(42)")
    {
      exec::function<int() noexcept> sndr([]() noexcept { return ex::just(42); });

      auto [fortytwo] = ex::sync_wait(std::move(sndr)).value();

      REQUIRE(fortytwo == 42);
    }

    SECTION("void() from throwing factory")
    {
      exec::function<void()> sndr([]() -> decltype(ex::just()) { throw "oops"; });

      REQUIRE_THROWS(ex::sync_wait(std::move(sndr)));
    }

    SECTION("void() from throwing then")
    {
      exec::function<void()> sndr([]() noexcept
                                  { return ex::just() | ex::then([] { throw "oops"; }); });

      REQUIRE_THROWS(ex::sync_wait(std::move(sndr)));
    }

    SECTION("void() from just_stopped()")
    {
      exec::function<void()> sndr([]() noexcept { return ex::just_stopped(); });

      auto ret = ex::sync_wait(std::move(sndr));

      REQUIRE_FALSE(ret.has_value());
    }

    SECTION("custom completions from just_error(42)")
    {
      exec::function<ex::sender_tag(),
                     ex::completion_signatures<ex::set_value_t(), ex::set_error_t(int)>>
        sndr([]() noexcept { return ex::just_error(42); });

      REQUIRE_THROWS_AS(ex::sync_wait(std::move(sndr)), int);
    }
  }

  TEST_CASE("exec::function forwards get_frame_allocator", "[types][function]")
  {
    exec::function<bool() noexcept> sndr(
      []() noexcept
      {
        return ex::read_env(exec::get_frame_allocator)
             | ex::then(
                 [](auto alloc) noexcept
                 {
                   return std::same_as<std::pmr::polymorphic_allocator<std::byte>, decltype(alloc)>;
                 });
      });

    std::pmr::polymorphic_allocator<std::byte> alloc;

    auto [ret] = ex::sync_wait(std::move(sndr)
                               | ex::write_env(ex::prop(exec::get_frame_allocator, alloc)))
                   .value();

    REQUIRE(ret);
  }

  TEST_CASE("exec::function is conditionally lvalue connectable", "[types][function]")
  {
    exec::function<int()> sndr([]() noexcept { return ex::just(42); });

    auto [ret] = ex::sync_wait(sndr).value();

    REQUIRE(ret == 42);
  }

  TEST_CASE("exec::function accepts lvalue callables", "[types][function]")
  {
    exec::function<int(int) noexcept> sndr(42, ex::just);

    auto [ret] = ex::sync_wait(sndr).value();

    REQUIRE(ret == 42);
  }

  struct iface
  {
    virtual exec::function<int() noexcept> get_i_virtually() const noexcept = 0;
  };

  struct iface2
  {
    exec::function<int(iface2 const *) noexcept> get_i_from_base() const noexcept
    {
      return exec::function<int(iface2 const *) noexcept>(this, &iface2::get_i_virtually);
    }

    virtual exec::function<int() noexcept> get_i_virtually() const noexcept = 0;
  };

  struct impl
    : iface
    , iface2
  {
    explicit impl(int i) noexcept
      : i_(i)
    {}

    auto just_i() const noexcept
    {
      return ex::just(i_);
    }

    static auto static_just_i(impl const *self) noexcept
    {
      return self->just_i();
    }

    exec::function<int() noexcept> get_i_with_capture() const noexcept
    {
      return exec::function<int() noexcept>([this]() noexcept { return just_i(); });
    }

    exec::function<int(impl const *) noexcept> get_i_with_pmfn() const noexcept
    {
      return exec::function<int(impl const *) noexcept>(this, &impl::just_i);
    }

    exec::function<int() noexcept> get_i_virtually() const noexcept override
    {
      return get_i_with_capture();
    }

   private:
    int i_;
  };

  TEST_CASE("exec::function accepts small trivially-copyable callables", "[types][function]")
  {
    SECTION("function<int() noexcept> accepts a lambda capturing this")
    {
      auto [ret] = ex::sync_wait(impl{42}.get_i_with_capture()).value();

      REQUIRE(ret == 42);
    }

    SECTION("function<int(impl const *) noexcept> accepts a pointer-to-member function")
    {
      auto [ret] = ex::sync_wait(impl{42}.get_i_with_pmfn()).value();

      REQUIRE(ret == 42);
    }

    SECTION("function<int(impl const *) noexcept> accepts a pointer-to-function")
    {
      impl imp{42};
      auto [ret] = ex::sync_wait(
                     exec::function<int(impl const *) noexcept>(&imp, &impl::static_just_i))
                     .value();

      REQUIRE(ret == 42);
    }

    SECTION("function<int()> can be the return type of a virtual member function")
    {
      auto [ret] = ex::sync_wait(impl{42}.get_i_virtually()).value();

      REQUIRE(ret == 42);
    }

    SECTION("function<int(iface const *) noexcept> accepts a pointer-to-member function")
    {
      impl imp{42};
      auto [ret] =
        ex::sync_wait(exec::function<int(iface const *)>(&imp, &iface::get_i_virtually)).value();

      REQUIRE(ret == 42);
    }

    SECTION("function<int(iface2 const *) noexcept> works on the base class")
    {
      auto [ret] = ex::sync_wait(impl{42}.get_i_from_base()).value();

      REQUIRE(ret == 42);
    }
  }

  TEST_CASE("completion_signature specification is order-independent", "[types][function]")
  {
    // by specifying the completions with a function signature, it's up to the library what
    // order the completion signatures are specified in
    using func1_t = exec::function<int(int) noexcept>;
    // this declaration chooses value before stopped
    using func2_t =
      exec::function<ex::sender_tag(int),
                     ex::completion_signatures<ex::set_value_t(int), ex::set_stopped_t()>>;
    // this declaration chooses stopped before value
    using func3_t =
      exec::function<ex::sender_tag(int),
                     ex::completion_signatures<ex::set_stopped_t(), ex::set_value_t(int)>>;

    SECTION("the function types are the same as each other")
    {
      STATIC_REQUIRE(std::same_as<func1_t, func2_t>);
      STATIC_REQUIRE(std::same_as<func1_t, func3_t>);
      STATIC_REQUIRE(std::same_as<func2_t, func3_t>);
    }

    SECTION("move-construction works in every direction between all three types")
    {
      STATIC_REQUIRE(std::constructible_from<func1_t, func1_t>);
      STATIC_REQUIRE(std::constructible_from<func1_t, func2_t>);
      STATIC_REQUIRE(std::constructible_from<func1_t, func3_t>);
      STATIC_REQUIRE(std::constructible_from<func2_t, func1_t>);
      STATIC_REQUIRE(std::constructible_from<func2_t, func2_t>);
      STATIC_REQUIRE(std::constructible_from<func2_t, func3_t>);
      STATIC_REQUIRE(std::constructible_from<func3_t, func1_t>);
      STATIC_REQUIRE(std::constructible_from<func3_t, func2_t>);
      STATIC_REQUIRE(std::constructible_from<func3_t, func3_t>);
    }

    SECTION("copy-construction works in every direction between all three types")
    {
      STATIC_REQUIRE(std::constructible_from<func1_t, func1_t const &>);
      STATIC_REQUIRE(std::constructible_from<func1_t, func2_t const &>);
      STATIC_REQUIRE(std::constructible_from<func1_t, func3_t const &>);
      STATIC_REQUIRE(std::constructible_from<func2_t, func1_t const &>);
      STATIC_REQUIRE(std::constructible_from<func2_t, func2_t const &>);
      STATIC_REQUIRE(std::constructible_from<func2_t, func3_t const &>);
      STATIC_REQUIRE(std::constructible_from<func3_t, func1_t const &>);
      STATIC_REQUIRE(std::constructible_from<func3_t, func2_t const &>);
      STATIC_REQUIRE(std::constructible_from<func3_t, func3_t const &>);
    }

    SECTION("move-assignment works in every direction between all three types")
    {
      STATIC_REQUIRE(std::assignable_from<func1_t &, func1_t>);
      STATIC_REQUIRE(std::assignable_from<func1_t &, func2_t>);
      STATIC_REQUIRE(std::assignable_from<func1_t &, func3_t>);
      STATIC_REQUIRE(std::assignable_from<func2_t &, func1_t>);
      STATIC_REQUIRE(std::assignable_from<func2_t &, func2_t>);
      STATIC_REQUIRE(std::assignable_from<func2_t &, func3_t>);
      STATIC_REQUIRE(std::assignable_from<func3_t &, func1_t>);
      STATIC_REQUIRE(std::assignable_from<func3_t &, func2_t>);
      STATIC_REQUIRE(std::assignable_from<func3_t &, func3_t>);
    }

    SECTION("copy-assignment works in every direction between all three types")
    {
      STATIC_REQUIRE(std::assignable_from<func1_t &, func1_t const &>);
      STATIC_REQUIRE(std::assignable_from<func1_t &, func2_t const &>);
      STATIC_REQUIRE(std::assignable_from<func1_t &, func3_t const &>);
      STATIC_REQUIRE(std::assignable_from<func2_t &, func1_t const &>);
      STATIC_REQUIRE(std::assignable_from<func2_t &, func2_t const &>);
      STATIC_REQUIRE(std::assignable_from<func2_t &, func3_t const &>);
      STATIC_REQUIRE(std::assignable_from<func3_t &, func1_t const &>);
      STATIC_REQUIRE(std::assignable_from<func3_t &, func2_t const &>);
      STATIC_REQUIRE(std::assignable_from<func3_t &, func3_t const &>);
    }
  }

  TEST_CASE("queries specification is order-independent", "[types][function]")
  {
    constexpr auto query1 = [](auto const &) noexcept
    {
      return 0;
    };

    constexpr auto query2 = [](auto const &, int i)
    {
      return (double) i;
    };

    using query1_t = decltype(query1);
    using query2_t = decltype(query2);

    using func1_t =
      exec::function<int(int), exec::queries<int(query1_t) noexcept, double(query2_t, int)>>;

    using func2_t =
      exec::function<int(int), exec::queries<double(query2_t, int), int(query1_t) noexcept>>;

    SECTION("the function types are the same as each other")
    {
      STATIC_REQUIRE(std::same_as<func1_t, func2_t>);
    }

    SECTION("move construction works in all directions with both types")
    {
      STATIC_REQUIRE(std::constructible_from<func1_t, func1_t>);
      STATIC_REQUIRE(std::constructible_from<func1_t, func2_t>);
      STATIC_REQUIRE(std::constructible_from<func2_t, func1_t>);
      STATIC_REQUIRE(std::constructible_from<func2_t, func2_t>);
    }

    SECTION("copy construction works in all directions with both types")
    {
      STATIC_REQUIRE(std::constructible_from<func1_t, func1_t const &>);
      STATIC_REQUIRE(std::constructible_from<func1_t, func2_t const &>);
      STATIC_REQUIRE(std::constructible_from<func2_t, func1_t const &>);
      STATIC_REQUIRE(std::constructible_from<func2_t, func2_t const &>);
    }

    SECTION("move-assignment works in every direction with both types")
    {
      STATIC_REQUIRE(std::assignable_from<func1_t &, func1_t>);
      STATIC_REQUIRE(std::assignable_from<func1_t &, func2_t>);
      STATIC_REQUIRE(std::assignable_from<func2_t &, func1_t>);
      STATIC_REQUIRE(std::assignable_from<func2_t &, func2_t>);
    }

    SECTION("copy-assignment works in every direction with both types")
    {
      STATIC_REQUIRE(std::assignable_from<func1_t &, func1_t const &>);
      STATIC_REQUIRE(std::assignable_from<func1_t &, func2_t const &>);
      STATIC_REQUIRE(std::assignable_from<func2_t &, func1_t const &>);
      STATIC_REQUIRE(std::assignable_from<func2_t &, func2_t const &>);
    }
  }

  struct none_such
  {};

  template <class Completion>
  inline constexpr auto get_completion_domain =
    ex::__first_callable{ex::get_completion_domain<Completion>, ex::__always{none_such()}};

  TEST_CASE("function reports a default completion domain by default", "[types][function]")
  {
    SECTION("throwing function reports a completion domain for all three channels")
    {
      exec::function<void()> fn(ex::just);
      auto                   attrs        = ex::get_env(fn);
      auto                   value_domain = get_completion_domain<ex::set_value_t>(attrs);
      auto                   error_domain = get_completion_domain<ex::set_error_t>(attrs);
      auto                   stop_domain  = get_completion_domain<ex::set_stopped_t>(attrs);

      STATIC_REQUIRE(std::same_as<ex::default_domain, decltype(value_domain)>);
      STATIC_REQUIRE(std::same_as<ex::default_domain, decltype(error_domain)>);
      STATIC_REQUIRE(std::same_as<ex::default_domain, decltype(stop_domain)>);
    }

    SECTION("no-throw function reports a completion domain for value and stop channels only")
    {
      exec::function<void() noexcept> fn(ex::just);
      auto                            attrs        = ex::get_env(fn);
      auto                            value_domain = get_completion_domain<ex::set_value_t>(attrs);
      auto                            error_domain = get_completion_domain<ex::set_error_t>(attrs);
      auto                            stop_domain = get_completion_domain<ex::set_stopped_t>(attrs);

      STATIC_REQUIRE(std::same_as<ex::default_domain, decltype(value_domain)>);
      STATIC_REQUIRE(std::same_as<none_such, decltype(error_domain)>);
      STATIC_REQUIRE(std::same_as<ex::default_domain, decltype(stop_domain)>);
    }

    SECTION("infallible function reports a completion domain for value channel only")
    {
      exec::function<ex::sender_tag(), ex::completion_signatures<ex::set_value_t()>> fn(ex::just);
      auto attrs        = ex::get_env(fn);
      auto value_domain = get_completion_domain<ex::set_value_t>(attrs);
      auto error_domain = get_completion_domain<ex::set_error_t>(attrs);
      auto stop_domain  = get_completion_domain<ex::set_stopped_t>(attrs);

      STATIC_REQUIRE(std::same_as<ex::default_domain, decltype(value_domain)>);
      STATIC_REQUIRE(std::same_as<none_such, decltype(error_domain)>);
      STATIC_REQUIRE(std::same_as<none_such, decltype(stop_domain)>);
    }

    SECTION("just_error function reports a completion domain for error channel only")
    {
      exec::function<ex::sender_tag(int), ex::completion_signatures<ex::set_error_t(int)>> fn(
        42,
        ex::just_error);
      auto attrs        = ex::get_env(fn);
      auto value_domain = get_completion_domain<ex::set_value_t>(attrs);
      auto error_domain = get_completion_domain<ex::set_error_t>(attrs);
      auto stop_domain  = get_completion_domain<ex::set_stopped_t>(attrs);

      STATIC_REQUIRE(std::same_as<none_such, decltype(value_domain)>);
      STATIC_REQUIRE(std::same_as<ex::default_domain, decltype(error_domain)>);
      STATIC_REQUIRE(std::same_as<none_such, decltype(stop_domain)>);
    }

    SECTION("just_stopped function reports a completion domain for stop channel only")
    {
      exec::function<ex::sender_tag(), ex::completion_signatures<ex::set_stopped_t()>> fn(
        ex::just_stopped);
      auto attrs        = ex::get_env(fn);
      auto value_domain = get_completion_domain<ex::set_value_t>(attrs);
      auto error_domain = get_completion_domain<ex::set_error_t>(attrs);
      auto stop_domain  = get_completion_domain<ex::set_stopped_t>(attrs);

      STATIC_REQUIRE(std::same_as<none_such, decltype(value_domain)>);
      STATIC_REQUIRE(std::same_as<none_such, decltype(error_domain)>);
      STATIC_REQUIRE(std::same_as<ex::default_domain, decltype(stop_domain)>);
    }
  }

  struct domain : ex::default_domain
  {};

  TEST_CASE("function's constructor is constrained based on the common domain", "[types][function]")
  {
    using queries = exec::queries<domain(ex::get_domain_t) noexcept>;

    SECTION("the constraint applies to set_value")
    {
      using function =
        exec::function<ex::sender_tag(), ex::completion_signatures<ex::set_value_t()>, queries>;

      STATIC_REQUIRE(std::constructible_from<function, ex::just_t>);

      function fn(ex::just);
      auto     attrs        = ex::get_env(fn);
      auto     value_domain = get_completion_domain<ex::set_value_t>(attrs);
      auto     error_domain = get_completion_domain<ex::set_error_t>(attrs);
      auto     stop_domain  = get_completion_domain<ex::set_stopped_t>(attrs);

      STATIC_REQUIRE(std::same_as<ex::default_domain, decltype(value_domain)>);
      STATIC_REQUIRE(std::same_as<none_such, decltype(error_domain)>);
      STATIC_REQUIRE(std::same_as<none_such, decltype(stop_domain)>);
    }

    SECTION("the constraint applies to set_error")
    {
      using function = exec::function<ex::sender_tag(int),
                                      ex::completion_signatures<ex::set_error_t(int)>,
                                      queries>;

      STATIC_REQUIRE(std::constructible_from<function, int, ex::just_error_t>);

      function fn(42, ex::just_error);
      auto     attrs        = ex::get_env(fn);
      auto     value_domain = get_completion_domain<ex::set_value_t>(attrs);
      auto     error_domain = get_completion_domain<ex::set_error_t>(attrs);
      auto     stop_domain  = get_completion_domain<ex::set_stopped_t>(attrs);

      STATIC_REQUIRE(std::same_as<none_such, decltype(value_domain)>);
      STATIC_REQUIRE(std::same_as<ex::default_domain, decltype(error_domain)>);
      STATIC_REQUIRE(std::same_as<none_such, decltype(stop_domain)>);
    }

    SECTION("the constraint applies to set_stopped")
    {
      using function =
        exec::function<ex::sender_tag(), ex::completion_signatures<ex::set_stopped_t()>, queries>;

      STATIC_REQUIRE(std::constructible_from<function, ex::just_stopped_t>);

      function fn(ex::just_stopped);
      auto     attrs        = ex::get_env(fn);
      auto     value_domain = get_completion_domain<ex::set_value_t>(attrs);
      auto     error_domain = get_completion_domain<ex::set_error_t>(attrs);
      auto     stop_domain  = get_completion_domain<ex::set_stopped_t>(attrs);

      STATIC_REQUIRE(std::same_as<none_such, decltype(value_domain)>);
      STATIC_REQUIRE(std::same_as<none_such, decltype(error_domain)>);
      STATIC_REQUIRE(std::same_as<ex::default_domain, decltype(stop_domain)>);
    }
  }

  template <auto Tag>
  using custom_domain_for =
    exec::attrs<domain(ex::get_completion_domain_t<std::remove_cvref_t<decltype(Tag)>>)>;

  TEST_CASE("function can't be constructed with a sender that completes in the wrong domain",
            "[types][function]")
  {
    SECTION("the constraint applies to set_value")
    {
      using function = exec::function<ex::sender_tag(),
                                      ex::completion_signatures<ex::set_value_t()>,
                                      exec::queries<>,
                                      custom_domain_for<ex::set_value>>;

      STATIC_REQUIRE(!std::constructible_from<function, ex::just_t>);

      // double check that it *would* work if the sender reported a custom domain
      STATIC_REQUIRE(std::constructible_from<function, domain_sender_t<ex::set_value_t, domain>>);
    }

    SECTION("the constraint applies to set_error")
    {
      using function = exec::function<ex::sender_tag(int),
                                      ex::completion_signatures<ex::set_error_t(int)>,
                                      exec::queries<>,
                                      custom_domain_for<ex::set_error>>;

      STATIC_REQUIRE(!std::constructible_from<function, int, ex::just_error_t>);

      // double check that it *would* work if the sender reported a custom domain
      STATIC_REQUIRE(
        std::constructible_from<function, int, domain_sender_t<ex::set_error_t, domain>>);
    }

    SECTION("the constraint applies to set_stopped")
    {
      using function = exec::function<ex::sender_tag(),
                                      ex::completion_signatures<ex::set_stopped_t()>,
                                      exec::queries<>,
                                      custom_domain_for<ex::set_stopped>>;

      STATIC_REQUIRE(!std::constructible_from<function, ex::just_stopped_t>);

      // double check that it *would* work if the sender reported a custom domain
      STATIC_REQUIRE(std::constructible_from<function, domain_sender_t<ex::set_stopped_t, domain>>);
    }
  }

  template <class Sigs, class Attrs>
  concept function_exists = requires { typename exec::function<ex::sender_tag(), Sigs, Attrs>; };

  TEST_CASE("function can't be specialized with invalid completion specifications")
  {
    SECTION("specifying a completion signature with no corresponding completion domain is fine")
    {
      STATIC_REQUIRE(function_exists<ex::completion_signatures<ex::set_value_t()>, exec::attrs<>>);
      STATIC_REQUIRE(
        function_exists<ex::completion_signatures<ex::set_error_t(int)>, exec::attrs<>>);
      STATIC_REQUIRE(
        function_exists<ex::completion_signatures<ex::set_stopped_t()>, exec::attrs<>>);
    }

    SECTION("specifying a completion domain is fine if you also specify a corresponding signature")
    {
      STATIC_REQUIRE(
        function_exists<ex::completion_signatures<ex::set_value_t()>,
                        exec::attrs<domain(ex::get_completion_domain_t<ex::set_value_t>)>>);
      STATIC_REQUIRE(
        function_exists<ex::completion_signatures<ex::set_error_t(int)>,
                        exec::attrs<domain(ex::get_completion_domain_t<ex::set_error_t>)>>);
      STATIC_REQUIRE(
        function_exists<ex::completion_signatures<ex::set_stopped_t()>,
                        exec::attrs<domain(ex::get_completion_domain_t<ex::set_stopped_t>)>>);
    }

    SECTION("you may not specify a completion domain if there's no corresponding signature")
    {
      STATIC_REQUIRE(
        !function_exists<ex::completion_signatures<ex::set_error_t(int)>,
                         exec::attrs<domain(ex::get_completion_domain_t<ex::set_value_t>)>>);
      STATIC_REQUIRE(
        !function_exists<ex::completion_signatures<ex::set_value_t(int)>,
                         exec::attrs<domain(ex::get_completion_domain_t<ex::set_error_t>)>>);
      STATIC_REQUIRE(
        !function_exists<ex::completion_signatures<ex::set_error_t(int)>,
                         exec::attrs<domain(ex::get_completion_domain_t<ex::set_stopped_t>)>>);
    }

    SECTION("you may specify only some completion domains")
    {
      STATIC_REQUIRE(
        function_exists<
          ex::completion_signatures<ex::set_value_t(), ex::set_error_t(int), ex::set_stopped_t()>,
          exec::attrs<domain(ex::get_completion_domain_t<ex::set_value_t>)>>);
      STATIC_REQUIRE(
        function_exists<
          ex::completion_signatures<ex::set_value_t(), ex::set_error_t(int), ex::set_stopped_t()>,
          exec::attrs<domain(ex::get_completion_domain_t<ex::set_error_t>)>>);
      STATIC_REQUIRE(
        function_exists<
          ex::completion_signatures<ex::set_value_t(), ex::set_error_t(int), ex::set_stopped_t()>,
          exec::attrs<domain(ex::get_completion_domain_t<ex::set_stopped_t>)>>);
    }

    SECTION("sender attributes other than completion domain queries don't break")
    {
      // TODO: it's not obvious that it makes sense to support sender attributes other than
      //       completion domain queries so this may be silly....
      auto query = [](auto const &)
      {
        return 0;
      };
      using query_t = decltype(query);

      STATIC_REQUIRE(
        function_exists<ex::completion_signatures<ex::set_value_t()>, exec::attrs<int(query_t)>>);
    }
  }
}  // namespace
