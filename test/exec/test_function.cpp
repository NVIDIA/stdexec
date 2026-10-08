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

#include <test_common/catch2.hpp>

#include <stdexec/execution.hpp>

#include <array>
#include <cstddef>
#include <exception>
#include <memory>
#include <memory_resource>
#include <stdexcept>
#include <string>
#include <utility>

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

    SECTION("void() from just_stopped()")
    {
      exec::function<void()> sndr([]() noexcept { return ex::just_stopped(); });

      auto ret = ex::sync_wait(std::move(sndr));

      REQUIRE_FALSE(ret.has_value());
    }

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
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

    SECTION("custom completions from just_error(42)")
    {
      exec::function<ex::sender_tag(),
                     ex::completion_signatures<ex::set_value_t(), ex::set_error_t(int)>>
        sndr([]() noexcept { return ex::just_error(42); });

      REQUIRE_THROWS_AS(ex::sync_wait(std::move(sndr)), int);
    }
#endif  // !STDEXEC_NO_STDCPP_EXCEPTIONS()
  }

  struct counting_resource : std::pmr::memory_resource
  {
    std::size_t count = 0;

    void *do_allocate(std::size_t bytes, std::size_t alignment) override
    {
      ++count;
      return std::pmr::get_default_resource()->allocate(bytes, alignment);
    }

    void do_deallocate(void *p, std::size_t bytes, std::size_t alignment) override
    {
      std::pmr::get_default_resource()->deallocate(p, bytes, alignment);
    }

    bool do_is_equal(std::pmr::memory_resource const &other) const noexcept override
    {
      return &other == this;
    }
  };

  TEST_CASE("exec::function forwards get_frame_allocator", "[types][function]")
  {
    counting_resource                                      res;
    exec::function<bool(counting_resource & res) noexcept> sndr(
      res,
      [](auto &res) noexcept
      {
        return ex::read_env(exec::get_frame_allocator)
             | ex::then(
                 [&res](auto alloc) noexcept
                 {
                   auto count = res.count;
                   alloc.deallocate(alloc.allocate(16), 16);
                   return res.count > count;
                 });
      });

    std::pmr::polymorphic_allocator<std::byte> alloc{&res};

    auto [ret] = ex::sync_wait(std::move(sndr)
                               | ex::write_env(ex::prop(exec::get_frame_allocator, alloc)))
                   .value();

    REQUIRE(ret);
  }

  TEST_CASE("exec::function accepts a pointer to a derived memory_resource as frame allocator",
            "[types][function]")
  {
    counting_resource res;
    exec::function<int() noexcept,
                   exec::queries<counting_resource *(exec::get_frame_allocator_t) noexcept>>
      sndr([]() noexcept { return ex::just(42); });

    auto [ret] = ex::sync_wait(std::move(sndr)
                               | ex::write_env(ex::prop(exec::get_frame_allocator, &res)))
                   .value();

    REQUIRE(ret == 42);
  }

  TEST_CASE("exec::function allocates a large operation state with the frame allocator",
            "[types][function]")
  {
    // big enough that the erased operation state can't be stored inline
    using big = std::array<char, 256>;

    counting_resource                          res;
    std::pmr::polymorphic_allocator<std::byte> alloc{&res};

    exec::function<big() noexcept> sndr([]() noexcept { return ex::just(big{}); });

    auto [ret] = ex::sync_wait(std::move(sndr)
                               | ex::write_env(ex::prop(exec::get_frame_allocator, alloc)))
                   .value();

    REQUIRE(ret == big{});
    REQUIRE(res.count == 1);
  }

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
  TEST_CASE("an exception thrown by the sender factory propagates out of connect",
            "[types][function]")
  {
    exec::function<int()> sndr([]() -> decltype(ex::just(0))
                               { throw std::runtime_error("factory failed"); });

    REQUIRE_THROWS_AS(ex::sync_wait(std::move(sndr)), std::runtime_error);
  }
#endif

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
    virtual exec::function<int() const & noexcept> get_i_virtually() const noexcept = 0;
  };

  struct iface2
  {
    exec::function<int() const & noexcept> get_i_from_base() const noexcept
    {
      return exec::function<int() const & noexcept>(*this, &iface2::get_i_virtually);
    }

    virtual exec::function<int() const & noexcept> get_i_virtually() const noexcept = 0;
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

    exec::function<int(impl const *) noexcept> get_i_with_pmfn() const noexcept
    {
      return exec::function<int(impl const *) noexcept>(this, &impl::just_i);
    }

    exec::function<int() const & noexcept> get_i_virtually() const noexcept override
    {
      return exec::function<int() const & noexcept>(*this, &impl::just_i);
    }

   private:
    int i_;
  };

  struct sender_holder
  {
    decltype(ex::just(42)) sndr = ex::just(42);
  };

  struct move_only_sender_holder
  {
    decltype(ex::just(std::unique_ptr<int>{})) sndr = ex::just(std::unique_ptr<int>{});
  };

  TEST_CASE("exec::function accepts only stateless sender factories", "[types][function]")
  {
    SECTION("function<int(sender_holder const *) noexcept> accepts a pointer to member data")
    {
      sender_holder h;
      auto [ret] = ex::sync_wait(
                     exec::function<int(sender_holder const *) noexcept>(&h, &sender_holder::sndr))
                     .value();

      REQUIRE(ret == 42);
    }

    SECTION("a pointer to member data yields an lvalue, so the sender must be copyable")
    {
      using function =
        exec::function<std::unique_ptr<int>(move_only_sender_holder const *) noexcept>;
      using factory = decltype(&move_only_sender_holder::sndr);

      STATIC_REQUIRE(!std::constructible_from<function, move_only_sender_holder const *, factory>);
    }

    SECTION("a callable with state is rejected, however small")
    {
      using function = exec::function<int() noexcept>;

      int  i        = 42;
      auto stateful = [i]() noexcept
      {
        return ex::just(i);
      };
      auto stateless = []() noexcept
      {
        return ex::just(42);
      };

      STATIC_REQUIRE(!std::constructible_from<function, decltype(stateful)>);
      STATIC_REQUIRE(std::constructible_from<function, decltype(stateless)>);
    }

    SECTION("the factory must be invocable as a const lvalue")
    {
      using function = exec::function<int()>;

      struct non_const_call
      {
        auto operator()() noexcept
        {
          return ex::just(42);
        }
      };

      struct rvalue_call
      {
        auto operator()() && noexcept
        {
          return ex::just(42);
        }
      };

      STATIC_REQUIRE(!std::constructible_from<function, non_const_call>);
      STATIC_REQUIRE(!std::constructible_from<function, rvalue_call>);
    }
  }

  TEST_CASE("exec::function accepts small trivially-copyable callables", "[types][function]")
  {
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

    SECTION("function<int() const & noexcept> can be the return type of a virtual member function")
    {
      auto [ret] = ex::sync_wait(impl{42}.get_i_virtually()).value();

      REQUIRE(ret == 42);
    }

    SECTION("function<int(iface const *)> accepts a pointer-to-member function")
    {
      impl imp{42};
      auto [ret] =
        ex::sync_wait(exec::function<int(iface const *)>(&imp, &iface::get_i_virtually)).value();

      REQUIRE(ret == 42);
    }

    SECTION("function<int() const & noexcept> works on the base class")
    {
      auto [ret] = ex::sync_wait(impl{42}.get_i_from_base()).value();

      REQUIRE(ret == 42);
    }
  }

  TEST_CASE("noexcept is part of a function's type, separately from its completions",
            "[types][function]")
  {
    using sigs =
      ex::completion_signatures<ex::set_value_t(int), ex::set_error_t(std::exception_ptr)>;

    using throwing_t = exec::function<ex::sender_tag(), sigs>;
    using nothrow_t  = exec::function<ex::sender_tag() noexcept, sigs>;

    STATIC_REQUIRE(!std::same_as<throwing_t, nothrow_t>);
    // the sender_tag form's completions are exactly as declared, noexcept or not
    STATIC_REQUIRE(std::same_as<ex::completion_signatures_of_t<throwing_t>,
                                ex::completion_signatures_of_t<nothrow_t>>);

    // the noexcept convenience form drops set_error(exception_ptr) but is
    // still a distinct type from the equivalent sender_tag form without noexcept
    STATIC_REQUIRE(
      !std::same_as<
        exec::function<int() noexcept>,
        exec::function<ex::sender_tag(),
                       ex::completion_signatures<ex::set_value_t(int), ex::set_stopped_t()>>>);
    STATIC_REQUIRE(
      std::same_as<
        exec::function<int() noexcept>,
        exec::function<ex::sender_tag() noexcept,
                       ex::completion_signatures<ex::set_value_t(int), ex::set_stopped_t()>>>);
  }

  TEST_CASE("completion_signature specification is order-independent", "[types][function]")
  {
    // by specifying the completions with a function signature, it's up to the
    // library what order the completion signatures are specified in
    using func1_t = exec::function<int(int) noexcept>;
    // this declaration chooses value before stopped
    using func2_t =
      exec::function<ex::sender_tag(int) noexcept,
                     ex::completion_signatures<ex::set_value_t(int), ex::set_stopped_t()>>;
    // this declaration chooses stopped before value
    using func3_t =
      exec::function<ex::sender_tag(int) noexcept,
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

  TEST_CASE("specifications are deduplicated, and attrs are order-independent", "[types][function]")
  {
    SECTION("duplicate completion signatures collapse")
    {
      using func1_t =
        exec::function<ex::sender_tag(),
                       ex::completion_signatures<ex::set_value_t(), ex::set_stopped_t()>>;
      using func2_t = exec::function<
        ex::sender_tag(),
        ex::completion_signatures<ex::set_stopped_t(), ex::set_value_t(), ex::set_value_t()>>;

      STATIC_REQUIRE(std::same_as<func1_t, func2_t>);
    }

    SECTION("attrs order doesn't matter")
    {
      struct my_domain : ex::default_domain
      {};

      using sigs  = ex::completion_signatures<ex::set_value_t(), ex::set_stopped_t()>;
      using value = my_domain(ex::get_completion_domain_t<ex::set_value_t>) noexcept;
      using stop  = my_domain(ex::get_completion_domain_t<ex::set_stopped_t>) noexcept;

      using func1_t =
        exec::function<ex::sender_tag(), sigs, exec::queries<>, exec::attrs<value, stop>>;
      using func2_t =
        exec::function<ex::sender_tag(), sigs, exec::queries<>, exec::attrs<stop, value>>;

      STATIC_REQUIRE(std::same_as<func1_t, func2_t>);
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

      // double check that it *would* work if the sender reported a custom
      // domain
      STATIC_REQUIRE(std::constructible_from<function, domain_sender_t<ex::set_value_t, domain>>);
    }

    SECTION("the constraint applies to set_error")
    {
      using function = exec::function<ex::sender_tag(int),
                                      ex::completion_signatures<ex::set_error_t(int)>,
                                      exec::queries<>,
                                      custom_domain_for<ex::set_error>>;

      STATIC_REQUIRE(!std::constructible_from<function, int, ex::just_error_t>);

      // double check that it *would* work if the sender reported a custom
      // domain
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

      // double check that it *would* work if the sender reported a custom
      // domain
      STATIC_REQUIRE(std::constructible_from<function, domain_sender_t<ex::set_stopped_t, domain>>);
    }
  }

  template <class Sigs, class Attrs>
  concept function_exists = requires { typename exec::function<ex::sender_tag(), Sigs, Attrs>; };

  TEST_CASE("get_completion_domain_t<> in attrs<...> means the set_value domain",
            "[types][function]")
  {
    using function = exec::function<ex::sender_tag(),
                                    ex::completion_signatures<ex::set_value_t()>,
                                    exec::queries<>,
                                    exec::attrs<domain(ex::get_completion_domain_t<>)>>;

    // the declared domain constrains the erased sender's set_value domain
    STATIC_REQUIRE(!std::constructible_from<function, ex::just_t>);
    STATIC_REQUIRE(std::constructible_from<function, domain_sender_t<ex::set_value_t, domain>>);

    // and the function reports it for both spellings of the value-channel query
    function fn(domain_sender<ex::set_value, domain>);
    auto     attrs = ex::get_env(fn);

    STATIC_REQUIRE(std::same_as<domain, decltype(get_completion_domain<ex::set_value_t>(attrs))>);
    STATIC_REQUIRE(std::same_as<domain, decltype(get_completion_domain<void>(attrs))>);
  }

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
      // TODO: it's not obvious that it makes sense to support sender attributes
      //       other than completion domain queries so this may be silly....
      auto query = [](auto const &)
      {
        return 0;
      };
      using query_t = decltype(query);

      STATIC_REQUIRE(
        function_exists<ex::completion_signatures<ex::set_value_t()>, exec::attrs<int(query_t)>>);
    }
  }

  struct pointer_factories
  {
    int i;

    static auto just_i(pointer_factories const &self) noexcept
    {
      return ex::just(self.i);
    }

    auto just_i_memfn() const noexcept
    {
      return ex::just(i);
    }
  };

  TEST_CASE("member-function functions built from pointer factories are nothrow constructible",
            "[types][function]")
  {
    using function = exec::function<int() const & noexcept>;
    using self     = pointer_factories const &;

    SECTION("pointer-to-function factory")
    {
      using factory = decltype(&pointer_factories::just_i);
      STATIC_REQUIRE(std::is_nothrow_constructible_v<function, self, factory>);

      pointer_factories pf{42};
      auto [ret] = ex::sync_wait(function(std::as_const(pf), &pointer_factories::just_i)).value();
      REQUIRE(ret == 42);
    }

    SECTION("pointer-to-member-function factory")
    {
      using factory = decltype(&pointer_factories::just_i_memfn);
      STATIC_REQUIRE(std::is_nothrow_constructible_v<function, self, factory>);

      pointer_factories pf{42};
      auto [ret] =
        ex::sync_wait(function(std::as_const(pf), &pointer_factories::just_i_memfn)).value();
      REQUIRE(ret == 42);
    }

    SECTION("pointer-to-member-data factory")
    {
      using factory = decltype(&sender_holder::sndr);
      STATIC_REQUIRE(std::is_nothrow_constructible_v<function, sender_holder const &, factory>);

      sender_holder h;
      auto [ret] = ex::sync_wait(function(std::as_const(h), &sender_holder::sndr)).value();
      REQUIRE(ret == 42);
    }
  }

  TEST_CASE("member-function functions require factories invocable as const lvalues",
            "[types][function]")
  {
    using function = exec::function<int() const &>;

    struct non_const_call
    {
      auto operator()(pointer_factories const &self) noexcept
      {
        return ex::just(self.i);
      }
    };

    struct const_call
    {
      auto operator()(pointer_factories const &self) const noexcept
      {
        return ex::just(self.i);
      }
    };

    STATIC_REQUIRE(!std::constructible_from<function, pointer_factories const &, non_const_call>);
    STATIC_REQUIRE(std::constructible_from<function, pointer_factories const &, const_call>);

    pointer_factories pf{42};
    auto [ret] = ex::sync_wait(function(pf, const_call{})).value();
    REQUIRE(ret == 42);
  }

  TEST_CASE("member-function functions accept the self arguments a synchronous member function "
            "would, except rvalues",
            "[types][function]")
  {
    using factory = decltype(&pointer_factories::just_i);

    SECTION("const & accepts const and non-const lvalues")
    {
      using fn = exec::function<int() const &>;
      STATIC_REQUIRE(std::constructible_from<fn, pointer_factories const &, factory>);
      STATIC_REQUIRE(std::constructible_from<fn, pointer_factories &, factory>);
      STATIC_REQUIRE(!std::constructible_from<fn, pointer_factories const &&, factory>);
      STATIC_REQUIRE(!std::constructible_from<fn, pointer_factories &&, factory>);
      STATIC_REQUIRE(!std::constructible_from<fn, pointer_factories volatile &, factory>);
      STATIC_REQUIRE(!std::constructible_from<fn, pointer_factories const volatile &, factory>);
    }

    SECTION("& accepts only non-const lvalues")
    {
      using fn       = exec::function<int() &>;
      using mfactory = decltype(ex::just(0)) (*)(pointer_factories &) noexcept;
      STATIC_REQUIRE(std::constructible_from<fn, pointer_factories &, mfactory>);
      STATIC_REQUIRE(!std::constructible_from<fn, pointer_factories const &, mfactory>);
      STATIC_REQUIRE(!std::constructible_from<fn, pointer_factories &&, mfactory>);
      STATIC_REQUIRE(!std::constructible_from<fn, pointer_factories volatile &, mfactory>);
    }

    SECTION("a non-const lvalue reaches a const & function's factory as a const reference")
    {
      pointer_factories pf{42};
      auto [ret] = ex::sync_wait(exec::function<int() const &>(
                                   pf,
                                   [](auto &self) noexcept
                                   {
                                     static_assert(
                                       std::is_const_v<std::remove_reference_t<decltype(self)>>);
                                     return ex::just(self.i);
                                   }))
                     .value();
      REQUIRE(ret == 42);
    }
  }

  struct trivial_move_only
  {
    trivial_move_only()                                     = default;
    trivial_move_only(trivial_move_only const &)            = delete;
    trivial_move_only(trivial_move_only &&)                 = default;
    trivial_move_only &operator=(trivial_move_only const &) = delete;
    trivial_move_only &operator=(trivial_move_only &&)      = default;
    ~trivial_move_only()                                    = default;
  };

  //! Trivially copy- and move-constructible, but not trivially copyable,
  //! because its assignment operators are user-provided.
  struct trivial_construction_only
  {
    trivial_construction_only()                                  = default;
    trivial_construction_only(trivial_construction_only const &) = default;
    trivial_construction_only(trivial_construction_only &&)      = default;
    trivial_construction_only &operator=(trivial_construction_only const &) noexcept
    {
      return *this;
    }
    trivial_construction_only &operator=(trivial_construction_only &&) noexcept
    {
      return *this;
    }
    ~trivial_construction_only() = default;
  };

  TEST_CASE("function takes trivially copy- and move-constructible curried arguments by const "
            "reference and all others by rvalue reference",
            "[types][function]")
  {
    using just_int_t = decltype(ex::just(0));

    SECTION("an int may be an lvalue or an rvalue")
    {
      using fn      = exec::function<int(int)>;
      using factory = just_int_t (*)(int);
      STATIC_REQUIRE(std::constructible_from<fn, int &, factory>);
      STATIC_REQUIRE(std::constructible_from<fn, int const &, factory>);
      STATIC_REQUIRE(std::constructible_from<fn, int, factory>);

      int i      = 42;
      auto [ret] = ex::sync_wait(fn(i, ex::just)).value();
      REQUIRE(ret == 42);
    }

    SECTION("a class with trivial copy and move constructors may be an lvalue, whatever its "
            "assignment operators")
    {
      // function only constructs its curried arguments, so their assignment
      // operators don't matter
      using type = trivial_construction_only;
      STATIC_REQUIRE(!std::is_trivially_copyable_v<type>);
      using fn      = exec::function<int(type)>;
      using factory = just_int_t (*)(type);
      STATIC_REQUIRE(std::constructible_from<fn, type &, factory>);
      STATIC_REQUIRE(std::constructible_from<fn, type, factory>);
    }

    SECTION("a type with a non-trivial copy or move must be an rvalue")
    {
      using fn      = exec::function<int(std::string)>;
      using factory = just_int_t (*)(std::string);
      STATIC_REQUIRE(!std::constructible_from<fn, std::string &, factory>);
      STATIC_REQUIRE(!std::constructible_from<fn, std::string const &, factory>);
      STATIC_REQUIRE(std::constructible_from<fn, std::string, factory>);
    }

    SECTION("a trivially copyable type whose copy constructor is deleted must be an rvalue")
    {
      STATIC_REQUIRE(std::is_trivially_copyable_v<trivial_move_only>);
      using fn      = exec::function<int(trivial_move_only)>;
      using factory = just_int_t (*)(trivial_move_only);
      STATIC_REQUIRE(!std::constructible_from<fn, trivial_move_only &, factory>);
      STATIC_REQUIRE(std::constructible_from<fn, trivial_move_only, factory>);
    }

    SECTION("reference-typed parameters are unchanged")
    {
      using lref_fn      = exec::function<int(int &)>;
      using lref_factory = just_int_t (*)(int &);
      STATIC_REQUIRE(std::constructible_from<lref_fn, int &, lref_factory>);
      STATIC_REQUIRE(!std::constructible_from<lref_fn, int const &, lref_factory>);
      STATIC_REQUIRE(!std::constructible_from<lref_fn, int, lref_factory>);

      using rref_fn      = exec::function<int(int &&)>;
      using rref_factory = just_int_t (*)(int &&);
      STATIC_REQUIRE(!std::constructible_from<rref_fn, int &, rref_factory>);
      STATIC_REQUIRE(std::constructible_from<rref_fn, int, rref_factory>);
    }

    SECTION("the rule applies to member functions' explicit arguments too")
    {
      struct example;
      using fn      = exec::function<int(int) const &>;
      using factory = just_int_t (*)(example const &, int);
      STATIC_REQUIRE(std::constructible_from<fn, example const &, int &, factory>);
      STATIC_REQUIRE(!std::constructible_from<exec::function<int(std::string) const &>,
                                              example const &,
                                              std::string &,
                                              just_int_t (*)(example const &, std::string)>);
    }
  }

  template <class Sig, class... Ts>
  concept valid_function_type = requires { typename exec::function<Sig, Ts...>; };

  TEST_CASE("function rejects rvalue, unqualified const, and volatile function types",
            "[types][function]")
  {
    using sigs = ex::completion_signatures<ex::set_value_t(int)>;

    STATIC_REQUIRE(valid_function_type<int()>);
    STATIC_REQUIRE(valid_function_type<int() &>);
    STATIC_REQUIRE(valid_function_type<int() const &>);
    STATIC_REQUIRE(valid_function_type<int() const & noexcept>);
    STATIC_REQUIRE(valid_function_type<ex::sender_tag() const &, sigs>);

    STATIC_REQUIRE(!valid_function_type<int() &&>);
    STATIC_REQUIRE(!valid_function_type<int() const &&>);
    STATIC_REQUIRE(!valid_function_type < int() && noexcept >);
    STATIC_REQUIRE(!valid_function_type < int() const && noexcept >);
    STATIC_REQUIRE(!valid_function_type<ex::sender_tag() &&, sigs>);

    STATIC_REQUIRE(!valid_function_type<int() const>);
    STATIC_REQUIRE(!valid_function_type<int() const noexcept>);
    STATIC_REQUIRE(!valid_function_type<ex::sender_tag() const, sigs>);

    STATIC_REQUIRE(!valid_function_type<int() volatile>);
    STATIC_REQUIRE(!valid_function_type<int() volatile &>);
    STATIC_REQUIRE(!valid_function_type<int() const volatile &>);
  }

  // Pointer-to-member factories whose representation can be larger than two
  // pointers under the Microsoft ABI, where the size of a pointer to member
  // depends on the class's inheritance model. Under the Itanium ABI every
  // pointer to member function is two pointers, so these only exercise
  // function's factory-storage sizing on Microsoft-ABI targets.
  struct mi_base1
  {
    int i = 1;
  };

  struct mi_base2
  {
    int j = 2;
  };

  struct multiple_inheritance
    : mi_base1
    , mi_base2
  {
    auto get() const noexcept
    {
      return ex::just(i + j);
    }
  };

  struct vi_base
  {
    int i = 3;
  };

  struct virtual_inheritance : virtual vi_base
  {
    auto get() const noexcept
    {
      return ex::just(i);
    }
  };

  struct late;
  // Formed while late is incomplete, so the Microsoft ABI must use its most
  // general pointer-to-member representation for this type.
  using late_getter = decltype(ex::just(0)) (late::*)() const noexcept;

  struct late
  {
    int i = 4;

    auto get() const noexcept
    {
      return ex::just(i);
    }
  };

#if !STDEXEC_MSVC() && !STDEXEC_CLANG_CL()
  template <class Function>
  struct function_then_char
  {
    STDEXEC_ATTRIBUTE(no_unique_address) Function fn;
    char c;
  };

  // Under the Itanium C++ ABI, a following member can reuse the tail padding of
  // a [[no_unique_address]] member whose type isn't POD for the purpose of
  // layout, so a function whose padding is all at its end leaves room for c.
  TEST_CASE("function's padding is at its end, where an enclosing object can reuse it",
            "[types][function]")
  {
    using takes_char = exec::function<int(char)>;
    using takes_int  = exec::function<int(int)>;

    STATIC_REQUIRE(sizeof(function_then_char<takes_char>) == sizeof(takes_char));
    STATIC_REQUIRE(sizeof(function_then_char<takes_int>) == sizeof(takes_int));
  }
#endif

  TEST_CASE("function stores pointer-to-member factories of any representation",
            "[types][function]")
  {
    SECTION("multiple inheritance")
    {
      multiple_inheritance obj;
      auto [ret] = ex::sync_wait(exec::function<int(multiple_inheritance const *) noexcept>(
                                   &obj,
                                   &multiple_inheritance::get))
                     .value();
      REQUIRE(ret == 3);
    }

    SECTION("virtual inheritance")
    {
      virtual_inheritance obj;
      auto [ret] = ex::sync_wait(exec::function<int(virtual_inheritance const *) noexcept>(
                                   &obj,
                                   &virtual_inheritance::get))
                     .value();
      REQUIRE(ret == 3);
    }

    SECTION("pointer to member formed before the class was complete")
    {
      late        obj;
      late_getter getter = &late::get;
      auto [ret] = ex::sync_wait(exec::function<int(late const *) noexcept>(&obj, getter)).value();
      REQUIRE(ret == 4);
    }
  }

  TEST_CASE("support for member functions works as expected", "[types][function]")
  {
    struct example
    {
      exec::function<int() const &> get_int() const &
      {
        return exec::function<int() const &>(*this,
                                             [](example const &self) { return ex::just(self.i_); });
      }

      exec::function<example &(int) &> set_int(int i) &
      {
        return exec::function<example &(int) &>(*this,
                                                int(i),
                                                [](auto &self, int i)
                                                {
                                                  return ex::just(i)
                                                       | ex::then(
                                                           [&self](int i) -> decltype(auto)
                                                           {
                                                             self.i_ = i;
                                                             return self;
                                                           });
                                                });
      }

     private:
      int i_{42};
    };

    {
      example e;

      auto [result] =
        ex::sync_wait(e.set_int(34) | ex::let_value([](auto &e) { return e.get_int(); })).value();

      REQUIRE(result == 34);
    }
  }

  template <class T>
  using async_getter =
    exec::function<ex::sender_tag() const &, ex::completion_signatures<ex::set_value_t(T)>>;

  template <class O, class T>
  using async_setter =
    exec::function<ex::sender_tag(T) &, ex::completion_signatures<ex::set_value_t(O &)>>;

  TEST_CASE("specifying more parameters of a function hiding a member function works",
            "[types][function]")
  {
    struct example
    {
      async_getter<int> get() const & noexcept
      {
        return async_getter<int>(*this, [](auto &self) noexcept { return ex::just(self.i_); });
      }

      async_setter<example, int> set(int i) & noexcept
      {
        return async_setter<example, int>(*this,
                                          int(i),
                                          [](auto &self, int i) noexcept
                                          {
                                            return ex::just(i)
                                                 | ex::then(
                                                     [&self](int i) noexcept -> example &
                                                     {
                                                       self.i_ = i;
                                                       return self;
                                                     });
                                          });
      }

     private:
      int i_{};
    };

    example e;

    auto [result] =
      ex::sync_wait(e.set(42) | ex::let_value([](auto &e) noexcept { return e.get(); })).value();

    REQUIRE(result == 42);
  }
}  // namespace
