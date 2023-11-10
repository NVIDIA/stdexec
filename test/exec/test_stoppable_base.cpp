/*
 * Copyright (c) 2026 Lauri Ojell-Vasama
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

#include <exec/stoppable_base.hpp>

#include <catch2/catch_all.hpp>

#include <atomic>
#include <functional>
#include <thread>
#include <type_traits>

#if __has_include(<stop_token>) && __cpp_lib_jthread >= 201911L
#  include <stop_token>
#  define STDEXEC_TEST_STD_STOP_TOKEN 1
#endif

namespace ex = STDEXEC;

namespace
{
  // on_stop_requested() counts calls and then runs an optional action.
  template <class StopToken>
  class stoppable : exec::stoppable_base_for_t<StopToken, stoppable<StopToken>>
  {
    using base = exec::stoppable_base_for_t<StopToken, stoppable<StopToken>>;

   public:
    stoppable() = default;

    using base::deregister_stoppable;
    using base::register_stoppable;

    std::atomic<int>                calls{0};
    std::function<void(stoppable&)> on_stop;

   private:
    void on_stop_requested() noexcept
    {
      ++calls;
      if (on_stop)
        on_stop(*this);
    }

    friend base;
  };

  template <class StopSource>
  using stoppable_for = stoppable<decltype(std::declval<StopSource&>().get_token())>;

#if STDEXEC_TEST_STD_STOP_TOKEN
#  define STOP_SOURCES ex::inplace_stop_source, std::stop_source
#else
#  define STOP_SOURCES ex::inplace_stop_source
#endif

  TEMPLATE_TEST_CASE("request_stop after registration signals stoppable_base_for",
                     "[other][stoppable_base_for]",
                     STOP_SOURCES)
  {
    TestType                source;
    stoppable_for<TestType> s;
    s.register_stoppable(source.get_token());
    REQUIRE(s.calls == 0);

    source.request_stop();
    REQUIRE(s.calls == 1);
  }

  TEMPLATE_TEST_CASE("request_stop before registration signals stoppable_base_for",
                     "[other][stoppable_base_for]",
                     STOP_SOURCES)
  {
    TestType source;
    source.request_stop();

    stoppable_for<TestType> s;
    s.register_stoppable(source.get_token());
    REQUIRE(s.calls == 1);
  }

  TEMPLATE_TEST_CASE("request_stop after destruction does not signal stoppable_base_for",
                     "[other][stoppable_base_for]",
                     STOP_SOURCES)
  {
    TestType         source;
    std::atomic<int> calls{0};
    {
      stoppable_for<TestType> s;
      s.on_stop = [&](auto&)
      {
        ++calls;
      };
      s.register_stoppable(source.get_token());
    }

    source.request_stop();
    REQUIRE(calls == 0);
  }

  TEMPLATE_TEST_CASE("stoppable_base_for is not registered on construction",
                     "[other][stoppable_base_for]",
                     STOP_SOURCES)
  {
    TestType source;
    {
      stoppable_for<TestType> s;
    }
    source.request_stop();

    stoppable_for<TestType> s;
    REQUIRE(s.calls == 0);
  }

  TEMPLATE_TEST_CASE("stoppable_base_for without a stop source",
                     "[other][stoppable_base_for]",
                     STOP_SOURCES)
  {
    using token_t = decltype(std::declval<TestType&>().get_token());
    stoppable_for<TestType> s;
    s.register_stoppable(token_t{});
    s.deregister_stoppable();
    REQUIRE(s.calls == 0);
  }

  TEMPLATE_TEST_CASE("deregister_stoppable is idempotent",
                     "[other][stoppable_base_for]",
                     STOP_SOURCES)
  {
    TestType                source;
    stoppable_for<TestType> s;
    s.deregister_stoppable();
    s.register_stoppable(source.get_token());
    s.deregister_stoppable();
    s.deregister_stoppable();

    source.request_stop();
    REQUIRE(s.calls == 0);
  }

  TEMPLATE_TEST_CASE("stoppable_base_for can be registered again after deregistration",
                     "[other][stoppable_base_for]",
                     STOP_SOURCES)
  {
    TestType                source;
    stoppable_for<TestType> s;
    s.register_stoppable(source.get_token());
    s.deregister_stoppable();
    s.register_stoppable(source.get_token());

    source.request_stop();
    REQUIRE(s.calls == 1);
  }

  TEMPLATE_TEST_CASE("stoppable_base_for can be registered again after being stopped",
                     "[other][stoppable_base_for]",
                     STOP_SOURCES)
  {
    TestType                source;
    stoppable_for<TestType> s;
    s.register_stoppable(source.get_token());
    source.request_stop();
    REQUIRE(s.calls == 1);

    s.deregister_stoppable();
    s.register_stoppable(source.get_token());  // stop already requested: runs inline
    REQUIRE(s.calls == 2);
  }

  TEMPLATE_TEST_CASE("stoppable_base_for can be registered with a different source",
                     "[other][stoppable_base_for]",
                     STOP_SOURCES)
  {
    TestType                first;
    TestType                second;
    stoppable_for<TestType> s;
    s.register_stoppable(first.get_token());
    s.deregister_stoppable();
    s.register_stoppable(second.get_token());

    first.request_stop();
    REQUIRE(s.calls == 0);
    second.request_stop();
    REQUIRE(s.calls == 1);
  }

  TEMPLATE_TEST_CASE("stoppable_base_for can deregister from on_stop_requested",
                     "[other][stoppable_base_for]",
                     STOP_SOURCES)
  {
    TestType                source;
    stoppable_for<TestType> s;
    s.on_stop = [](auto& self)
    {
      self.deregister_stoppable();
    };
    s.register_stoppable(source.get_token());

    source.request_stop();
    REQUIRE(s.calls == 1);
  }

  TEMPLATE_TEST_CASE("stoppable_base_for can be destroyed from on_stop_requested",
                     "[other][stoppable_base_for]",
                     STOP_SOURCES)
  {
    TestType source;
    auto*    s = new stoppable_for<TestType>();
    s->on_stop = [](auto& self)
    {
      delete &self;
    };
    s->register_stoppable(source.get_token());

    source.request_stop();
  }

  TEMPLATE_TEST_CASE("stoppable_base_for deregistration races with request_stop",
                     "[other][stoppable_base_for]",
                     STOP_SOURCES)
  {
    for (int i = 0; i < 1000; ++i)
    {
      TestType                source;
      stoppable_for<TestType> s;
      s.register_stoppable(source.get_token());

      std::thread t([&] { source.request_stop(); });
      s.deregister_stoppable();
      int const calls = s.calls;  // no calls after deregister_stoppable() returns
      t.join();

      REQUIRE(calls <= 1);
      REQUIRE(s.calls == calls);
    }
  }

  struct first_tag
  {};

  struct second_tag
  {};

  // Two stoppable bases on one object, told apart by their tags.
  template <class Token1, class Token2>
  class two_stoppables
    : exec::stoppable_base_for_t<Token1, two_stoppables<Token1, Token2>, first_tag>
    , exec::stoppable_base_for_t<Token2, two_stoppables<Token1, Token2>, second_tag>
  {
    using first_base  = exec::stoppable_base_for_t<Token1, two_stoppables, first_tag>;
    using second_base = exec::stoppable_base_for_t<Token2, two_stoppables, second_tag>;

   public:
    void register_first(Token1 token)
    {
      first_base::register_stoppable(std::move(token));
    }

    void register_second(Token2 token)
    {
      second_base::register_stoppable(std::move(token));
    }

    void deregister_first()
    {
      first_base::deregister_stoppable();
    }

    std::atomic<int> first_calls{0};
    std::atomic<int> second_calls{0};

   private:
    void on_stop_requested(first_tag) noexcept
    {
      ++first_calls;
    }

    void on_stop_requested(second_tag) noexcept
    {
      ++second_calls;
    }

    friend first_base;
    friend second_base;
  };

  template <class Source1, class Source2>
  struct source_pair
  {
    using first_t  = Source1;
    using second_t = Source2;
  };

#if STDEXEC_TEST_STD_STOP_TOKEN
#  define SOURCE_PAIRS                                                                            \
    (source_pair<ex::inplace_stop_source, ex::inplace_stop_source>),                              \
      (source_pair<ex::inplace_stop_source, std::stop_source>),                                   \
      (source_pair<std::stop_source, std::stop_source>)
#else
#  define SOURCE_PAIRS (source_pair<ex::inplace_stop_source, ex::inplace_stop_source>)
#endif

  TEMPLATE_TEST_CASE("stoppable_base_for tags distinguish several bases",
                     "[other][stoppable_base_for]",
                     SOURCE_PAIRS)
  {
    typename TestType::first_t                                                first;
    typename TestType::second_t                                               second;
    two_stoppables<decltype(first.get_token()), decltype(second.get_token())> s;
    s.register_first(first.get_token());
    s.register_second(second.get_token());

    first.request_stop();
    REQUIRE(s.first_calls == 1);
    REQUIRE(s.second_calls == 0);

    second.request_stop();
    REQUIRE(s.first_calls == 1);
    REQUIRE(s.second_calls == 1);
  }

  TEMPLATE_TEST_CASE("stoppable_base_for tagged bases deregister independently",
                     "[other][stoppable_base_for]",
                     SOURCE_PAIRS)
  {
    typename TestType::first_t                                                first;
    typename TestType::second_t                                               second;
    two_stoppables<decltype(first.get_token()), decltype(second.get_token())> s;
    s.register_first(first.get_token());
    s.register_second(second.get_token());
    s.deregister_first();

    first.request_stop();
    second.request_stop();
    REQUIRE(s.first_calls == 0);
    REQUIRE(s.second_calls == 1);
  }

  TEST_CASE("stoppable_base_for with never_stop_token", "[other][stoppable_base_for]")
  {
    using base_t =
      exec::stoppable_base_for_t<ex::never_stop_token, stoppable<ex::never_stop_token>>;
    STATIC_REQUIRE(std::is_empty_v<base_t>);

    stoppable<ex::never_stop_token> s;
    s.register_stoppable(ex::never_stop_token{});
    s.deregister_stoppable();
    REQUIRE(s.calls == 0);
  }

  struct pointer_callback
  {
    void* self;

    void operator()() const noexcept {}
  };

  TEST_CASE("stoppable_base_for sizes", "[other][stoppable_base_for]")
  {
    STATIC_REQUIRE(sizeof(ex::inplace_stop_callback<pointer_callback>) == 5 * sizeof(void*));
    STATIC_REQUIRE(sizeof(exec::inplace_stoppable_base<stoppable<ex::inplace_stop_token>>)
                   == 4 * sizeof(void*));
#if STDEXEC_TEST_STD_STOP_TOKEN
    STATIC_REQUIRE(sizeof(exec::stoppable_base_for_t<std::stop_token, stoppable<std::stop_token>>)
                   == sizeof(ex::__optional<std::stop_callback<pointer_callback>>));
#endif
  }

}  // namespace
