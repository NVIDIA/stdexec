/*
 * Copyright (c) 2024 Rishabh Dwivedi <rishabhdwivedi17@gmail.com>
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

#include "exec/libdispatch_queue.hpp"
#include "stdexec/execution.hpp"
#include "test_common/catch2.hpp"
#include "test_common/type_helpers.hpp"

#include <atomic>
#include <numeric>
#include <utility>
#include <vector>

namespace
{
  struct lvalue_connect_sender
  {
    using sender_concept = STDEXEC::sender_tag;

    template <class, class...>
    static consteval auto
    get_completion_signatures() -> STDEXEC::completion_signatures<STDEXEC::set_value_t(int)>
    {
      return {};
    }

    auto get_env() const noexcept -> STDEXEC::env<>
    {
      return {};
    }

    template <STDEXEC::receiver Receiver>
    struct operation
    {
      using operation_state_concept = STDEXEC::operation_state_tag;

      Receiver receiver_;

      void start() & noexcept
      {
        STDEXEC::set_value(std::move(receiver_), 42);
      }
    };

    template <STDEXEC::receiver Receiver>
    auto connect(Receiver receiver) & noexcept -> operation<Receiver>
    {
      return {std::move(receiver)};
    }
  };

  TEST_CASE("libdispatch queue should be able to process tasks")
  {
    exec::libdispatch_queue queue;
    auto                    sch = queue.get_scheduler();

    std::vector<int> data{1, 2, 3, 4, 5};
    auto             add = [](auto const &data)
    {
      return std::accumulate(std::begin(data), std::end(data), 0);
    };
    auto sender = STDEXEC::just(std::move(data)) | STDEXEC::continues_on(sch) | STDEXEC::then(add);

    auto completion_scheduler = STDEXEC::get_completion_scheduler<STDEXEC::set_value_t>(
      STDEXEC::get_env(sender));

    CHECK(completion_scheduler == sch);
    auto [res] = STDEXEC::sync_wait(sender).value();
    CHECK(res == 15);
  }

  TEST_CASE("libdispatch queue bulk algorithm should call callback function with all allowed "
            "shapes")
  {
    exec::libdispatch_queue queue;
    auto                    sch = queue.get_scheduler();

    std::vector<int> data{1, 2, 3, 4, 5};
    auto             size                  = data.size();
    auto             expensive_computation = [](auto i, auto &data)
    {
      data[i] = 2 * data[i];
    };
    auto add = [](auto const &data)
    {
      return std::accumulate(std::begin(data), std::end(data), 0);
    };
    auto sender = STDEXEC::just(std::move(data)) | STDEXEC::continues_on(sch)
                | STDEXEC::bulk(STDEXEC::par, size, expensive_computation) | STDEXEC::then(add);

    auto completion_scheduler = STDEXEC::get_completion_scheduler<STDEXEC::set_value_t>(
      STDEXEC::get_env(sender));

    CHECK(completion_scheduler == sch);
    auto [res] = STDEXEC::sync_wait(sender).value();
    CHECK(res == 30);
  }

  TEST_CASE("libdispatch queue bulk_chunked uses one task per index for parallel policies")
  {
    exec::libdispatch_queue queue;
    auto                    sch = queue.get_scheduler();

    std::vector<int> visited(5, 0);
    std::atomic<int> chunks{0};

    auto sender = STDEXEC::schedule(sch)
                | STDEXEC::bulk_chunked(STDEXEC::par,
                                        5,
                                        [&](int begin, int end)
                                        {
                                          ++chunks;
                                          for (; begin != end; ++begin)
                                            visited[begin] = 1;
                                        });

    REQUIRE(STDEXEC::sync_wait(std::move(sender)).has_value());

    CHECK(chunks.load() == 5);
    CHECK(visited == std::vector<int>{1, 1, 1, 1, 1});
  }

  TEST_CASE("libdispatch queue runs a non-parallel bulk_chunked in a single task")
  {
    exec::libdispatch_queue queue;
    auto                    sch = queue.get_scheduler();

    std::vector<int> bounds;

    auto sender = STDEXEC::schedule(sch)
                | STDEXEC::bulk_chunked(STDEXEC::seq,
                                        5,
                                        [&](int begin, int end)
                                        {
                                          bounds.push_back(begin);
                                          bounds.push_back(end);
                                        });

    REQUIRE(STDEXEC::sync_wait(std::move(sender)).has_value());

    // `seq` forbids splitting the index space, so a single chunk covers all of it
    CHECK(bounds == std::vector<int>{0, 5});
  }

  TEST_CASE("libdispatch queue bulk_unchunked should call callback function with every index")
  {
    exec::libdispatch_queue queue;
    auto                    sch = queue.get_scheduler();

    std::vector<int> data{1, 2, 3, 4, 5};
    auto             size                  = data.size();
    auto             expensive_computation = [](auto i, auto &data)
    {
      data[i] = 2 * data[i];
    };
    auto add = [](auto const &data)
    {
      return std::accumulate(std::begin(data), std::end(data), 0);
    };
    auto sender = STDEXEC::just(std::move(data)) | STDEXEC::continues_on(sch)
                | STDEXEC::bulk_unchunked(STDEXEC::par, size, expensive_computation)
                | STDEXEC::then(add);

    auto [res] = STDEXEC::sync_wait(sender).value();
    CHECK(res == 30);
  }

  TEST_CASE("libdispatch queue runs a non-parallel bulk_unchunked in a single task")
  {
    exec::libdispatch_queue queue;
    auto                    sch = queue.get_scheduler();

    std::vector<int> indices;

    auto sender = STDEXEC::schedule(sch)
                | STDEXEC::bulk_unchunked(STDEXEC::seq,
                                          4,
                                          [&](int idx) { indices.push_back(idx); });

    REQUIRE(STDEXEC::sync_wait(std::move(sender)).has_value());

    CHECK(indices == std::vector<int>{0, 1, 2, 3});
  }

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
  TEST_CASE("libdispatch bulk should handle exceptions gracefully")
  {
    exec::libdispatch_queue queue;
    auto                    sch = queue.get_scheduler();

    std::vector<int> data{1, 2, 3, 4, 5};
    auto             size                  = data.size();
    auto             expensive_computation = [](auto i, auto data)
    {
      if (i == 0)
        throw 999;
      return 2 * data[i];
    };
    auto add = [](auto const &data)
    {
      return std::accumulate(std::begin(data), std::end(data), 0);
    };
    auto sender = STDEXEC::just(std::move(data)) | STDEXEC::continues_on(sch)
                | STDEXEC::bulk(STDEXEC::par, size, expensive_computation) | STDEXEC::then(add);

    STDEXEC_TRY
    {
      STDEXEC::sync_wait(sender);
      CHECK(false);
    }
    STDEXEC_CATCH(int e)
    {
      CHECK(e == 999);
    }
    STDEXEC_CATCH_ALL
    {
      FAIL("invalid exception caught");
    }
  }

  TEST_CASE("libdispatch bulk stops after value capture fails")
  {
    struct value_capture_error
    {};

    struct throwing_value
    {
      throwing_value() = default;

      throwing_value(throwing_value const &)
      {
        throw value_capture_error{};
      }

      throwing_value(throwing_value &&)
      {
        throw value_capture_error{};
      }
    };

    exec::libdispatch_queue queue;
    auto                    sch = queue.get_scheduler();

    auto sender = STDEXEC::schedule(sch) | STDEXEC::then([]() noexcept { return throwing_value{}; })
                | STDEXEC::bulk(STDEXEC::par, 0, [](int, throwing_value &) noexcept {});

    STATIC_REQUIRE(
      set_equivalent<STDEXEC::completion_signatures_of_t<decltype(sender), STDEXEC::env<>>,
                     STDEXEC::completion_signatures<STDEXEC::set_value_t(throwing_value),
                                                    STDEXEC::set_error_t(std::exception_ptr),
                                                    STDEXEC::set_stopped_t()>>);

    STDEXEC_TRY
    {
      STDEXEC::sync_wait(std::move(sender));
      CHECK(false);
    }
    STDEXEC_CATCH(value_capture_error const &)
    {
    }
    STDEXEC_CATCH_ALL
    {
      FAIL("invalid exception caught");
    }
  }

#endif

  TEST_CASE("libdispatch bulk preserves lvalue-reference value categories")
  {
    struct lvalue_value
    {
      lvalue_value()
        : value(0)
      {}

      explicit lvalue_value(int value)
        : value(value)
      {}

      lvalue_value(lvalue_value const &) noexcept = default;

      lvalue_value(lvalue_value &&other) noexcept(false)
        : value(other.value)
      {
        other.moved_from = true;
      }

      int  value;
      bool moved_from = false;
    } value{42};

    exec::libdispatch_queue queue;
    auto                    sch  = queue.get_scheduler();
    int                     seen = 0;

    auto sender = STDEXEC::schedule(sch)
                | STDEXEC::then([&]() noexcept -> lvalue_value & { return value; })
                | STDEXEC::bulk(STDEXEC::par,
                                1,
                                [&](int, lvalue_value &item) noexcept { seen = item.value; });

    STATIC_REQUIRE(
      set_equivalent<STDEXEC::completion_signatures_of_t<decltype(sender), STDEXEC::env<>>,
                     STDEXEC::completion_signatures<STDEXEC::set_value_t(lvalue_value),
                                                    STDEXEC::set_stopped_t()>>);

    auto result = STDEXEC::sync_wait(std::move(sender));

    REQUIRE(result.has_value());
    CHECK(seen == 42);
    CHECK_FALSE(value.moved_from);
  }

  TEST_CASE("libdispatch bulk connects an lvalue child sender as an lvalue")
  {
    exec::libdispatch_queue queue;
    auto                    fun = [](int, int, int &) noexcept {};
    using sender_t =
      exec::__libdispatch::bulk_sender<lvalue_connect_sender, int, decltype(fun), true>;

    sender_t sender{queue, lvalue_connect_sender{}, 0, std::move(fun), true};
    auto     result = STDEXEC::sync_wait(sender);

    REQUIRE(result.has_value());
    CHECK(std::get<0>(*result) == 42);
  }
}  // namespace
