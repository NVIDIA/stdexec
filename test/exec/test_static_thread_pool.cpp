#include "catch2/catch_all.hpp"
#include "exec/sequence/ignore_all_values.hpp"
#include "exec/sequence/transform_each.hpp"
#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

#include <atomic>
#include <mutex>
#include <ranges>
#include <thread>
#include <unordered_set>
namespace ex = STDEXEC;

TEST_CASE("static_thread_pool::get_scheduler_on_thread Test start on a specific thread",
          "[types][static_thread_pool]")
{
  constexpr size_t const   num_of_threads = 5;
  exec::static_thread_pool pool{num_of_threads};

  std::unordered_set<std::thread::id> thread_ids;
  for (size_t i = 0; i < num_of_threads; ++i)
  {
    auto sender = ex::schedule(pool.get_scheduler_on_thread(i))
                | ex::then([&]() -> void { thread_ids.insert(std::this_thread::get_id()); });
    ex::sync_wait(std::move(sender));
  }
  REQUIRE(thread_ids.size() == num_of_threads);
}

TEST_CASE("bulk on static_thread_pool executes on multiple threads", "[types][static_thread_pool]")
{
  constexpr size_t const   num_of_threads = 5;
  exec::static_thread_pool pool{num_of_threads};

  std::mutex                          mtx;
  std::unordered_set<std::thread::id> thread_ids;
  auto                                sender = ex::starts_on(pool.get_scheduler(),
                              ex::just()
                                | ex::bulk(ex::par_unseq,
                                           num_of_threads,
                                           [&](size_t) -> void
                                           {
                                             std::this_thread::sleep_for(
                                               std::chrono::milliseconds(100));
                                             std::lock_guard lock(mtx);
                                             thread_ids.insert(std::this_thread::get_id());
                                           }));
  ex::sync_wait(std::move(sender));
  REQUIRE(thread_ids.size() == num_of_threads);
}

TEST_CASE("schedule_all on static_thread_pool handles ranges smaller than available parallelism",
          "[types][static_thread_pool]")
{
  constexpr size_t const num_of_threads = 5;
  constexpr int const    range_size     = 3;

  exec::static_thread_pool pool{num_of_threads};
  REQUIRE(range_size < pool.available_parallelism());

  std::atomic<int> count{0};
  std::atomic<int> sum{0};
  auto             sender =
    exec::schedule_all(pool, std::views::iota(0, range_size))
    | exec::transform_each(ex::then(
      [&](int x) noexcept
      {
        count.fetch_add(1, std::memory_order_relaxed);
        sum.fetch_add(x, std::memory_order_relaxed);
      }))
    | exec::ignore_all_values();

  CHECK(ex::sync_wait(std::move(sender)));
  CHECK(count.load(std::memory_order_relaxed) == range_size);
  CHECK(sum.load(std::memory_order_relaxed) == 3);
}

TEST_CASE("bulk on static_thread_pool executes on multiple threads, take 2",
          "[types][static_thread_pool]")
{
  constexpr size_t const   num_of_threads = 5;
  exec::static_thread_pool pool{num_of_threads};

  std::mutex                          mtx;
  std::unordered_set<std::thread::id> thread_ids;
  auto                                sender = ex::schedule(pool.get_scheduler())
              | ex::bulk(ex::par_unseq,
                         num_of_threads,
                         [&](size_t) -> void
                         {
                           std::this_thread::sleep_for(std::chrono::milliseconds(100));
                           std::lock_guard lock(mtx);
                           thread_ids.insert(std::this_thread::get_id());
                         });
  ex::sync_wait(std::move(sender));
  REQUIRE(thread_ids.size() == num_of_threads);
}
