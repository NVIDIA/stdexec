#include "catch2/catch.hpp"
#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

#include <mutex>
#include <thread>
#include <unordered_set>
namespace ex = STDEXEC;

TEST_CASE(
  "static_thread_pool::get_scheduler_on_thread Test start on a specific thread",
  "[types][static_thread_pool]") {
  constexpr const size_t num_of_threads = 5;
  exec::static_thread_pool pool{num_of_threads};

  std::unordered_set<std::thread::id> thread_ids;
  for (size_t i = 0; i < num_of_threads; ++i) {
    auto sender = ex::schedule(pool.get_scheduler_on_thread(i))
                | ex::then([&]() -> void { thread_ids.insert(std::this_thread::get_id()); });
    ex::sync_wait(std::move(sender));
  }
  REQUIRE(thread_ids.size() == num_of_threads);
}

TEST_CASE(
  "bulk on static_thread_pool executes on multiple threads",
  "[types][static_thread_pool]") {
  constexpr const size_t num_of_threads = 5;
  exec::static_thread_pool pool{num_of_threads};

  std::mutex mtx;
  std::unordered_set<std::thread::id> thread_ids;
  auto sender = ex::starts_on(
    pool.get_scheduler(), ex::just() | ex::bulk(ex::par_unseq, num_of_threads, [&](size_t) -> void {
                            std::this_thread::sleep_for(std::chrono::milliseconds(100));
                            std::lock_guard lock(mtx);
                            thread_ids.insert(std::this_thread::get_id());
                          }));
  ex::sync_wait(std::move(sender));
  REQUIRE(thread_ids.size() == num_of_threads);
}

TEST_CASE(
  "bulk on static_thread_pool executes on multiple threads, take 2",
  "[types][static_thread_pool]") {
  constexpr const size_t num_of_threads = 5;
  exec::static_thread_pool pool{num_of_threads};

  std::mutex mtx;
  std::unordered_set<std::thread::id> thread_ids;
  auto sender = ex::schedule(pool.get_scheduler())
              | ex::bulk(ex::par_unseq, num_of_threads, [&](size_t) -> void {
                  std::this_thread::sleep_for(std::chrono::milliseconds(100));
                  std::lock_guard lock(mtx);
                  thread_ids.insert(std::this_thread::get_id());
                });
  ex::sync_wait(std::move(sender));
  REQUIRE(thread_ids.size() == num_of_threads);
}
