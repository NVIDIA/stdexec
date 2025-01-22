#include "catch2/catch.hpp"
#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

#include <thread>
#include <unordered_set>
namespace ex = stdexec;

TEST_CASE(
  "static_thread_pool::get_scheduler_on_thread Test start on a specific thread",
  "[types][static_thread_pool]") {
  constexpr const size_t num_of_threads = 5;
  exec::static_thread_pool pool{num_of_threads};

  std::unordered_set<std::thread::id> thread_ids;
  for (size_t i = 0; i < num_of_threads; ++i) {
    auto sender = ex::schedule(pool.get_scheduler_on_thread(i))
                | ex::then([&] { thread_ids.insert(std::this_thread::get_id()); });
    ex::sync_wait(std::move(sender));
  }
  REQUIRE(thread_ids.size() == num_of_threads);
}
