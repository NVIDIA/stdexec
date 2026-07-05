#include "catch2/catch_all.hpp"
#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

#include <exception>
#include <mutex>
#include <ranges>
#include <stdexcept>
#include <thread>
#include <unordered_set>
namespace ex = STDEXEC;

namespace
{
#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
  struct throwing_set_next_receiver
  {
    using receiver_concept = ex::receiver_tag;

    bool&               set_value_called_;
    bool&               set_stopped_called_;
    std::exception_ptr& error_;

    template <class Item>
    auto set_next(Item&&) -> decltype(ex::just())
    {
      throw std::runtime_error{"set_next failed"};
    }

    void set_value() noexcept
    {
      set_value_called_ = true;
    }

    void set_stopped() noexcept
    {
      set_stopped_called_ = true;
    }

    void set_error(std::exception_ptr error) noexcept
    {
      error_ = error;
    }

    auto get_env() const noexcept -> ex::env<>
    {
      return {};
    }
  };
#endif
}  // namespace

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

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
TEST_CASE("schedule_all on static_thread_pool sends errors from set_next",
          "[types][static_thread_pool]")
{
  exec::static_thread_pool pool{1};
  bool                     set_value_called   = false;
  bool                     set_stopped_called = false;
  std::exception_ptr       error;

  auto op =
    exec::subscribe(exec::schedule_all(pool, std::views::iota(0, 1)),
                    throwing_set_next_receiver{set_value_called, set_stopped_called, error});

  ex::start(op);

  CHECK_FALSE(set_value_called);
  CHECK_FALSE(set_stopped_called);
  REQUIRE(error);
  CHECK_THROWS_AS(std::rethrow_exception(error), std::runtime_error);
}
#endif

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
