#include <catch2/catch_all.hpp>

#include <asioexec/use_sender.hpp>

#include <boost/asio.hpp>

namespace {
    TEST_CASE("asioexec::use_sender general test")
    {
        auto start = std::chrono::steady_clock::now();
        auto main_thread_id = std::this_thread::get_id();

        // Create 2 timers, one shorter and one longer.
        auto timer_0 = boost::asio::steady_timer(boost::asio::system_executor(), std::chrono::seconds(3));
        auto timer_1 = boost::asio::steady_timer(boost::asio::system_executor(), std::chrono::seconds(5));
        
        // Compose tasks as sender.
        auto task_0 = timer_0.async_wait(asioexec::use_sender)
                    | stdexec::then([&] { CHECK(std::this_thread::get_id() == main_thread_id); }); // This scope (or "callback") should still run on main thread.
        auto task_1 = timer_1.async_wait(asioexec::use_sender)
                    | stdexec::then([&] { CHECK(std::this_thread::get_id() == main_thread_id); });

         // Parallel launch the 2 tasks.
        stdexec::sync_wait(stdexec::when_all(task_0, task_1));

        // Check: the total task cost about 5 seconds (may fluctuate).
        auto finish = std::chrono::steady_clock::now();
        auto time_cost = finish - start;
        CHECK(time_cost >= std::chrono::milliseconds(4900));
        CHECK(time_cost <= std::chrono::milliseconds(5100));
    }
} // namespace 

