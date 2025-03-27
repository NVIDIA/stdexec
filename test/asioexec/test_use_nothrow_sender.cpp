#include <catch2/catch.hpp>

#include <asioexec/use_sender.hpp>

#include <boost/asio.hpp>

namespace {
    TEST_CASE("asioexec::use_nothrow_sender general test")
    {
        // Create 2 timers.
        auto timer_0 = boost::asio::steady_timer(boost::asio::system_executor(), std::chrono::seconds(65536));
        auto timer_1 = boost::asio::steady_timer(boost::asio::system_executor(), std::chrono::seconds(1)); 
        
        // Compose tasks as sender.
        auto task_0 = timer_0.async_wait(asioexec::use_nothrow_sender) // use_nothrow_sender will return the error_code. User maybe prefer to handle this partial success.
                    | stdexec::then([] (boost::system::error_code ec) { CHECK(ec == boost::asio::error::operation_aborted); }); // Check the explicit error_code.
        auto task_1 = timer_1.async_wait(asioexec::use_sender)
                    | stdexec::then([&] { timer_0.cancel(); }); // In order to trigger an error, which is expected to be boost::asio::operation_aborted.

        // Parallel launch the 2 tasks.
        stdexec::sync_wait(stdexec::when_all(task_0, task_1));
    }
} // namespace 
