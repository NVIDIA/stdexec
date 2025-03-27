#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <exec/when_any.hpp>
#include <asioexec/use_sender.hpp>

#include <boost/asio.hpp>

namespace {
    TEST_CASE("asioexec::use_sender using an boost::asio cancellation")
    {
        // A looooooong timer, and a signal_set to catch SIGINT.
        auto timer = boost::asio::steady_timer(boost::asio::system_executor(), std::chrono::seconds(65536)); 
        auto signals = boost::asio::signal_set(boost::asio::system_executor());
        signals.add(SIGINT);

        // Compose tasks as sender.
        auto task_0 = timer.async_wait(asioexec::use_sender); // The main task, which is loooong.
        auto task_1 = signals.async_wait(asioexec::use_sender)
                    | stdexec::let_value([] (auto&&...) { return stdexec::just_stopped(); }); // The signal_set listening for a SIGINT. Once catched, it will set_stopped() the task.

        // You can press Ctrl+C to trigger SIGINT.
        // Here we use a raw thread to simulate "the user press Ctrl+C" after 1 second.
        auto interruptor = std::thread([] { std::this_thread::sleep_for(std::chrono::seconds(1)); std::raise(SIGINT); });
        interruptor.detach();

        // Parallel launch the 2 tasks.
        // When the second task catches a SIGINT, it will demand the total task to stop (in stdexec's recommended way),
        // meanwhile **use an cancellation_slot to cancel the asio timer (in boost::asio's recommended way)**.
        auto task = exec::when_any(task_0, task_1)
                  | stdexec::into_variant();
        stdexec::sync_wait(task);

        // Check: the timer has been cancelled using a boost::asio method, so "the number of asynchronous operations that were cancelled (in this scope)" == 0.
        CHECK(timer.cancel() == 0);
    }
} // namespace 