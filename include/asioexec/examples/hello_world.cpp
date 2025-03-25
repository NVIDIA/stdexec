#include <iostream>
#include <boost/asio.hpp>
#include "../use_sender.hpp" // <asioexec/use_sender.hpp>

int main()
{
    std::cout << "(process started)\n";

    auto timer_0 = boost::asio::system_timer(boost::asio::system_executor(), std::chrono::seconds(3)); // Create a timer.
    auto timer_1 = boost::asio::system_timer(boost::asio::system_executor(), std::chrono::seconds(5)); // Create a timer.
    
    auto task_0 = stdexec::just() // Starts on main thread.
                | stdexec::let_value([&] { return timer_0.async_wait(asioexec::use_sender); }) // Await the timer...
                | stdexec::then([] (auto&&...) { std::cout << "hello world after 3 seconds!\n"; }); // ... and this scope will execute after 3 seconds on main thread.

    auto task_1 = stdexec::just() // Starts on main thread.
                | stdexec::let_value([&] { return timer_1.async_wait(asioexec::use_sender); }) // Await the timer...
                | stdexec::then([] (auto&&...) { std::cout << "hello world after 5 seconds!\n"; }); // ... and this scope will execute after 5 seconds on main thread.

    stdexec::sync_wait(stdexec::when_all(task_0, task_1)); // Concurrently start the 2 tasks, which totally costs 5s (rather than 8s).

    std::cout << "(process exit)\n";
}

/* Expected output
   + - time - + ---------- output ---------- +
   | now      | (process started)            |
   | now + 3s | hello world after 3 seconds! |
   | now + 5s | hello world after 5 seconds! |
   | now + 5s | (process exit)               |
   + -------- + ---------------------------- +
*/