#include <iostream>
#include <boost/asio.hpp>
#include <exec/static_thread_pool.hpp>
#include <exec/when_any.hpp>
#include "../use_sender.hpp" // <asioexec/use_sender.hpp>

exec::static_thread_pool thread_pool = exec::static_thread_pool(1); // A global execution_context. In order to make everything simple, we only create 1 thread.

int main()
{
    auto socket = boost::asio::ip::tcp::socket(boost::asio::system_executor()); // Create a socket.
    auto connect_task = stdexec::schedule(thread_pool.get_scheduler()) // The connection task starts on thread_pool.
                      | stdexec::then([]
                          {
                              return boost::asio::ip::tcp::resolver(boost::asio::system_executor()); // Create a resolver.
                          })
                      | stdexec::let_value([] (auto&& resolver)
                          {
                              return resolver.async_resolve("www.google.com", "80", asioexec::use_sender); // Async resolve. This thread will then do other things, until the system notify that the resolve-task is done (system will later call schedule(thread_pool) | stdexec::then(...)).
                          })
                      | stdexec::let_value([&] (auto&& ec, auto&& resolve_results)
                          {
                              if (ec) throw boost::system::system_error(ec); // This scope will still be executed on thread_pool.
                              return socket.async_connect(*resolve_results.begin(), asioexec::use_sender); // Async resolve. This thread will then do other things, until the system notify that the connect-task is done (system will later call schedule(thread_pool) | stdexec::then(...)).
                          })
                      | stdexec::then([] (auto&& ec)
                          {
                              if (ec) throw boost::system::system_error(ec);
                              std::cout << "Connected!\n"; // OK! Your socket is connected.
                          });

    auto timer = boost::asio::steady_timer(boost::asio::system_executor(), std::chrono::milliseconds(10));
    auto expiry_task = stdexec::schedule(thread_pool.get_scheduler())
                     | stdexec::let_value([&]
                         {
                             return timer.async_wait(asioexec::use_sender);
                         })
                     | stdexec::let_value([] (auto&& ec) 
                         {
                            if (ec) throw boost::system::system_error(ec);
                             std::cout << "Time Expired!\n"; 
                             return stdexec::just_stopped(); // Cancel the connection.
                         });

    stdexec::sync_wait(exec::when_any(connect_task, expiry_task));

    // socket.read(); socket.write(); ...
}

/* Expected Output

   (If connected to www.google.com:80 in 10 ms) Connected!
   (else, after 10 ms)                          Time Expired!

*/