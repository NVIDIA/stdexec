#include <iostream>
#include <asioexec/use_sender.hpp>
#include <exec/static_thread_pool.hpp>
#include <boost/asio.hpp>

exec::static_thread_pool thread_pool = exec::static_thread_pool(1);

/* Graph of tcp echo_server
   1 listener_thread listening the port,
       -> accept a connection,
       -> move ownership of the connection to the thread_pool (with n worker_thread),
           -> then the n worker_thread will be responsible for echo,
       -> once connection is not owned by listener_thread, the latter one will go back to listen for another connection.
*/

int main()
{
    int port = 12345; // Your target port.

    // clang-format off
    auto task = stdexec::just()
              | stdexec::then([&]
                  {
                      std::cout << "Listening port " << port << " on main-thread " << std::this_thread::get_id() << std::endl;
                      return boost::asio::ip::tcp::acceptor(boost::asio::system_executor(), boost::asio::ip::tcp::endpoint(boost::asio::ip::address_v4(), port)); 
                  })
              | stdexec::let_value([] (auto&& acceptor) 
                  {
                      return acceptor.async_accept(asioexec::use_sender)
                           | stdexec::then([] (auto&& socket)
                               {
                                   std::cout << "Accepted a connection on main-thread " << std::this_thread::get_id() << std::endl; // Excepted: thread_id == main-thread.
                                   return std::move(socket);
                               });
                  })
              | stdexec::then([] (auto&& socket)
                  {
                      auto echo_task = stdexec::starts_on(thread_pool.get_scheduler(), stdexec::just(std::move(socket), std::array<char,1024>()))
                                     | stdexec::let_value([] (auto&& socket, auto&& buffer)
                                         {
                                             return stdexec::just()
                                                  | stdexec::let_value([&] 
                                                      {
                                                          return socket.async_read_some(boost::asio::mutable_buffer(buffer.data(), buffer.size()), asioexec::use_sender); 
                                                      })
                                                  | stdexec::let_value([&] (size_t n)
                                                      {
                                                          return boost::asio::async_write(socket, boost::asio::const_buffer(buffer.data(), n), asioexec::use_sender);
                                                      })
                                                  | stdexec::then([] (size_t /*n*/)
                                                      {
                                                          std::cout << "Echo some message on thread-pool " << std::this_thread::get_id() << std::endl; // Expected: thread_id == exec::static_thread_pool.thread_id().
                                                      });
                                         });
                      stdexec::start_detached(std::move(echo_task));  
                  })
              | stdexec::upon_error([] (auto&& error) { try { std::rethrow_exception(error); } catch (const std::exception& e) { std::cout << "Error: " << e.what() << std::endl; } })
              | stdexec::upon_stopped([] { std::cout << "Stopped" << std::endl; });
    // clang-format on
    
    while(true)
        stdexec::sync_wait(task);
}

/* Expected output
   
----- This Process -----
Listening port 12345  on main-thread 1
Accepted a connection on main-thread 1
Listening port 12345  on main-thread 1
Echo some message     on thread-pool 2
...
------------------------

----- Another Terminal -----
>>> echo hello_world | nc localhost 12345
hello_world
>>> echo good_luck | nc localhost 12345
good_luck
----------------------------

*/