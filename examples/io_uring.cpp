#include "stdexec/execution.hpp"

#include "exec/linux/io_uring_context.hpp"

#include <chrono>
#include <thread>

#include <iostream>

int main() {
  exec::io_uring_context context(128);
  std::thread io_thread{[&] {
    context.run();
  }};
  auto scheduler = context.get_scheduler();
  using namespace std::chrono_literals;
  stdexec::sync_wait(stdexec::when_all(
    exec::schedule_after(scheduler, 1s) | stdexec::then([] { std::cout << "Hello, 1!\n"; }),
    exec::schedule_after(scheduler, 2s) | stdexec::then([] { std::cout << "Hello, 2!\n"; }),
    exec::schedule_after(scheduler, 3s) | stdexec::then([] { std::cout << "Stop it!\n"; }),
    exec::schedule_after(scheduler, 4s) | stdexec::then([&] { context.request_stop(); }),
    exec::schedule_after(scheduler, 10s)    //
      | stdexec::then([] {                  //
          std::cout << "Hello, world!\n";   //
        })
      | stdexec::upon_stopped([] {          //
          std::cout << "Hello, stopped.\n"; //
        })));
  io_thread.join();
}