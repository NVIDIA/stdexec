#include "exec/linux/io_uring_context.hpp"

#include "exec/when_any.hpp"
#include "exec/finally.hpp"

#include "stdexec/execution.hpp"

#include <chrono>
#include <thread>

#include <iostream>

int main() {
  exec::io_uring_context context;
  exec::io_uring_context context2;
  std::thread io_thread{[&] {
    context.run();
  }};
  std::thread io_thread2{[&] {
    context2.run();
  }};
  auto scheduler = context.get_scheduler();
  auto scheduler2 = context2.get_scheduler();
  using namespace std::chrono_literals;

  stdexec::sync_wait(exec::when_any(
    exec::schedule_after(scheduler, 1s) | stdexec::then([] { std::cout << "Hello, 1!\n"; }),
    exec::schedule_after(scheduler2, 2s) | stdexec::then([] { std::cout << "Hello, 2!\n"; })
      | stdexec::upon_stopped([] { std::cout << "Hello, 2, stopped.\n"; })));

  stdexec::sync_wait(exec::when_any(
    exec::schedule_after(scheduler, 1s) | stdexec::then([] { std::cout << "Hello, 1!\n"; })
      | stdexec::upon_stopped([] { std::cout << "Hello, 1, stopped.\n"; }),
    exec::schedule_after(scheduler2, 500ms) | stdexec::then([] { std::cout << "Hello, 2!\n"; })
      | stdexec::upon_stopped([] { std::cout << "Hello, 2, stopped.\n"; })));

  stdexec::sync_wait(stdexec::when_all(
    stdexec::schedule(scheduler) | stdexec::then([] { std::cout << "Hello, 0!\n"; }),
    exec::schedule_after(scheduler, 1s) | stdexec::then([] { std::cout << "Hello, 1!\n"; }),
    exec::schedule_after(scheduler2, 2s) | stdexec::then([] { std::cout << "Hello, 2!\n"; }),
    exec::schedule_after(scheduler, 3s) | stdexec::then([] { std::cout << "Stop it!\n"; }),
    exec::schedule_after(scheduler2, 4s) | stdexec::then([&] { context.request_stop(); }),
    exec::finally(
      exec::schedule_after(scheduler, 4s),
      stdexec::just() | stdexec::then([&] { context2.request_stop(); })),
    exec::schedule_after(scheduler, 10s)    //
      | stdexec::then([] {                  //
          std::cout << "Hello, world!\n";   //
        })
      | stdexec::upon_stopped([] {          //
          std::cout << "Hello, stopped.\n"; //
        }),
    exec::schedule_after(scheduler2, 10s)   //
      | stdexec::then([] {                  //
          std::cout << "Hello, world!\n";   //
        })
      | stdexec::upon_stopped([] {          //
          std::cout << "Hello, stopped.\n"; //
        })));
  io_thread.join();
  io_thread2.join();

  io_thread = std::thread{[&] {
    context.run();
  }};

  stdexec::sync_wait(exec::when_any(
    exec::schedule_after(scheduler, 1s) | stdexec::then([] { std::cout << "Hello, 1!\n"; }),
    exec::schedule_after(scheduler, 500ms) | stdexec::then([] { std::cout << "Hello, 2!\n"; })));

  context.request_stop();
  io_thread.join();
}