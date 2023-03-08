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
  std::thread timer_thread{[&] {
    for (int i = 0; i < 6; ++i) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      std::cout << "Waking up the io_uring thread #" << i << "...\n";
      context.wakeup();
    }
    context.request_stop();
  }};
  auto scheduler = context.get_scheduler();
  using namespace std::chrono_literals;
  stdexec::sync_wait(
    exec::schedule_after(scheduler, 3s) //
    | stdexec::then([] {                //
        std::cout << "Hello, world!\n"; //
      }));
  io_thread.join();
  timer_thread.join();
}