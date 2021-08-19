#include <iostream>
#include <tuple>
#include <variant>

// Pull in the reference implementation of P2300:
#include <execution.hpp>

#include "./inline_scheduler.hpp"

using namespace std::execution;

int main() {
  //scheduler auto sch = get_thread_pool().scheduler();                         // 1
  scheduler auto sch = inline_scheduler{};

  sender auto begin = schedule(sch);                                          // 2
  sender auto hi_again = lazy_then(begin, []{                                 // 3
    std::cout << "Hello world! Have an int.\n";                               // 3
    return 13;                                                                // 3
  });                                                                         // 3

  sender auto add_42 = lazy_then(hi_again, [](int arg) { return arg + 42; }); // 4

  auto [i] = std::this_thread::sync_wait(std::move(add_42)).value();          // 5
  std::cout << "Result: " << i << std::endl;
}
