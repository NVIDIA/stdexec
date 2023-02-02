#include <exec/at_coroutine_exit.hpp>
#include <catch2/catch.hpp>

namespace ex = stdexec;

namespace {
exec::task<void> one_exit_task_action(int& result) {
  ++result;
  co_await exec::at_coroutine_exit([](int& result) -> exec::task<void> {
    result *= 2;
    co_return;
  }, result);
  ++result;
}
}

TEST_CASE("at_coroutine_exit invokes at coroutine destruction", "[task][at_coroutine_exit]")
{
  int result = 0;
  stdexec::sync_wait(one_exit_task_action(result));
  REQUIRE(result == 4);
}

namespace {
exec::task<void> two_exit_task_actions(int& result) {
  ++result;
  co_await exec::at_coroutine_exit([](int& result) -> exec::task<void> {
    result += 1;
    co_return;
  }, result);
  co_await exec::at_coroutine_exit([](int& result) -> exec::task<void> {
    result *= 2;
    co_return;
  }, result);
  ++result;
}
}

TEST_CASE("at_coroutine_exit invokes in correct order", "[task][at_coroutine_exit]")
{
  int result = 0;
  stdexec::sync_wait(two_exit_task_actions(result));
  REQUIRE(result == 5);
}

