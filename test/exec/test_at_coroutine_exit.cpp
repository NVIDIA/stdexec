#include <exec/at_coroutine_exit.hpp>
#include <catch2/catch.hpp>

#include <test_common/schedulers.hpp>

#ifdef __linux__
#include <unistd.h>
#include <sys/wait.h>

namespace {
template<class F, class... Args>
void REQUIRE_TERMINATE(F&& f, Args&&... args)
{
  //spawn a new process
  auto child_pid = ::fork();

  //if the fork succeed
  if (child_pid >= 0){

    //if we are in the child process
    if (child_pid == 0){

        //call the function that we expect to abort
        std::set_terminate([]{ std::exit(EXIT_FAILURE); });

        std::invoke((F&&) f, (Args&&) args...);

        //if the function didn't abort, we'll exit cleanly
        std::exit(EXIT_SUCCESS);
    }
  }

  //determine if the child process aborted
  int exit_status{};
  ::wait(&exit_status);

  // we check the exit status instead of a signal interrupt, because
  // Catch is going to catch the signal and exit with an error
  bool aborted = WEXITSTATUS(exit_status);

  REQUIRE(aborted);
}
}
#endif

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

exec::task<void> two_exit_task_actions_as_sender(int& result) {
  ++result;
  co_await exec::at_coroutine_exit([](int& result) {
    return stdexec::just() | stdexec::then([&result] { result += 1; }); 
  }, result);
  co_await exec::at_coroutine_exit([](int& result) {
    return stdexec::just() | stdexec::then([&result] { result *= 2; }); 
  }, result);
  ++result;
}
}

TEST_CASE("at_coroutine_exit invokes two actions in correct order", "[task][at_coroutine_exit]")
{
  int result = 0;
  stdexec::sync_wait(two_exit_task_actions(result));
  REQUIRE(result == 5);
  result = 0;
  stdexec::sync_wait(two_exit_task_actions_as_sender(result));
  REQUIRE(result == 5);
}

namespace {
struct test_exception : std::runtime_error {
  using std::runtime_error::runtime_error;
};

exec::task<void> invoke_action_after_exception(int& result) {
  ++result;
  co_await exec::at_coroutine_exit([](int& result) -> exec::task<void> {
    result += 1;
    co_return;
  }, result);
  co_await exec::at_coroutine_exit([](int& result) -> exec::task<void> {
    result *= 2;
    co_return;
  }, result);
  throw test_exception("test");
  ++result;
}
}

TEST_CASE("at_coroutine_exit invokes two actions in correct order after exception", "[task][at_coroutine_exit]")
{
  int result = 0;
  REQUIRE_THROWS_AS(stdexec::sync_wait(invoke_action_after_exception(result)), test_exception);
  REQUIRE(result == 3);
}

namespace {
exec::task<void> invoke_actions_after_stop(int& result, auto stop) {
  ++result;
  co_await exec::at_coroutine_exit([](int& result) -> exec::task<void> {
    result += 1;
    co_return;
  }, result);
  co_await exec::at_coroutine_exit([](int& result) -> exec::task<void> {
    result *= 2;
    co_return;
  }, result);
  co_await stop;
  ++result;
}
}

TEST_CASE("at_coroutine_exit invokes two actions in correct order after stop signal", "[task][at_coroutine_exit]")
{
  int result = 0;
  stopped_scheduler scheduler;
  stdexec::sync_wait(invoke_actions_after_stop(result, stdexec::schedule(scheduler)));
  REQUIRE(result == 3);
}


#ifdef __linux__
namespace {
exec::task<void> exception_in_action() {
  co_await exec::at_coroutine_exit([]() -> exec::task<void> {
    throw test_exception("test");
    co_return;
  });
}
}

TEST_CASE("at_coroutine_exit terminates after exception within action", "[task][at_coroutine_exit]")
{
  REQUIRE_TERMINATE([] { stdexec::sync_wait(exception_in_action()); });
}

namespace {
exec::task<void> stop_in_action() {
  co_await exec::at_coroutine_exit([] { return stdexec::just_stopped(); });
}
}

TEST_CASE("at_coroutine_exit terminates after stop within action", "[task][at_coroutine_exit]")
{
  REQUIRE_TERMINATE([] { stdexec::sync_wait(stop_in_action()); });
}
#endif