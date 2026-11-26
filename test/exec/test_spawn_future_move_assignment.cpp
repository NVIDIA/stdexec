/*
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 */

#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <exec/async_scope.hpp>
#include <optional>
#include <stdexec/execution.hpp>
#include <string_view>
#include <utility>

namespace regression_spawn_future_move_assignment
{
  // Public APIs only, unmodified production headers. Both futures have already
  // started their schedule operations, which are still queued in the run_loop.

  int run_case(int argc, char** argv)
  {
    auto              mode = argc > 1 ? std::string_view(argv[1]) : std::string_view("assign");
    exec::async_scope scope;
    stdexec::run_loop loop;
    auto              task = stdexec::schedule(loop.get_scheduler());
    using future           = decltype(scope.spawn_future(task));
    std::optional<future> first{scope.spawn_future(task)};
    auto                  second = scope.spawn_future(task);
    if (mode == "completed")
    {
      loop.finish();
      loop.run();
    }
    if (mode == "drop")
      first.reset();
    else
      *first = std::move(second);
    std::fputs("future replacement/drop returned; draining queued operations\n", stderr);
    if (mode != "completed")
    {
      loop.finish();
      loop.run();
    }
    stdexec::sync_wait(scope.on_empty());
    std::printf("PASS mode=%.*s scope drained\n", int(mode.size()), mode.data());
    return 0;
  }

}  // namespace regression_spawn_future_move_assignment

TEST_CASE("spawn_future_move_assignment: assign", "[regression][async-scope-future]")
{
  char  program[] = "regression";
  char  arg0[]    = "assign";
  char* argv[]    = {program, arg0};
  REQUIRE(regression_spawn_future_move_assignment::run_case(2, argv) == 0);
}

TEST_CASE("spawn_future_move_assignment: drop", "[regression][async-scope-future]")
{
  char  program[] = "regression";
  char  arg0[]    = "drop";
  char* argv[]    = {program, arg0};
  REQUIRE(regression_spawn_future_move_assignment::run_case(2, argv) == 0);
}

TEST_CASE("spawn_future_move_assignment: completed", "[regression][async-scope-future]")
{
  char  program[] = "regression";
  char  arg0[]    = "completed";
  char* argv[]    = {program, arg0};
  REQUIRE(regression_spawn_future_move_assignment::run_case(2, argv) == 0);
}
