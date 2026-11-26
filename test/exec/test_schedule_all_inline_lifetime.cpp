/*
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 */

#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <exec/sequence_senders.hpp>
#include <exec/static_thread_pool.hpp>
#include <ranges>
#include <stdexec/execution.hpp>

namespace regression_schedule_all_inline_lifetime
{

  struct control
  {
    void* operation{};
    void (*destroy)(void*){};
    bool destroy_on_complete = true;
    bool completed           = false;
  };
  struct receiver
  {
    using receiver_concept = stdexec::receiver_tag;
    control* state;
    template <class Item>
    auto set_next(Item&&) noexcept
    {
      // Declining an item by returning just_stopped is also exercised by
      // the repository's sequence/iterate tests.
      return stdexec::just_stopped();
    }
    void finish() noexcept
    {
      auto* external      = state;
      external->completed = true;
      if (external->destroy_on_complete)
      {
        std::fputs("receiver completed; deleting operation state\n", stderr);
        external->destroy(external->operation);
        external->operation = nullptr;
      }
    }
    void set_value() noexcept
    {
      finish();
    }
    void set_stopped() noexcept
    {
      finish();
    }
    void set_error(std::exception_ptr) noexcept
    {
      std::terminate();
    }
  };
  int run_case(int argc, char**)
  {
    exec::static_thread_pool pool{1};
    control                  state;
    state.destroy_on_complete = argc == 1;
    auto sequence             = exec::schedule_all(pool, std::views::iota(0, 1));
    using operation           = exec::subscribe_result_t<decltype(sequence), receiver>;
    state.destroy             = [](void* p)
    {
      delete static_cast<operation*>(p);
    };
    auto* op        = new operation(exec::subscribe(std::move(sequence), receiver{&state}));
    state.operation = op;
    stdexec::start(*op);
    if (state.operation)
      state.destroy(state.operation);
    std::printf("start returned; completed=%d\n", state.completed);
    return state.completed ? 0 : 2;
  }
}  // namespace regression_schedule_all_inline_lifetime

TEST_CASE("schedule_all_inline_lifetime: default", "[regression][schedule-all-completion]")
{
  char  program[] = "regression";
  char* argv[]    = {program};
  REQUIRE(regression_schedule_all_inline_lifetime::run_case(1, argv) == 0);
}

TEST_CASE("schedule_all_inline_lifetime: keepalive", "[regression][schedule-all-completion]")
{
  char  program[] = "regression";
  char  arg0[]    = "keepalive";
  char* argv[]    = {program, arg0};
  REQUIRE(regression_schedule_all_inline_lifetime::run_case(2, argv) == 0);
}
