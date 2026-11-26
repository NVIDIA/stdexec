/*
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 */

#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <stdexec/execution.hpp>
#include <string_view>

namespace regression_spawn_future_reentrant_cancel
{
  // Source shape follows test/stdexec/algos/adaptors/test_spawn_future.cpp's
  // never sender, but connects/starts the future before requesting cancellation.

  struct control
  {
    bool  complete_inline = true;
    bool  stop_seen       = false;
    void* operation{};
    void (*complete)(void*){};
  };
  struct source
  {
    using sender_concept        = STDEXEC::sender_tag;
    using completion_signatures = STDEXEC::completion_signatures<STDEXEC::set_stopped_t()>;
    control* state;
    template <class R>
    struct operation
    {
      R        receiver;
      control* state;
      struct callback
      {
        operation* self;
        void       operator()() const noexcept
        {
          auto* saved             = self;
          saved->state->stop_seen = true;
          if (saved->state->complete_inline)
            saved->finish();
        }
      };
      using token = STDEXEC::stop_token_of_t<STDEXEC::env_of_t<R>>;
      using cb    = STDEXEC::stop_callback_for_t<token, callback>;
      std::optional<cb> registration;
      operation(R r, control* c)
        : receiver(std::move(r))
        , state(c)
      {}
      void finish() noexcept
      {
        registration.reset();
        STDEXEC::set_stopped(std::move(receiver));
      }
      void start() & noexcept
      {
        state->operation = this;
        state->complete  = [](void* p)
        {
          static_cast<operation*>(p)->finish();
        };
        registration.emplace(STDEXEC::get_stop_token(STDEXEC::get_env(receiver)), callback{this});
      }
    };
    template <class R>
    auto connect(R r) const
    {
      return operation<R>{std::move(r), state};
    }
  };
  struct receiver
  {
    using receiver_concept = STDEXEC::receiver_tag;
    STDEXEC::inplace_stop_token token;
    int*                        completed;
    auto                        get_env() const noexcept
    {
      return STDEXEC::prop{STDEXEC::get_stop_token, token};
    }
    void set_stopped() noexcept
    {
      ++*completed;
      std::fputs("future receiver stopped\n", stderr);
    }
    void set_error(std::exception_ptr) noexcept
    {
      std::abort();
    }
  };
  int run_case(int argc, char** argv)
  {
    auto mode = argc > 1 ? std::string_view(argv[1]) : std::string_view("inline");
    STDEXEC::simple_counting_scope scope;
    STDEXEC::inplace_stop_source   stop;
    control                        state;
    state.complete_inline = mode != "deferred";
    int  completed        = 0;
    auto future           = STDEXEC::spawn_future(source{&state}, scope.get_token());
    if (mode == "pre-stopped")
      stop.request_stop();
    auto op = STDEXEC::connect(std::move(future), receiver{stop.get_token(), &completed});
    STDEXEC::start(op);
    std::fputs("requesting consumer stop\n", stderr);
    stop.request_stop();
    if (mode == "deferred")
      state.complete(state.operation);
    STDEXEC::sync_wait(scope.join());
    std::printf("PASS mode=%.*s completions=%d source_stopped=%d\n",
                int(mode.size()),
                mode.data(),
                completed,
                state.stop_seen);
    return completed == 1 && state.stop_seen ? 0 : 2;
  }
}  // namespace regression_spawn_future_reentrant_cancel

TEST_CASE("spawn_future_reentrant_cancel: inline", "[regression][spawn-future-cancel]")
{
  char  program[] = "regression";
  char  arg0[]    = "inline";
  char* argv[]    = {program, arg0};
  REQUIRE(regression_spawn_future_reentrant_cancel::run_case(2, argv) == 0);
}

TEST_CASE("spawn_future_reentrant_cancel: deferred", "[regression][spawn-future-cancel]")
{
  char  program[] = "regression";
  char  arg0[]    = "deferred";
  char* argv[]    = {program, arg0};
  REQUIRE(regression_spawn_future_reentrant_cancel::run_case(2, argv) == 0);
}

TEST_CASE("spawn_future_reentrant_cancel: pre-stopped", "[regression][spawn-future-cancel]")
{
  char  program[] = "regression";
  char  arg0[]    = "pre-stopped";
  char* argv[]    = {program, arg0};
  REQUIRE(regression_spawn_future_reentrant_cancel::run_case(2, argv) == 0);
}
