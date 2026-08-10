/*
 * Copyright (c) 2026 NVIDIA Corporation
 *
 * Licensed under the Apache License Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <exec/thread_pool_base.hpp>
#include <stdexec/execution.hpp>
#include <test_common/catch2.hpp>

#include <cstdint>
#include <exception>

namespace ex = STDEXEC;

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
namespace
{
  class inline_test_thread_pool : public exec::thread_pool_base<inline_test_thread_pool>
  {
   public:
    [[nodiscard]]
    auto available_parallelism() const noexcept -> std::uint32_t
    {
      return 1;
    }

    [[nodiscard]]
    static constexpr auto forward_progress_guarantee() noexcept -> ex::forward_progress_guarantee
    {
      return ex::forward_progress_guarantee::parallel;
    }

    void enqueue(exec::_pool_::task_base *task, std::uint32_t tid = 0) noexcept
    {
      ++enqueued_;
      task->execute_(task, tid);
    }

    std::uint32_t enqueued_ = 0;
  };

  struct value_capture_error
  {};

  struct throwing_value
  {
    throwing_value() = default;

    throwing_value(throwing_value const &)
    {
      throw value_capture_error{};
    }

    throwing_value(throwing_value &&)
    {
      throw value_capture_error{};
    }
  };

  struct completion_state
  {
    int                completions_ = 0;
    std::exception_ptr error_{};
  };

  struct counting_receiver
  {
    using receiver_concept = ex::receiver_tag;

    template <class... Values>
    void set_value(Values &&...) noexcept
    {
      ++state_->completions_;
    }

    void set_error(std::exception_ptr error) noexcept
    {
      ++state_->completions_;
      state_->error_ = std::move(error);
    }

    void set_stopped() noexcept
    {
      ++state_->completions_;
    }

    [[nodiscard]]
    auto get_env() const noexcept -> ex::env<>
    {
      return {};
    }

    completion_state *state_;
  };

  TEST_CASE("thread_pool_base bulk stops after value capture fails", "[thread_pool_base][bulk]")
  {
    inline_test_thread_pool pool;
    completion_state        state;
    int                     bulk_calls = 0;

    auto sndr = ex::schedule(pool.get_scheduler())
              | ex::then([]() noexcept { return throwing_value{}; })
              | ex::bulk_chunked(ex::par,
                                 1,
                                 [&](int, int, throwing_value &) noexcept { ++bulk_calls; });
    auto op = ex::connect(std::move(sndr), counting_receiver{&state});

    ex::start(op);

    CHECK(state.completions_ == 1);
    REQUIRE(state.error_);
    CHECK_THROWS_AS(std::rethrow_exception(state.error_), value_capture_error);
    CHECK(bulk_calls == 0);
    CHECK(pool.enqueued_ == 1);
  }
}  // namespace
#endif  // !STDEXEC_NO_STDCPP_EXCEPTIONS()
