/*
 * Copyright (c) 2024 Rishabh Dwivedi <rishabhdwivedi17@gmail.com>
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

#include "catch2/catch.hpp"
#include "exec/libdispatch_queue.hpp"
#include "stdexec/execution.hpp"

namespace {
  TEST_CASE("libdispatch queue should be able to process tasks") {
    exec::libdispatch_queue queue;
    auto sch = queue.get_scheduler();

    std::vector<int> data{1, 2, 3, 4, 5};
    auto add = [](auto const & data) {
      return std::accumulate(std::begin(data), std::end(data), 0);
    };
    auto sender = stdexec::transfer_just(sch, std::move(data)) | stdexec::then(add);

    auto completion_scheduler = stdexec::get_completion_scheduler<stdexec::set_value_t>(
      stdexec::get_env(sender));

    CHECK(completion_scheduler == sch);
    auto [res] = stdexec::sync_wait(sender).value();
    CHECK(res == 15);
  }

  TEST_CASE(
    "libdispatch queue bulk algorithm should call callback function with all allowed shapes") {
    exec::libdispatch_queue queue;
    auto sch = queue.get_scheduler();

    std::vector<int> data{1, 2, 3, 4, 5};
    auto size = data.size();
    auto expensive_computation = [](auto i, auto& data) {
      data[i] = 2 * data[i];
    };
    auto add = [](auto const & data) {
      return std::accumulate(std::begin(data), std::end(data), 0);
    };
    auto sender = stdexec::transfer_just(sch, std::move(data))
                | stdexec::bulk(stdexec::par, size, expensive_computation) | stdexec::then(add);

    auto completion_scheduler = stdexec::get_completion_scheduler<stdexec::set_value_t>(
      stdexec::get_env(sender));

    CHECK(completion_scheduler == sch);
    auto [res] = stdexec::sync_wait(sender).value();
    CHECK(res == 30);
  }

  TEST_CASE("libdispatch bulk should handle exceptions gracefully") {
    exec::libdispatch_queue queue;
    auto sch = queue.get_scheduler();

    std::vector<int> data{1, 2, 3, 4, 5};
    auto size = data.size();
    auto expensive_computation = [](auto i, auto data) {
      if (i == 0)
        throw 999;
      return 2 * data[i];
    };
    auto add = [](auto const & data) {
      return std::accumulate(std::begin(data), std::end(data), 0);
    };
    auto sender = stdexec::transfer_just(sch, std::move(data))
                | stdexec::bulk(stdexec::par, size, expensive_computation) | stdexec::then(add);


    STDEXEC_TRY {
      stdexec::sync_wait(sender);
      CHECK(false);
    }
    STDEXEC_CATCH(int e) {
      CHECK(e == 999);
    }
  }
} // namespace
