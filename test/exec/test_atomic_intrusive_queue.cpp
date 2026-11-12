/*
 * Copyright (c) 2026 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions
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

#include <catch2/catch_all.hpp>

#include <exec/detail/atomic_intrusive_queue.hpp>

#include <utility>

namespace
{
  struct test_node
  {
    test_node* next_{nullptr};
  };

  using test_queue   = STDEXEC::__intrusive_queue<&test_node::next_>;
  using atomic_queue = exec::__atomic_intrusive_queue<&test_node::next_>;

  TEST_CASE("atomic_intrusive_queue::prepend handles empty input", "[atomic_intrusive_queue]")
  {
    atomic_queue queue;
    queue.prepend(test_queue{});
    CHECK(queue.empty());

    test_node existing{};
    REQUIRE(queue.push_front(&existing));
    queue.prepend(test_queue{});
    CHECK_FALSE(queue.empty());

    auto contents = queue.pop_all();
    REQUIRE(contents.pop_front() == &existing);
    CHECK(contents.empty());
  }

  TEST_CASE("atomic_intrusive_queue::prepend preserves queue order", "[atomic_intrusive_queue]")
  {
    atomic_queue queue;
    test_node    existing{};
    test_node    first{};
    test_node    second{};
    REQUIRE(queue.push_front(&existing));

    test_queue incoming;
    incoming.push_back(&first);
    incoming.push_back(&second);
    queue.prepend(std::move(incoming));

    auto contents = queue.pop_all();
    CHECK(contents.pop_front() == &first);
    CHECK(contents.pop_front() == &second);
    CHECK(contents.pop_front() == &existing);
    CHECK(contents.empty());
  }
}  // namespace
