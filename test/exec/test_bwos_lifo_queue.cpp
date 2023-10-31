/*
 * Copyright (c) 2023 Maikel Nadolski
 * Copyright (c) 2023 NVIDIA Corporation
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
#include "exec/__detail/__bwos_lifo_queue.hpp"

#include <catch2/catch.hpp>

TEST_CASE("exec::bwos::lifo_queue - ", "[bwos]") {
  exec::bwos::lifo_queue<int*> queue(8, 2);
  int x = 1;
  int y = 2;
  SECTION("Observers") {
    CHECK(queue.block_size() == 2);
    CHECK(queue.num_blocks() == 8);
  }
  SECTION("Empty Get") {
    CHECK(queue.pop_back() == nullptr);
  }
  SECTION("Empty Steal") {
    CHECK(queue.steal_front() == nullptr);
  }
  SECTION("Put one, get one") {
    CHECK(queue.push_back(&x));
    CHECK(queue.pop_back() == &x);
    CHECK(queue.pop_back() == nullptr);
  }
  SECTION("Put one, steal none") {
    CHECK(queue.push_back(&x));
    CHECK(queue.steal_front() == nullptr);
    CHECK(queue.pop_back() == &x);
  }
  SECTION("Put one, get one, put one, get one") {
    CHECK(queue.push_back(&x));
    CHECK(queue.pop_back() == &x);
    CHECK(queue.push_back(&y));
    CHECK(queue.pop_back() == &y);
    CHECK(queue.pop_back() == nullptr);
  }
  SECTION("Put two, get two") {
    CHECK(queue.push_back(&x));
    CHECK(queue.push_back(&y));
    CHECK(queue.pop_back() == &y);
    CHECK(queue.pop_back() == &x);
    CHECK(queue.pop_back() == nullptr);
  }
  SECTION("Put three, Steal two") {
    CHECK(queue.push_back(&x));
    CHECK(queue.push_back(&y));
    CHECK(queue.push_back(&x));
    CHECK(queue.steal_front() == &x);
    CHECK(queue.steal_front() == &y);
    CHECK(queue.steal_front() == nullptr);
    CHECK(queue.pop_back() == &x);
    CHECK(queue.pop_back() == nullptr);
  }
  SECTION("Put 4, Steal 1, Get 3") {
    CHECK(queue.push_back(&x));
    CHECK(queue.push_back(&y));
    CHECK(queue.push_back(&x));
    CHECK(queue.push_back(&y));
    CHECK(queue.steal_front() == &x);
    CHECK(queue.pop_back() == &y);
    CHECK(queue.pop_back() == &x);
    CHECK(queue.pop_back() == &y);
    CHECK(queue.pop_back() == nullptr);
  }
}