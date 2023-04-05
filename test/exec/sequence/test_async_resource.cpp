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

#include "exec/sequence/async_resource.hpp"

#include <catch2/catch.hpp>

struct Resource {
  int n_open_called = 0;
  int n_close_called = 0;

  struct Token {
    Resource* r;

    friend auto tag_invoke(exec::async_resource::close_t, Token& t) {
      return stdexec::just() | stdexec::then([r = t.r] { ++r->n_close_called; });
    }
  };

  friend auto tag_invoke(exec::async_resource::open_t, Resource& r) noexcept {
    return stdexec::just(&r) | stdexec::then([](Resource* r) {
             ++r->n_open_called;
             return Token{r};
           });
  }
};

TEST_CASE("async_resource", "[sequence][async_resource]") {
  Resource resource;
  // bool called = false;
  exec::async_resource::open(resource);
  stdexec::sync_wait(exec::ignore_all(exec::async_resource::run(resource)));
  // CHECK(called);
  CHECK(resource.n_open_called == 1);
  CHECK(resource.n_close_called == 1);
}