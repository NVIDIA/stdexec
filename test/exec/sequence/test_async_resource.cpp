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

#include "exec/variant_sender.hpp"

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

struct StopResource {
  struct Token {
    friend auto tag_invoke(exec::async_resource::close_t, Token& t) {
      return stdexec::just();
    }
  };

  friend auto tag_invoke(exec::async_resource::open_t, StopResource& r) noexcept {
    using just_token_t = decltype(stdexec::just(Token{}));
    using just_stopped_t = decltype(stdexec::just_stopped());
    return exec::variant_sender<just_token_t, just_stopped_t>{stdexec::just_stopped()};
  }
};

TEST_CASE("async_resource - use_resources", "[sequence][async_resource]") {
  Resource resource;
  bool called = false;
  stdexec::sync_wait(exec::use_resources(
    [&](auto&&...) {
      called = true;
      return stdexec::just();
    },
    resource, resource, resource));
  CHECK(called);
  CHECK(resource.n_open_called == 3);
  CHECK(resource.n_close_called == 3);
}

TEST_CASE("async_resource - stopped use_resources", "[sequence][async_resource]") {
  Resource resource;
  StopResource stop_resource;
  bool called = false;
  stdexec::sync_wait(exec::use_resources(
    [&](Resource::Token& t1, StopResource::Token& t2) {
      called = true;
      return stdexec::just();
    },
    resource, stop_resource));
  CHECK_FALSE(called);
  CHECK(resource.n_open_called == 0);
  CHECK(resource.n_close_called == 0);
}