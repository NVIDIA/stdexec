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
  Resource* upstream_;

  struct Token {
    friend auto tag_invoke(exec::async_resource::close_t, Token& t) {
      return stdexec::just();
    }
  };

  using just_token_t = decltype(stdexec::just(Token{}));
  using just_stopped_t = decltype(stdexec::just_stopped());

  friend exec::variant_sender<just_token_t, just_stopped_t>
    tag_invoke(exec::async_resource::open_t, StopResource& r) noexcept {
    return stdexec::just_stopped();
  }
};

struct ErrorResource {
  struct Token {
    friend auto tag_invoke(exec::async_resource::close_t, Token& t) {
      return stdexec::just();
    }
  };

  using just_token_t = decltype(stdexec::just(Token{}));
  using just_error_t = decltype(stdexec::just_error(0));

  friend exec::variant_sender<just_token_t, just_error_t>
    tag_invoke(exec::async_resource::open_t, ErrorResource& r) noexcept {
    return stdexec::just_error(42);
  }
};

TEST_CASE("async_resource - use_resources", "[sequence][async_resource]") {
  Resource resource;
  bool called = false;
  auto [value] = stdexec::sync_wait(exec::use_resources(
    [&](auto&&...) {
      called = true;
      return stdexec::just(42);
    },
    resource,
    resource,
    resource)).value();
  CHECK(value == 42);
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
    resource,
    stop_resource));
  CHECK_FALSE(called);
  CHECK(resource.n_open_called == 1);
  CHECK(resource.n_close_called == 1);
}

TEST_CASE("async_resource - error use_resources", "[sequence][async_resource]") {
  Resource resource;
  ErrorResource error_resource;
  bool called = false;
  CHECK_THROWS(stdexec::sync_wait(exec::use_resources(
    [&](Resource::Token& t1, ErrorResource::Token& t2) {
      called = true;
      return stdexec::just();
    },
    resource,
    error_resource)));
  CHECK_FALSE(called);
  CHECK(resource.n_open_called == 1);
  CHECK(resource.n_close_called == 1);
}