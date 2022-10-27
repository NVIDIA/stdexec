/*
 * Copyright (c) 2022 Lucian Radu Teodorescu
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

#pragma once

#include <catch2/catch.hpp>
#include <test_common/type_helpers.hpp>
#include <stdexec/execution.hpp>

namespace ex = stdexec;

template <class... Values>
struct fallible_just {
  std::tuple<Values...> values_;
  using completion_signatures =
    ex::completion_signatures<
      ex::set_value_t(Values...),
      ex::set_error_t(std::exception_ptr)>;

  template <class Receiver>
  struct operation : immovable {
    std::tuple<Values...> values_;
    Receiver rcvr_;

    friend void tag_invoke(ex::start_t, operation& self) noexcept try {
      std::apply(
        [&](Values&... ts) {
          ex::set_value(std::move(self.rcvr_), std::move(ts)...);
        },
        self.values_);
    } catch(...) {
      ex::set_error(std::move(self.rcvr_), std::current_exception());
    }
  };

  template <class Receiver>
  friend auto tag_invoke(ex::connect_t, fallible_just&& self, Receiver&& rcvr) ->
      operation<std::decay_t<Receiver>> {
    return {{}, std::move(self.values_), std::forward<Receiver>(rcvr)};
  }
};

template <class... Values>
fallible_just(Values...) -> fallible_just<Values...>;
