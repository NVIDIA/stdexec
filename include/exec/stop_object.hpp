/*
 * Copyright (c) 2024 Kirk Shoop
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
#pragma once

#include "stdexec/__detail/__execution_fwd.hpp"

#include "stdexec/concepts.hpp"
#include "stdexec/functional.hpp"
#include "stdexec/stop_token.hpp"

#include "finally.hpp"

#include "async_object.hpp"
#include "async_using.hpp"
#include "packaged_async_object.hpp"

#include <array>

namespace exec {

template<class _Token, class _Callback>
struct stop_callback_object {
  using object = typename _Token::template callback_type<_Callback>;
  class handle {
    friend struct stop_callback_object;
    explicit handle() {}
  };
  using storage = std::optional<object>;

  template<class _T, class _C>
  auto async_construct(storage& stg, _T&& __t, _C&& __c) const noexcept {
    auto construct = [](storage& stg, auto&& __t, auto&& __c) noexcept -> handle {
      stg.emplace(static_cast<_T&&>(__t), static_cast<_C&&>(__c));
      return handle{};
    };
    return stdexec::then(stdexec::just(std::ref(stg), static_cast<_T&&>(__t), static_cast<_C&&>(__c)), construct);
  }
  auto async_destruct(storage& stg) const noexcept {
    auto destruct = [](storage& stg) noexcept {
      stg.reset();
    };
    return stdexec::then(stdexec::just(std::ref(stg)), destruct);
  }
};

struct stop_object {
  using object = stdexec::inplace_stop_source;
  class handle {
    object* source;
    friend struct stop_object;
    explicit handle(object& s) : source(&s) {}
  public:
    handle() = delete;
    handle(const handle&) = default;
    handle(handle&&) = default;
    handle& operator=(const handle&) = default;
    handle& operator=(handle&&) = default;

    stdexec::inplace_stop_token get_token() const noexcept {
      return source->get_token();
    }
    bool stop_requested() const noexcept {
      return source->stop_requested();
    }
    static constexpr bool stop_possible() noexcept {
      return true;
    }
    bool request_stop() noexcept {
      return source->request_stop();
    }
    // chain has two effects
    // 1. chain applies the stop_token for this stop-source to the env of 
    // the given sender.
    // 2. chain retrieves the stop_token from the environment of the receiver 
    // connected to the returned sender, and uses a stop_callback_object 
    // to forward a stop_request from the external stop-source to this 
    // stop-source  
    auto chain(auto sender) noexcept {
      auto stop_token = source->get_token();
      auto bind = [sender, stop_token, source = this->source](auto ext_stop) noexcept {
        auto callback = [source]() noexcept {source->request_stop();};
        auto with_callback = [sender, stop_token](auto cb) noexcept {
          return stdexec::__write_env(
            std::move(sender), 
            stdexec::__env::__with(stop_token, stdexec::get_stop_token));
        };
        exec::packaged_async_object cb{stop_callback_object<decltype(ext_stop), decltype(callback)>{}, ext_stop, callback};
        return exec::async_using(with_callback, cb);
      };
      return stdexec::let_value(stdexec::read_env(stdexec::get_stop_token), bind);
    }
  };
  using storage = std::optional<object>;

  auto async_construct(storage& stg) const noexcept {
    auto construct = [](storage& stg) noexcept -> handle {
      stg.emplace();
      return handle{stg.value()};
    };
    return stdexec::then(stdexec::just(std::ref(stg)), construct);
  }
  auto async_destruct(storage& stg) const noexcept {
    auto destruct = [](storage& stg) noexcept {
      stg.reset();
    };
    return stdexec::then(stdexec::just(std::ref(stg)), destruct);
  }
};


} // namespace exec 
