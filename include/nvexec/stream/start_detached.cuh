/*
 * Copyright (c) 2022 NVIDIA Corporation
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

// clang-format Language: Cpp

#pragma once

#include <exception>

#include "common.cuh"
#include "submit.cuh"

namespace nvexec::_strm {
  namespace _start_detached {
    template <class Env>
    struct detached_receiver_t : stream_receiver_base {
      STDEXEC_ATTRIBUTE((no_unique_address)) Env env_;

      template <class... _Args>
      void set_value(_Args&&...) noexcept {
      }

      template <class _Error>
      [[noreturn]]
      void set_error(_Error&&) noexcept {
        std::terminate();
      }

      void set_stopped() noexcept {
      }

      auto get_env() const noexcept -> const Env& {
        return env_;
      }
    };

  } // namespace _start_detached

  template <>
  struct apply_sender_for<start_detached_t> {
    template <class Sender, class Env = __root_env>
    void operator()(Sender&& sndr, Env&& env = {}) const {
      using _receiver_t = _start_detached::detached_receiver_t<__decay_t<Env>>;
      _submit::submit_t{}(static_cast<Sender&&>(sndr), _receiver_t{{}, static_cast<Env&&>(env)});
    }
  };
} // namespace nvexec::_strm
