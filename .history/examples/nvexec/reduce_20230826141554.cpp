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

#include <nvexec/stream_context.cuh>
#include <stdexec/execution.hpp>

#include <thrust/device_vector.h>

#include <cstdio>
#include <span>

namespace ex = stdexec;
using stdexec::__tag_invoke::tag_invoke;

struct sink_receiver {
  using is_receiver = void;

  friend void tag_invoke(stdexec::set_value_t, sink_receiver, auto&&...) noexcept {
  }

  friend void tag_invoke(stdexec::set_error_t, sink_receiver, auto&&) noexcept {
  }

  friend void tag_invoke(stdexec::set_stopped_t, sink_receiver) noexcept {
  }

  friend stdexec::empty_env tag_invoke(stdexec::get_env_t, sink_receiver) noexcept {
    return {};
  }
};

struct empty_environment { };

// unqualified call to tag_invoke:
int main() {

  const int n = 2 * 1024;
  thrust::device_vector<float> input(n, 1.0f);
  float* first = thrust::raw_pointer_cast(input.data());
  float* last = thrust::raw_pointer_cast(input.data()) + input.size();
  nvexec::stream_context stream_ctx{};
  auto snd = ex::just(std::span{first, last}) | nvexec::reduce(52.0f);

  using snd_type = decltype(snd);
  typename snd_type::__t::_completion_signatures_t<snd_type, int> hey;

  stdexec::completion_signatures sup = hey;
  stdexec::print(sup);
  // stdexec::__compl_sigs_impl
  // auto [result] =
  // stdexec::sync_wait(ex::on(stream_ctx.get_scheduler(), std::move(snd))).value();
}

/*
 static_assert(stdexec::__completion_signature<decltype(hey)>, "Wat");

  stdexec::print(sup);
  // nvexec::stream_scheduler gpu = stream_ctx.get_scheduler();
  // using stdexec::__tag_invoke::tag_invoke;

  // tag_invoke(stdexec::get_completion_signatures, snd, stdexec::no_env{});
*/