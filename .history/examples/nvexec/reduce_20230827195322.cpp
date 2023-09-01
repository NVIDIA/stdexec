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

struct test_env { };

struct test_tag { };

struct test_sender { };

template <class _InitT, class _Fun>
struct Data {
  _InitT __initT_;
  STDEXEC_NO_UNIQUE_ADDRESS _Fun __fun_;
  static constexpr auto __mbrs_ = stdexec::__mliterals<&Data::__initT_, &Data::__fun_>();
};

struct Children { };

template <class _Tag, class... _Captures>
struct fn_struct { };

int main() {
  using namespace stdexec;

  const int n = 2 * 1024;
  struct MySender {};
struct MyEnv {};
struct MySigs {};
struct MySetValue {};
struct MySetError {};
struct MySetStopped {};

using MyCompletionSignature = 
    __try_make_completion_signatures<
        MySender,
        MyEnv,
        MySigs,
        MySetValue,
        MySetError,
        MySetStopped
    >;

  thrust::device_vector<float> input(n, 1.0f);
  float* first = thrust::raw_pointer_cast(input.data());
  float* last = thrust::raw_pointer_cast(input.data()) + input.size();
  auto stream_ctx = nvexec::stream_context{};
  auto snd = ex::just(std::span{first, last})
                      | nvexec::reduce(52.0f);
  static_assert()                    
  // using stdexec::__tag_invoke::tag_invoke;

  // tag_invoke(stdexec::get_completion_signatures, snd, test_env{});
  
  //  auto [result] =
  //  stdexec::sync_wait(ex::on(stream_ctx.get_scheduler(), std::move(snd))).value();
}

/*
 auto snd = ex::just(std::span{first, last})
                      | nvexec::reduce(52.0f);

 


  nvexec::stream_scheduler gpu = stream_ctx.get_scheduler();

*/