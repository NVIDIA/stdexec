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

template <class T>
struct type_print;

int main() {
  using namespace stdexec;

  const int n = 2 * 1024;


  thrust::device_vector<float> input(n, 1.0f);
  float* first = thrust::raw_pointer_cast(input.data());
  float* last = thrust::raw_pointer_cast(input.data()) + input.size();
  auto stream_ctx = nvexec::stream_context{};
  auto span = std::span{first, last};
  auto test = nvexec::reduce(3.0f);
  auto snd = ex::just(std::span{first, last}) | nvexec::reduce(52.0f);
  // stdexec::print(snd);
  using eval = __meval<                       //
      __gather_signal,
      set_value_t,
      __completion_signatures_of_t<_Sender, _Env>,
      _Tuple,
      _Variant>;

  using stdexec::__tag_invoke::tag_invoke;
  tag_invoke(stdexec::get_completion_signatures, snd, test_env{});

  // auto [result] = stdexec::sync_wait(ex::on(stream_ctx.get_scheduler(), std::move(snd))).value();
}

/*
 auto snd = ex::just(std::span{first, last})
                      | nvexec::reduce(52.0f);

 
  using eval =  __mtry_eval<__try_value_types_of_t, decltype(snd), __sync_wait::__env,  nvexec::_strm::reduce_t::_set_value_t<decltype(span), float>, __q<__compl_sigs::__ensure_concat>>;
  type_print<eval> heybro;



using complet = __gather_completions_for<
    set_value_t,
    decltype(snd),
    test_env,
    nvexec::_strm::reduce_t::_set_value_t<decltype(span), float>,
    __q<__compl_sigs::__ensure_concat>>;
  type_print<complet> heybro;
  nvexec::stream_scheduler gpu = stream_ctx.get_scheduler();

*/
