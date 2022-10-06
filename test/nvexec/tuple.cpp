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

#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>

#include "nvexec/detail/throw_on_cuda_error.cuh"
#include "nvexec/detail/tuple.cuh"
#include "common.cuh"

using example::cuda::tuple_t;
using example::cuda::apply;

TEST_CASE("tuple size is correct", "[cuda][stream][containers][tuple]") {
  STATIC_REQUIRE(tuple_t<double>::size == 1);
  STATIC_REQUIRE(tuple_t<int, double>::size == 2);
  STATIC_REQUIRE(tuple_t<char, int, double>::size == 3);
}

TEST_CASE("tuple emplaces from CPU", "[cuda][stream][containers][tuple]") {
  tuple_t<int, double> t;
  apply([](int i, double d) {
      REQUIRE(i == 0);
      REQUIRE(d == 0.0);
  }, t);

  t = tuple_t<int, double>{42, 4.2};
  apply([](int i, double d) {
      REQUIRE(i == 42);
      REQUIRE(d == 4.2);
  }, t);

  t = tuple_t<int, double>{24, 2.4};
  apply([](int i, double d) {
      REQUIRE(i == 24);
      REQUIRE(d == 2.4);
  }, t);
}

template <class Tuple, class... As>
__global__ void kernel(Tuple* t, As... as) {
  *t = tuple_t<As...>(as...);
}

TEST_CASE("tuple emplaces alternative from GPU", "[cuda][stream][containers][tuple]") {
  using tuple_t = tuple_t<int, double>;
  tuple_t *t{};
  THROW_ON_CUDA_ERROR(cudaMallocHost(&t, sizeof(tuple_t)));
  new (t) tuple_t();
  apply([](int i, double d) {
      REQUIRE(i == 0);
      REQUIRE(d == 0.0);
  }, *t);

  kernel<<<1, 1>>>(t, 42, 4.2);
  THROW_ON_CUDA_ERROR(cudaDeviceSynchronize());
  apply([](int i, double d) {
      REQUIRE(i == 42);
      REQUIRE(d == 4.2);
  }, *t);

  kernel<<<1, 1>>>(t, 24, 2.4);
  THROW_ON_CUDA_ERROR(cudaDeviceSynchronize());
  apply([](int i, double d) {
      REQUIRE(i == 24);
      REQUIRE(d == 2.4);
  }, *t);

  THROW_ON_CUDA_ERROR(cudaFreeHost(t));
}

