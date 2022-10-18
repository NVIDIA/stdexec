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

#include <cuda/std/tuple>

#include "nvexec/detail/throw_on_cuda_error.cuh"
#include "nvexec/detail/variant.cuh"
#include "common.cuh"

using nvexec::variant_t;
using nvexec::visit;

TEST_CASE("variant max size is correct", "[cuda][stream][containers][variant]") {
  STATIC_REQUIRE(variant_t<double>::max_size == sizeof(double));
  STATIC_REQUIRE(variant_t<int, double>::max_size == sizeof(double));
  STATIC_REQUIRE(variant_t<char, int, double>::max_size == sizeof(double));
  STATIC_REQUIRE(variant_t<char, int>::max_size == sizeof(int));
  STATIC_REQUIRE(variant_t<char>::max_size == sizeof(char));
}

TEST_CASE("variant max alignment is correct", "[cuda][stream][containers][variant]") {
  STATIC_REQUIRE(variant_t<double>::max_size == std::alignment_of_v<double>);
  STATIC_REQUIRE(variant_t<int, double>::max_size == std::alignment_of_v<double>);
  STATIC_REQUIRE(variant_t<char, int, double>::max_size == std::alignment_of_v<double>);
  STATIC_REQUIRE(variant_t<char, int>::max_size == std::alignment_of_v<int>);
  STATIC_REQUIRE(variant_t<char>::max_size == std::alignment_of_v<char>);
}

TEST_CASE("variant size is correct", "[cuda][stream][containers][variant]") {
  STATIC_REQUIRE(variant_t<double>::size == 1);
  STATIC_REQUIRE(variant_t<int, double>::size == 2);
  STATIC_REQUIRE(variant_t<char, int, double>::size == 3);
}

TEST_CASE("variant emplaces alternative from CPU", "[cuda][stream][containers][variant]") {
  variant_t<int, double> v;
  REQUIRE(v.index_ == 0);

  v.emplace<double>(4.2);
  visit([](auto alt) {
      REQUIRE(alt == 4.2);
  }, v);

  v.emplace<double>(42);
  visit([](auto alt) {
      REQUIRE(alt == 42);
  }, v);
}

template <class T, class V>
__global__ void kernel(V* v, T alt) {
  v->template emplace<T>(alt);
}

TEST_CASE("variant emplaces alternative from GPU", "[cuda][stream][containers][variant]") {
  using variant_t = variant_t<int, double>;
  variant_t *v{};
  THROW_ON_CUDA_ERROR(cudaMallocHost(&v, sizeof(variant_t)));
  new (v) variant_t();

  REQUIRE(v->index_ == 0);

  kernel<<<1, 1>>>(v, 4.2);
  THROW_ON_CUDA_ERROR(cudaDeviceSynchronize());

  visit([](auto alt) {
      REQUIRE(alt == 4.2);
  }, *v);

  kernel<<<1, 1>>>(v, 42);
  THROW_ON_CUDA_ERROR(cudaDeviceSynchronize());

  visit([](auto alt) {
      REQUIRE(alt == 42);
  }, *v);

  THROW_ON_CUDA_ERROR(cudaFreeHost(v));
}

TEST_CASE("variant works with cuda tuple", "[cuda][stream][containers][variant]") {
  variant_t<cuda::std::tuple<int, double>, cuda::std::tuple<char, int>> v;
  REQUIRE(v.index_ == 0);

  v.emplace<cuda::std::tuple<int, double>>(42, 4.2);
  visit([](auto& tuple) {
      cuda::std::apply([](auto i, auto d) {
        REQUIRE(i == 42);
        REQUIRE(d == 4.2);
      }, tuple);
  }, v);

  v.emplace<cuda::std::tuple<char, int>>('f', 4);
  visit([](auto& tuple) {
      cuda::std::apply([](auto c, auto i) {
        REQUIRE(c == 'f');
        REQUIRE(i == 4);
      }, tuple);
  }, v);
}

TEST_CASE("variant internal index bypass works", "[cuda][stream][containers][variant]") {
  variant_t<cuda::std::tuple<int, double>, cuda::std::tuple<char, int>> v;

  v.emplace<cuda::std::tuple<int, double>>(42, 4.2);
  visit([](auto& tuple) {
      cuda::std::apply([](auto i, auto d) {
        REQUIRE(i == 42);
        REQUIRE(d == 4.2);
      }, tuple);
  }, v, 0);

  v.emplace<cuda::std::tuple<char, int>>('f', 4);
  visit([](auto& tuple) {
      cuda::std::apply([](auto c, auto i) {
        REQUIRE(c == 'f');
        REQUIRE(i == 4);
      }, tuple);
  }, v, 1);
}

