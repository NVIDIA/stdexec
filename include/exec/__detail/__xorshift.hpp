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

/* I have taken and modified this code from https://gist.github.com/Leandros/6dc334c22db135b033b57e9ee0311553 */
/* Copyright (c) 2018 Arvid Gerstmann. */
/* This code is licensed under MIT license. */

#pragma once

#include <cstdint>
#include <random>

namespace exec {

  class xorshift {
   public:
    using result_type = std::uint32_t;

    static constexpr auto(min)() -> result_type {
      return 0;
    }

    static constexpr auto(max)() -> result_type {
      return UINT32_MAX;
    }

    friend auto operator==(xorshift const &, xorshift const &) -> bool = default;

    xorshift()
      : m_seed(0xc1f651c67c62c6e0ull) {
    }

    explicit xorshift(std::random_device &rd) {
      seed(rd);
    }

    explicit xorshift(std::uint64_t seed)
      : m_seed(seed) {
    }

    void seed(std::random_device &rd) {
      m_seed = std::uint64_t(rd()) << 31 | std::uint64_t(rd());
    }

    auto operator()() -> result_type {
      std::uint64_t result = m_seed * 0xd989bcacc137dcd5ull;
      m_seed ^= m_seed >> 11;
      m_seed ^= m_seed << 31;
      m_seed ^= m_seed >> 18;
      return std::uint32_t(result >> 32ull);
    }

    void discard(unsigned long long n) {
      for (unsigned long long i = 0; i < n; ++i)
        operator()();
    }

   private:
    std::uint64_t m_seed;
  };

} // namespace exec