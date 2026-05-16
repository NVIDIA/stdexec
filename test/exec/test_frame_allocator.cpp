/*
 * Copyright (c) 2026 Ian Petersen
 * Copyright (c) 2026 NVIDIA Corporation
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

#include <exec/__frame_allocator.hpp>

#include <catch2/catch_all.hpp>

#include <stdexec/execution.hpp>

#include <concepts>
#include <memory>

namespace ex = STDEXEC;

namespace
{
  TEST_CASE("exec::__frame_allocator is constructible", "[types][frame_allocator]")
  {
    using namespace exec::__fa;

    SECTION("with a memory_resource*")
    {
      std::pmr::memory_resource*         rsc = std::pmr::new_delete_resource();
      __frame_allocator_t<decltype(rsc)> fa(rsc);

      STATIC_REQUIRE(std::same_as<decltype(fa), std::pmr::polymorphic_allocator<std::byte>>);
    }

    SECTION("with a specific resource")
    {
      std::pmr::unsynchronized_pool_resource rsc;
      __frame_allocator_t<decltype(rsc)*>    fa(&rsc);

      STATIC_REQUIRE(!std::same_as<decltype(fa), std::pmr::polymorphic_allocator<std::byte>>);
    }

    SECTION("with an allocator")
    {
      std::allocator<int>                rsc;
      __frame_allocator_t<decltype(rsc)> fa(rsc);

      STATIC_REQUIRE(std::same_as<decltype(fa), std::allocator<std::byte>>);
    }
  }
}  // namespace
