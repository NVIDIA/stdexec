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
    using namespace exec;

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

  template <class Resource>
  struct resource_for
  {
    using type = Resource;
  };

  template <class Resource>
    requires std::is_pointer_v<Resource>
  struct resource_for<Resource>
  {
    using type = std::pmr::unsynchronized_pool_resource;
  };

  template <class Resource>
  using resource_for_t = resource_for<Resource>::type;

  template <class Resource>
  constexpr decltype(auto) frame_allocator_arg_from(Resource& resource) noexcept
  {
    if constexpr (ex::__std::derived_from<Resource, std::pmr::memory_resource>)
    {
      return &resource;
    }
    else
    {
      return resource;
    }
  }

  template <class Alloc>
  struct uses_leading_allocator
  {
    using allocator_type = Alloc;

    constexpr explicit uses_leading_allocator(int) noexcept
      : usedAllocator(false)
    {}

    constexpr explicit uses_leading_allocator(std::allocator_arg_t, Alloc const &, int) noexcept
      : usedAllocator(true)
    {}

    bool usedAllocator;
  };

  TEMPLATE_TEST_CASE("__frame_allocator correctly does uses-allocator construction with "
                     "leading-allocator types",
                     "[types][frame_allocator]",
                     std::allocator<int>,
                     std::pmr::memory_resource*,
                     std::pmr::unsynchronized_pool_resource*)
  {
    using raw_alloc = exec::__frame_allocator_t<TestType>;
    using traits_t =
      std::allocator_traits<exec::__frame_allocator_t<TestType>>::template rebind_traits<
        uses_leading_allocator<raw_alloc>>;
    using alloc_t = traits_t::allocator_type;

    resource_for_t<TestType> rsc;
    alloc_t                  alloc(frame_allocator_arg_from(rsc));

    uses_leading_allocator<raw_alloc>* p = traits_t::allocate(alloc, 1);
    traits_t::construct(alloc, p, 42);

    if (std::is_pointer_v<TestType>)
    {
      // std::pmr::polymorphic_allocator and the static equivalent do uses-allocator
      // construction
      REQUIRE(p->usedAllocator);
    }
    else
    {
      // std::allocator does not do uses-allocator construction
      REQUIRE(!p->usedAllocator);
    }

    traits_t::destroy(alloc, p);
    traits_t::deallocate(alloc, p, 1);
  }

  template <class Alloc>
  struct uses_trailing_allocator
  {
    using allocator_type = Alloc;

    constexpr explicit uses_trailing_allocator(int) noexcept
      : usedAllocator(false)
    {}

    constexpr explicit uses_trailing_allocator(int, Alloc const &) noexcept
      : usedAllocator(true)
    {}

    bool usedAllocator;
  };

  TEMPLATE_TEST_CASE("__frame_allocator correctly does uses-allocator construction with "
                     "trailing-allocator types",
                     "[types][frame_allocator]",
                     std::allocator<int>,
                     std::pmr::memory_resource*,
                     std::pmr::unsynchronized_pool_resource*)
  {
    using raw_alloc = exec::__frame_allocator_t<TestType>;
    using traits_t =
      std::allocator_traits<exec::__frame_allocator_t<TestType>>::template rebind_traits<
        uses_trailing_allocator<raw_alloc>>;
    using alloc_t = traits_t::allocator_type;

    resource_for_t<TestType> rsc;
    alloc_t                  alloc(frame_allocator_arg_from(rsc));

    uses_trailing_allocator<raw_alloc>* p = traits_t::allocate(alloc, 1);
    traits_t::construct(alloc, p, 42);

    if (std::is_pointer_v<TestType>)
    {
      // std::pmr::polymorphic_allocator and the static equivalent do uses-allocator
      // construction
      REQUIRE(p->usedAllocator);
    }
    else
    {
      // std::allocator does not do uses-allocator construction
      REQUIRE(!p->usedAllocator);
    }

    traits_t::destroy(alloc, p);
    traits_t::deallocate(alloc, p, 1);
  }
}  // namespace
