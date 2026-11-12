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
  template <class Alloc>
  concept declares_construct_or_destroy = requires(Alloc &alloc, int *p) { alloc.construct(p, 42); }
                                       || requires(Alloc &alloc, int *p) { alloc.destroy(p); };

  TEST_CASE("exec::__frame_allocator is constructible", "[types][frame_allocator]")
  {
    using namespace exec;

    SECTION("with a memory_resource*")
    {
      std::pmr::memory_resource         *rsc = std::pmr::new_delete_resource();
      __frame_allocator_t<decltype(rsc)> fa(rsc);

      STATIC_REQUIRE(std::same_as<decltype(fa)::value_type, std::byte>);
      STATIC_REQUIRE(!declares_construct_or_destroy<decltype(fa)>);
      REQUIRE(fa.resource() == rsc);
    }

    SECTION("with a specific resource")
    {
      std::pmr::unsynchronized_pool_resource rsc;
      __frame_allocator_t<decltype(rsc) *>   fa(&rsc);

      STATIC_REQUIRE(std::same_as<decltype(fa)::value_type, std::byte>);
      STATIC_REQUIRE(!declares_construct_or_destroy<decltype(fa)>);
      STATIC_REQUIRE(
        std::same_as<decltype(fa.resource()), std::pmr::unsynchronized_pool_resource *>);
    }

    SECTION("with an allocator")
    {
      std::allocator<int>                rsc;
      __frame_allocator_t<decltype(rsc)> fa(rsc);

      STATIC_REQUIRE(std::same_as<decltype(fa)::value_type, std::byte>);
      STATIC_REQUIRE(!declares_construct_or_destroy<decltype(fa)>);
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
  constexpr decltype(auto) frame_allocator_arg_from(Resource &resource) noexcept
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

  TEMPLATE_TEST_CASE("__frame_allocator doesn't do uses-allocator construction of "
                     "leading-allocator types",
                     "[types][frame_allocator]",
                     std::allocator<int>,
                     std::pmr::memory_resource *,
                     std::pmr::unsynchronized_pool_resource *)
  {
    using raw_alloc = exec::__frame_allocator_t<TestType>;
    using traits_t =
      std::allocator_traits<exec::__frame_allocator_t<TestType>>::template rebind_traits<
        uses_leading_allocator<raw_alloc>>;
    using alloc_t = traits_t::allocator_type;

    resource_for_t<TestType> rsc;
    alloc_t                  alloc(frame_allocator_arg_from(rsc));

    uses_leading_allocator<raw_alloc> *p = traits_t::allocate(alloc, 1);
    traits_t::construct(alloc, p, 42);

    // a frame allocator only provides storage
    REQUIRE(!p->usedAllocator);

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

  TEMPLATE_TEST_CASE("__frame_allocator doesn't do uses-allocator construction of "
                     "trailing-allocator types",
                     "[types][frame_allocator]",
                     std::allocator<int>,
                     std::pmr::memory_resource *,
                     std::pmr::unsynchronized_pool_resource *)
  {
    using raw_alloc = exec::__frame_allocator_t<TestType>;
    using traits_t =
      std::allocator_traits<exec::__frame_allocator_t<TestType>>::template rebind_traits<
        uses_trailing_allocator<raw_alloc>>;
    using alloc_t = traits_t::allocator_type;

    resource_for_t<TestType> rsc;
    alloc_t                  alloc(frame_allocator_arg_from(rsc));

    uses_trailing_allocator<raw_alloc> *p = traits_t::allocate(alloc, 1);
    traits_t::construct(alloc, p, 42);

    // a frame allocator only provides storage
    REQUIRE(!p->usedAllocator);

    traits_t::destroy(alloc, p);
    traits_t::deallocate(alloc, p, 1);
  }

  struct construct_counts
  {
    int construct = 0;
    int destroy   = 0;
  };

  //! Counts calls to its own construct and destroy
  template <class T>
  struct constructing_allocator : std::allocator<T>
  {
    template <class U>
    struct rebind
    {
      using other = constructing_allocator<U>;
    };

    construct_counts *counts_;

    explicit constructing_allocator(construct_counts *counts) noexcept
      : counts_(counts)
    {}

    template <class U>
    constructing_allocator(constructing_allocator<U> const &other) noexcept
      : counts_(other.counts_)
    {}

    template <class U, class... Args>
    void construct(U *p, Args &&...args)
    {
      ++counts_->construct;
      std::construct_at(p, static_cast<Args &&>(args)...);
    }

    template <class U>
    void destroy(U *p) noexcept
    {
      ++counts_->destroy;
      std::destroy_at(p);
    }
  };

  TEST_CASE("__frame_allocator doesn't call a delegate allocator's construct or destroy",
            "[types][frame_allocator]")
  {
    using alloc_t  = exec::__frame_allocator_t<constructing_allocator<int>>;
    using traits_t = std::allocator_traits<alloc_t>::template rebind_traits<int>;

    construct_counts         counts;
    traits_t::allocator_type alloc{alloc_t(constructing_allocator<int>(&counts))};

    int *p = traits_t::allocate(alloc, 1);
    traits_t::construct(alloc, p, 42);
    REQUIRE(*p == 42);
    traits_t::destroy(alloc, p);
    traits_t::deallocate(alloc, p, 1);

    REQUIRE(counts.construct == 0);
    REQUIRE(counts.destroy == 0);
  }
}  // namespace
