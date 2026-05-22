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

#include <exec/__memory_resource_adaptor.hpp>

#include <catch2/catch_all.hpp>

#include <stdexec/execution.hpp>

#include <bit>
#include <concepts>
#include <memory>

namespace ex = STDEXEC;

namespace
{
  // this is a trivial class template that satisfies __simple_allocator but isn't
  // exactly std::allocator
  template <class T>
  struct custom_allocator : std::allocator<T>
  {};

  template <class Resource>
  auto make_underlying_resource()
  {
    if constexpr (std::is_pointer_v<Resource>)
    {
      return std::pmr::unsynchronized_pool_resource();
    }
    else
    {
      return Resource();
    }
  }

  template <class Resource>
  decltype(auto) adaptor_arg_from_resource(Resource* rsc) noexcept
  {
    if constexpr (ex::__simple_allocator<Resource>)
    {
      return *rsc;
    }
    else
    {
      return rsc;
    }
  }

  TEMPLATE_TEST_CASE("exec::__memory_resource_adaptor is constructible",
                     "[types][memory_resource_adaptor]",
                     std::allocator<std::byte>,
                     std::allocator<int>,
                     custom_allocator<std::byte>,
                     custom_allocator<int>,
                     std::pmr::memory_resource*,
                     std::pmr::unsynchronized_pool_resource*)
  {
    using adaptor_t = exec::__memory_resource_adaptor_t<TestType>;

    auto      underlying = make_underlying_resource<TestType>();
    adaptor_t rsc(adaptor_arg_from_resource(&underlying));

    std::pmr::polymorphic_allocator<> alloc(&rsc);

    auto* p = alloc.allocate(4);
    REQUIRE(p != nullptr);
    alloc.deallocate(p, 4);
  }

  template <std::size_t Alignment>
  using alignment = std::integral_constant<std::size_t, Alignment>;

  TEMPLATE_TEST_CASE("exec::__memory_resource_adaptor supports over-alignment",
                     "[types][memory_resource_adaptor]",
                     std::allocator<std::byte>,
                     std::allocator<int>,
                     custom_allocator<std::byte>,
                     custom_allocator<int>,
                     std::pmr::memory_resource*,
                     std::pmr::unsynchronized_pool_resource*)
  {
    using adaptor_t = exec::__memory_resource_adaptor_t<TestType>;

    auto      underlying = make_underlying_resource<TestType>();
    adaptor_t rsc(adaptor_arg_from_resource(&underlying));

    std::pmr::polymorphic_allocator<> alloc(&rsc);

    auto make_assertion = [&alloc]<std::size_t Alignment>(alignment<Alignment>)
    {
      auto* p = alloc.allocate_bytes(10, Alignment);

      // this inlines std::is_sufficiently_aligned<Alignment>(p) so that failures read
      // more clearly in the output (the actual and expected alignment show up in the
      // output this way, instead of "false").
      REQUIRE(std::countr_zero(std::bit_cast<std::uintptr_t>(p)) >= std::countr_zero(Alignment));

      alloc.deallocate_bytes(p, 10, Alignment);
    };

    make_assertion(alignment<1>());
    make_assertion(alignment<2>());
    make_assertion(alignment<4>());
    make_assertion(alignment<8>());
    make_assertion(alignment<16>());
    make_assertion(alignment<32>());
    make_assertion(alignment<64>());
    make_assertion(alignment<128>());
    make_assertion(alignment<256>());
    make_assertion(alignment<512>());
    make_assertion(alignment<1'024>());
    make_assertion(alignment<2'048>());
  }
}  // namespace
