#include <catch2/catch.hpp>
#include <iostream>

#include "nvexec/detail/memory.cuh"
#include "tracer_resource.h"

namespace {

  TEST_CASE("synchronized pool releases storage", "[cuda][stream][memory][synchronized pool]") {
    tracer_resource resource{};

    {
      nvdetail::synchronized_pool_resource pool{&resource};

      void* ptr_1 = pool.allocate(128, 8);
      void* ptr_2 = pool.allocate(256, 16);
      REQUIRE(ptr_1 != nullptr);
      REQUIRE(ptr_2 != nullptr);
      REQUIRE(ptr_1 != ptr_2);
      REQUIRE(2 == resource.allocations.size());

      pool.deallocate(ptr_2, 256, 16);
      pool.deallocate(ptr_1, 128, 8);
      REQUIRE(2 == resource.allocations.size());
    }

    REQUIRE(0 == resource.allocations.size());
  }

  TEST_CASE("synchronized pool caches allocations", "[cuda][stream][memory][synchronized pool]") {
    tracer_resource resource{};

    {
      nvdetail::synchronized_pool_resource pool{&resource};

      for (int i = 0; i < 10; i++) {
        void* ptr_1 = pool.allocate(128, 8);
        void* ptr_2 = pool.allocate(256, 16);
        REQUIRE(ptr_1 != nullptr);
        REQUIRE(ptr_2 != nullptr);
        REQUIRE(ptr_1 != ptr_2);
        REQUIRE(2 == resource.allocations.size());

        pool.deallocate(ptr_2, 256, 16);
        pool.deallocate(ptr_1, 128, 8);
        REQUIRE(2 == resource.allocations.size());
      }

      REQUIRE(2 == resource.allocations.size());
    }

    REQUIRE(0 == resource.allocations.size());
  }

  TEST_CASE(
    "synchronized pool doesn't touch allocated memory",
    "[cuda][stream][memory][synchronized pool]") {
    tracer_resource resource{};
    nvdetail::synchronized_pool_resource pool{&resource};

    for (int n = 32; n < 512; n *= 2) {
      int bytes = n * sizeof(int);
      int alignment = alignof(int);

      int* ptr = reinterpret_cast<int*>(pool.allocate(bytes, alignment));
      std::iota(ptr, ptr + n, n);
      pool.deallocate(ptr, 128, 8);

      ptr = reinterpret_cast<int*>(pool.allocate(bytes, alignment));
      for (int i = 0; i < n; i++) {
        REQUIRE(ptr[i] == i + n);
      }
      pool.deallocate(ptr, 128, 8);
    }
  }

  TEST_CASE(
    "synchronized pool provides required alignment",
    "[cuda][stream][memory][synchronized pool]") {
    tracer_resource resource{};
    nvdetail::synchronized_pool_resource pool{&resource};

    for (int alignment = 1; alignment < 512; alignment *= 2) {
      void* ptr = pool.allocate(32, alignment);
      INFO("Alignment: " << alignment);
      REQUIRE(reinterpret_cast<uintptr_t>(ptr) % alignment == 0);
      pool.deallocate(ptr, 32, alignment);
    }
  }
} // namespace
