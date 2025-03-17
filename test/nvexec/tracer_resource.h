#pragma once

#include <new>
#include <vector>
#include <memory_resource>

#include <catch2/catch.hpp>

namespace nvdetail = nvexec::_strm;

namespace {

  struct tracer_resource : public std::pmr::memory_resource {
    struct allocation_info_t {
      void* ptr;
      size_t bytes;
      size_t alignment;

      bool operator==(const allocation_info_t& other) const noexcept {
        return ptr == other.ptr && bytes == other.bytes && alignment == other.alignment;
      }
    };

    std::vector<allocation_info_t> allocations;

    void* do_allocate(size_t bytes, size_t alignment) override {
      INFO("Allocate: " << bytes << " bytes, " << alignment << " alignment");
      void* ptr = ::operator new[](bytes, std::align_val_t(alignment));
      allocations.push_back(allocation_info_t{ptr, bytes, alignment});
      return ptr;
    }

    void do_deallocate(void* ptr, size_t bytes, size_t alignment) override {
      INFO("Deallocate: " << bytes << " bytes, " << alignment << " alignment");

      auto it =
        std::find(allocations.begin(), allocations.end(), allocation_info_t{ptr, bytes, alignment});

      REQUIRE(it != allocations.end());

      if (it != allocations.end()) {
        allocations.erase(it);
        ::operator delete[](ptr, std::align_val_t(alignment));
      }
    }

    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
      return this == &other;
    }
  };
} // namespace