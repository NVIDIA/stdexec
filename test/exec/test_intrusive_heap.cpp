#include <stdexec/__detail/__config.hpp>

#include <bit>
#include <cstddef>

// Exercise the fallback even when the standard library provides std::bit_ceil.
#undef __cpp_lib_int_pow2

#include <exec/detail/intrusive_heap.hpp>

#include <test_common/catch2.hpp>

namespace
{
  struct node
  {
    int   key_{};
    node* prev_{};
    node* left_{};
    node* right_{};
  };

  using heap_t =
    exec::intrusive_heap<node, int, &node::key_, &node::prev_, &node::left_, &node::right_>;

  TEST_CASE("intrusive_heap fallback bit_ceil handles powers of two", "[intrusive_heap]")
  {
    CHECK(exec::detail::bit_ceil(std::size_t{1}) == 1);
    CHECK(exec::detail::bit_ceil(std::size_t{2}) == 2);
    CHECK(exec::detail::bit_ceil(std::size_t{3}) == 4);
    CHECK(exec::detail::bit_ceil(std::size_t{4}) == 4);
    CHECK(exec::detail::bit_ceil(std::size_t{5}) == 8);
    CHECK(exec::detail::bit_ceil(std::size_t{8}) == 8);
  }

  TEST_CASE("intrusive_heap fallback inserts three nodes", "[intrusive_heap]")
  {
    node   first{1};
    node   second{2};
    node   third{3};
    heap_t heap;

    heap.insert(&first);
    heap.insert(&second);
    heap.insert(&third);

    CHECK(heap.front() == &first);
  }
}  // namespace
