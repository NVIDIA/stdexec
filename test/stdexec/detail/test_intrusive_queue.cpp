/*
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
#include <catch2/catch_all.hpp>

#include <stdexec/__detail/__config.hpp>

#include <stdexec/__detail/__intrusive_queue.hpp>

#if STDEXEC_USE_MODULES()
import std;
#else
#  include <array>
#  include <cstddef>
#  include <iterator>
#  include <list>
#  include <random>
#  include <vector>
#endif

namespace
{
  struct test_node
  {
    int        value_{0};
    test_node* next_{nullptr};
  };

  using test_queue = STDEXEC::__intrusive_queue<&test_node::next_>;

  // Walk the raw links to detect lists that still contain nodes from another queue.
  auto links_of(test_queue const & queue) -> std::vector<int>
  {
    std::vector<int> result;
    for (test_node* node = queue.front(); node != nullptr; node = node->next_)
    {
      result.push_back(node->value_);
      if (result.size() > 64)
      {
        FAIL("cycle detected in intrusive queue");
      }
    }
    return result;
  }

  auto items_of(test_queue const & queue) -> std::vector<int>
  {
    std::vector<int> result;
    for (test_node* node: queue)
    {
      result.push_back(node->value_);
    }
    return result;
  }

  void check_queue(test_queue const & queue, std::list<int> const & expected)
  {
    std::vector<int> const expected_vec(expected.begin(), expected.end());
    CHECK(links_of(queue) == expected_vec);
    CHECK(items_of(queue) == expected_vec);
    if (expected.empty())
    {
      CHECK(queue.empty());
      CHECK(queue.front() == nullptr);
      CHECK(queue.back() == nullptr);
    }
    else
    {
      REQUIRE_FALSE(queue.empty());
      REQUIRE(queue.front() != nullptr);
      REQUIRE(queue.back() != nullptr);
      CHECK(queue.front()->value_ == expected.front());
      CHECK(queue.back()->value_ == expected.back());
      CHECK(queue.back()->next_ == nullptr);
    }
  }

  auto nth(test_queue const & queue, std::size_t n) -> test_queue::iterator
  {
    auto it = queue.begin();
    for (std::size_t i = 0; i < n; ++i)
    {
      ++it;
    }
    return it;
  }

  TEST_CASE("intrusive_queue::splice matches std::list::splice", "[detail][intrusive_queue]")
  {
    constexpr std::size_t max_dst = 3;
    constexpr std::size_t max_src = 4;

    for (std::size_t dst_size = 0; dst_size <= max_dst; ++dst_size)
    {
      for (std::size_t src_size = 0; src_size <= max_src; ++src_size)
      {
        for (std::size_t pos = 0; pos <= dst_size; ++pos)
        {
          for (std::size_t first = 0; first <= src_size; ++first)
          {
            for (std::size_t last = first; last <= src_size; ++last)
            {
              CAPTURE(dst_size, src_size, pos, first, last);

              std::vector<test_node> dst_nodes(dst_size);
              std::vector<test_node> src_nodes(src_size);
              test_queue             dst;
              test_queue             src;
              std::list<int>         dst_expected;
              std::list<int>         src_expected;

              for (std::size_t i = 0; i < dst_size; ++i)
              {
                dst_nodes[i].value_ = static_cast<int>(100 + i);
                dst.push_back(&dst_nodes[i]);
                dst_expected.push_back(dst_nodes[i].value_);
              }
              for (std::size_t i = 0; i < src_size; ++i)
              {
                src_nodes[i].value_ = static_cast<int>(i);
                src.push_back(&src_nodes[i]);
                src_expected.push_back(src_nodes[i].value_);
              }

              dst.splice(nth(dst, pos), src, nth(src, first), nth(src, last));
              dst_expected.splice(std::next(dst_expected.begin(), static_cast<std::ptrdiff_t>(pos)),
                                  src_expected,
                                  std::next(src_expected.begin(),
                                            static_cast<std::ptrdiff_t>(first)),
                                  std::next(src_expected.begin(),
                                            static_cast<std::ptrdiff_t>(last)));

              check_queue(dst, dst_expected);
              check_queue(src, src_expected);

              // Both queues must remain independently usable afterwards.
              test_node extra_dst{.value_ = 1000};
              test_node extra_src{.value_ = 2000};
              dst.push_back(&extra_dst);
              src.push_back(&extra_src);
              dst_expected.push_back(extra_dst.value_);
              src_expected.push_back(extra_src.value_);
              check_queue(dst, dst_expected);
              check_queue(src, src_expected);

              dst.clear();
              src.clear();
            }
          }
        }
      }
    }
  }
  TEST_CASE("intrusive_queue::splice remains valid after chained transfers",
            "[detail][intrusive_queue]")
  {
    std::array<test_node, 36> nodes{};
    std::array<test_queue, 3> queues{};
    std::array<std::list<int>, 3> expected{};

    for (std::size_t i = 0; i < nodes.size(); ++i)
    {
      nodes[i].value_ = static_cast<int>(i);
      queues[i % queues.size()].push_back(&nodes[i]);
      expected[i % expected.size()].push_back(nodes[i].value_);
    }

    std::mt19937 rng{0x531CEu};
    for (std::size_t step = 0; step < 1500; ++step)
    {
      std::size_t const src   = rng() % queues.size();
      std::size_t const dst   = (src + 1 + rng() % (queues.size() - 1)) % queues.size();
      std::size_t const first = rng() % (expected[src].size() + 1);
      std::size_t const last  = first + rng() % (expected[src].size() - first + 1);
      std::size_t const pos   = rng() % (expected[dst].size() + 1);

      CAPTURE(step, src, dst, first, last, pos);
      queues[dst].splice(nth(queues[dst], pos),
                         queues[src],
                         nth(queues[src], first),
                         nth(queues[src], last));
      expected[dst].splice(std::next(expected[dst].begin(), static_cast<std::ptrdiff_t>(pos)),
                           expected[src],
                           std::next(expected[src].begin(), static_cast<std::ptrdiff_t>(first)),
                           std::next(expected[src].begin(), static_cast<std::ptrdiff_t>(last)));

      for (std::size_t i = 0; i < queues.size(); ++i)
      {
        check_queue(queues[i], expected[i]);
      }
    }

    for (auto& queue: queues)
    {
      queue.clear();
    }
  }

}  // namespace
