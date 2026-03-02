#include <catch2/catch.hpp>
#include <stdexec/__detail/__intrusive_mpsc_queue.hpp>

#include <algorithm>
#include <atomic>
#include <set>
#include <thread>
#include <vector>

namespace
{

  struct test_node
  {
    STDEXEC::__std::atomic<test_node*> next_{nullptr};
    int                                value_{0};

    test_node() = default;

    explicit test_node(int val)
      : value_(val)
    {}
  };

  using test_queue = STDEXEC::__intrusive_mpsc_queue<&test_node::next_>;

  TEST_CASE("intrusive_mpsc_queue with 2 producers and 1 consumer",
            "[detail][intrusive_mpsc_queue]")
  {
    test_queue    queue;
    constexpr int num_items_per_producer = 500;
    constexpr int num_producers          = 2;
    constexpr int total_items            = num_items_per_producer * num_producers;

    std::vector<std::unique_ptr<test_node>> nodes1;
    std::vector<std::unique_ptr<test_node>> nodes2;

    for (int i = 0; i < num_items_per_producer; ++i)
    {
      nodes1.push_back(std::make_unique<test_node>(i * 1000));
      nodes2.push_back(std::make_unique<test_node>(i * 1000 + 1));
    }

    std::atomic<int> produced_count{0};

    std::set<test_node*> consumed_addrs;

    std::thread producer1(
      [&]()
      {
        for (int i = 0; i < num_items_per_producer; ++i)
        {
          queue.push_back(nodes1[i].get());
          produced_count.fetch_add(1, std::memory_order_relaxed);
        }
      });

    std::thread producer2(
      [&]()
      {
        for (int i = 0; i < num_items_per_producer; ++i)
        {
          queue.push_back(nodes2[i].get());
          produced_count.fetch_add(1, std::memory_order_relaxed);
        }
      });

    std::set<int> consumed;
    std::thread   consumer(
      [&]()
      {
        int count = 0;
        while (count < total_items)
        {
          test_node* node = queue.pop_front();
          if (node)
          {
            consumed.insert(node->value_);
            ++count;
          }
          else
          {
            std::this_thread::yield();
          }
        }
      });

    producer1.join();
    producer2.join();
    consumer.join();

    REQUIRE(consumed.size() == total_items);

    for (int i = 0; i < num_items_per_producer; ++i)
    {
      CHECK(consumed.count(i * 1000) == 1);
      CHECK(consumed.count(i * 1000 + 1) == 1);
    }
  }

}  // namespace
