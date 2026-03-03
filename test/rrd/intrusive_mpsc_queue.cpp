/*
 * Copyright (c) 2025 NVIDIA Corporation
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

#include <stdexec_relacy.hpp>

#include <stdexec/__detail/__intrusive_mpsc_queue.hpp>

struct test_node
{
  std::atomic<test_node*> next_{nullptr};
  int                     value_{0};

  test_node() = default;
  explicit test_node(int val)
    : value_(val)
  {}
};

using test_queue = STDEXEC::__intrusive_mpsc_queue<&test_node::next_>;

struct mpsc_single_producer_consumer : rl::test_suite<mpsc_single_producer_consumer, 2>
{
  test_queue queue;
  test_node  node1{42};
  test_node  node2{100};
  int        consumed_count{0};
  int        values_sum{0};

  void thread(unsigned thread_id)
  {
    if (thread_id == 0)
    {
      queue.push_back(&node1);
      queue.push_back(&node2);
    }
    else
    {
      while (consumed_count < 2)
      {
        test_node* node = queue.pop_front();
        if (node)
        {
          values_sum += node->value_;
          ++consumed_count;
        }
      }
    }
  }

  void after()
  {
    RL_ASSERT(consumed_count == 2);
    RL_ASSERT(values_sum == 142);  // 42 + 100
  }
};

struct mpsc_two_producers : rl::test_suite<mpsc_two_producers, 3>
{
  test_queue queue;
  test_node  nodes[4] = {test_node{1}, test_node{2}, test_node{3}, test_node{4}};
  int        consumed_count{0};
  bool       seen[4];

  void before()
  {
    for (int i = 0; i < 4; ++i)
    {
      seen[i] = false;
    }
  }

  void thread(unsigned thread_id)
  {
    if (thread_id == 0)
    {
      // Producer 1
      queue.push_back(&nodes[0]);
      queue.push_back(&nodes[1]);
    }
    else if (thread_id == 1)
    {
      // Producer 2
      queue.push_back(&nodes[2]);
      queue.push_back(&nodes[3]);
    }
    else
    {
      // Consumer
      while (consumed_count < 4)
      {
        test_node* node = queue.pop_front();
        if (node)
        {
          int idx = node->value_ - 1;
          RL_ASSERT(idx >= 0 && idx < 4);
          bool was_seen = std::exchange(seen[idx], true);
          RL_ASSERT(!was_seen);  // Each node should be seen exactly once
          ++consumed_count;
        }
      }
    }
  }

  void after()
  {
    RL_ASSERT(consumed_count == 4);
    for (int i = 0; i < 4; ++i)
    {
      RL_ASSERT(seen[i]);
    }
  }
};

struct mpsc_push_return_value : rl::test_suite<mpsc_push_return_value, 1>
{
  test_queue queue;
  test_node  node1{1};
  test_node  node2{2};
  test_node  node3{3};

  void thread(unsigned thread_id)
  {
    RL_ASSERT(queue.push_back(&node1));
    RL_ASSERT(!queue.push_back(&node2));
    RL_ASSERT(!queue.push_back(&node3));

    queue.pop_front();
    RL_ASSERT(!queue.push_back(&node1));

    queue.pop_front();
    queue.pop_front();
    queue.pop_front();

    RL_ASSERT(queue.push_back(&node1));
    RL_ASSERT(!queue.push_back(&node2));
    RL_ASSERT(!queue.push_back(&node3));
  }
};

struct mpsc_fifo_order : rl::test_suite<mpsc_fifo_order, 2>
{
  test_queue queue;
  test_node  nodes[3] = {test_node{1}, test_node{2}, test_node{3}};
  int        order[3];
  int        consumed_count{0};

  void before()
  {
    for (int i = 0; i < 3; ++i)
    {
      order[i] = -1;
    }
  }

  void thread(unsigned thread_id)
  {
    if (thread_id == 0)
    {
      queue.push_back(&nodes[0]);
      queue.push_back(&nodes[1]);
      queue.push_back(&nodes[2]);
    }
    else
    {
      int pop_order = 0;
      while (consumed_count < 3)
      {
        test_node* node = queue.pop_front();
        if (node)
        {
          int idx    = node->value_ - 1;
          order[idx] = pop_order++;
          ++consumed_count;
        }
      }
    }
  }

  void after()
  {
    RL_ASSERT(consumed_count == 3);
    RL_ASSERT(order[0] == 0);
    RL_ASSERT(order[1] == 1);
    RL_ASSERT(order[2] == 2);
  }
};

struct mpsc_pop_from_empty_never_returns_node
  : rl::test_suite<mpsc_pop_from_empty_never_returns_node, 2>
{
  test_queue        queue;
  test_node         node{99};
  std::atomic<bool> pushed{false};

  void thread(unsigned thread_id)
  {
    if (thread_id == 0)
    {
      queue.push_back(&node);
      pushed.store(true);
    }
    else
    {
      while (!pushed.load())
        ;

      test_node* node = queue.pop_front();
      RL_ASSERT(node->value_ == 99);

      for (int i = 0; i != 10; ++i)
      {
        node = queue.pop_front();
        RL_ASSERT(node == nullptr);
      }
    }
  }
};

struct mpsc_five_prod_one_cons : rl::test_suite<mpsc_five_prod_one_cons, 6>
{
  test_queue queue;
  test_node  nodes[10] = {// Producer 0: values 0-1
                         test_node{0},
                         test_node{1},
                         // Producer 1: values 10000-10001
                         test_node{10000},
                         test_node{10001},
                         // Producer 2: values 20000-20001
                         test_node{20000},
                         test_node{20001},
                         // Producer 3: values 30000-30001
                         test_node{30000},
                         test_node{30001},
                         // Producer 4: values 40000-40001
                         test_node{40000},
                         test_node{40001}};
  int        consumed_count{0};
  bool       seen[10];

  void before()
  {
    for (int i = 0; i < 10; ++i)
    {
      seen[i] = false;
    }
  }

  void thread(unsigned thread_id)
  {
    if (thread_id < 5)
    {
      // Producer threads (0-4)
      int base_idx = thread_id * 2;
      queue.push_back(&nodes[base_idx]);
      queue.push_back(&nodes[base_idx + 1]);
    }
    else
    {
      // Consumer thread (5)
      while (consumed_count < 10)
      {
        test_node* node = queue.pop_front();
        if (node)
        {
          // Map value to index
          int idx;
          if (node->value_ < 10000)
          {
            idx = node->value_;  // 0 or 1
          }
          else
          {
            int producer_id      = node->value_ / 10000;
            int item_in_producer = node->value_ % 10000;
            idx                  = producer_id * 2 + item_in_producer;
          }
          RL_ASSERT(idx >= 0 && idx < 10);
          bool was_seen = std::exchange(seen[idx], true);
          RL_ASSERT(!was_seen);  // Each node should be seen exactly once
          ++consumed_count;
        }
      }
    }
  }

  void after()
  {
    RL_ASSERT(consumed_count == 10);
    for (int i = 0; i < 10; ++i)
    {
      RL_ASSERT(seen[i]);
    }
  }
};

struct mpsc_five_producers_ordered : rl::test_suite<mpsc_five_producers_ordered, 6>
{
  static constexpr int ITEMS_PER_PRODUCER = 100;
  static constexpr int NUM_PRODUCERS      = 5;
  static constexpr int TOTAL_ITEMS        = ITEMS_PER_PRODUCER * NUM_PRODUCERS;

  test_queue queue;
  test_node  nodes[TOTAL_ITEMS];
  int        consumed_count{0};
  int        consumed_values[TOTAL_ITEMS];

  void before()
  {
    for (int i = 0; i < TOTAL_ITEMS; ++i)
    {
      consumed_values[i] = -1;
      // Initialize nodes with their values
      // Producer 0: 1-100, Producer 1: 101-200, etc.
      nodes[i].value_ = i + 1;
    }
  }

  void thread(unsigned thread_id)
  {
    if (thread_id < NUM_PRODUCERS)
    {
      int start_idx = thread_id * ITEMS_PER_PRODUCER;
      for (int i = 0; i < ITEMS_PER_PRODUCER; ++i)
      {
        queue.push_back(&nodes[start_idx + i]);
      }
    }
    else
    {
      while (consumed_count < TOTAL_ITEMS)
      {
        test_node* node = queue.pop_front();
        if (node)
        {
          consumed_values[consumed_count] = node->value_;
          ++consumed_count;
        }
      }
    }
  }

  void after()
  {
    RL_ASSERT(consumed_count == TOTAL_ITEMS);

    // Check that each value appears exactly once
    bool seen[TOTAL_ITEMS + 1] = {false};  // values are 1-500, so need 501 elements
    for (int i = 0; i < TOTAL_ITEMS; ++i)
    {
      int value = consumed_values[i];
      RL_ASSERT(value >= 1 && value <= TOTAL_ITEMS);
      RL_ASSERT(!seen[value]);  // Each value should appear exactly once
      seen[value] = true;
    }

    // Group consumed values into 5 arrays based on their range
    int range_values[NUM_PRODUCERS][ITEMS_PER_PRODUCER];
    int range_counts[NUM_PRODUCERS] = {0};

    for (int i = 0; i < TOTAL_ITEMS; ++i)
    {
      int value = consumed_values[i];
      // Determine which producer this value belongs to (0-4)
      int producer = (value - 1) / ITEMS_PER_PRODUCER;
      RL_ASSERT(producer >= 0 && producer < NUM_PRODUCERS);
      range_values[producer][range_counts[producer]++] = value;
    }

    // Verify each producer contributed exactly ITEMS_PER_PRODUCER items
    for (int producer = 0; producer < NUM_PRODUCERS; ++producer)
    {
      RL_ASSERT(range_counts[producer] == ITEMS_PER_PRODUCER);
    }

    // Verify each range is in ascending order
    for (int producer = 0; producer < NUM_PRODUCERS; ++producer)
    {
      int range_start = producer * ITEMS_PER_PRODUCER + 1;
      for (int i = 0; i < ITEMS_PER_PRODUCER; ++i)
      {
        RL_ASSERT(range_values[producer][i] == range_start + i);
      }
    }
  }
};

auto main(int argc, char** argv) -> int
{
  int             iterations = argc > 1 ? strtol(argv[1], nullptr, 10) : 250000;
  rl::test_params p;
  p.iteration_count       = iterations;
  p.execution_depth_limit = 10000;
  p.search_type           = rl::random_scheduler_type;

#define CHECK(x) if (!(x)) { std::cout << "Test " #x " failed\n"; return 1; }

  printf("Running mpsc_single_producer_consumer...\n");
  CHECK(rl::simulate<mpsc_single_producer_consumer>(p));

  printf("Running mpsc_two_producers...\n");
  CHECK(rl::simulate<mpsc_two_producers>(p));

  printf("Running mpsc_push_return_value...\n");
  CHECK(rl::simulate<mpsc_push_return_value>(p));

  printf("Running mpsc_fifo_order...\n");
  CHECK(rl::simulate<mpsc_fifo_order>(p));

  printf("Running five_prod_one_cons...\n");
  CHECK(rl::simulate<mpsc_five_prod_one_cons>(p));

  printf("Running mpsc_pop_from_empty_never_returns_node...\n");
  CHECK(rl::simulate<mpsc_pop_from_empty_never_returns_node>(p));

  // Beefy test...
  p.iteration_count = 5000;
  printf("Running mpsc_five_producers_ordered...\n");
  CHECK(rl::simulate<mpsc_five_producers_ordered>(p));

  printf("All tests passed!\n");
  return 0;
}
