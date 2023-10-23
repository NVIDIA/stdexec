#include "exec/__detail/__bwos_lifo_queue.hpp"

#include <catch2/catch.hpp>

TEST_CASE("exec::bwos::lifo_queue - ", "[bwos]") {
  exec::bwos::lifo_queue<int*> queue(8, 2);
  int x = 1;
  int y = 2;
  SECTION("Observers") {
    CHECK(queue.get_block_size() == 2);
  }
  SECTION("Empty Get") {
    CHECK(queue.pop_back() == nullptr);
  }
  SECTION("Empty Steal") {
    CHECK(queue.steal_front() == nullptr);
  }
  SECTION("Put one, get one") {
    CHECK(queue.push_back(&x));
    CHECK(queue.pop_back() == &x);
    CHECK(queue.pop_back() == nullptr);
  }
  SECTION("Put one, steal none") {
    CHECK(queue.push_back(&x));
    CHECK(queue.steal_front() == nullptr);
    CHECK(queue.pop_back() == &x);
  }
  SECTION("Put one, get one, put one, get one") {
    CHECK(queue.push_back(&x));
    CHECK(queue.pop_back() == &x);
    CHECK(queue.push_back(&y));
    CHECK(queue.pop_back() == &y);
    CHECK(queue.pop_back() == nullptr);
  }
  SECTION("Put two, get two") {
    CHECK(queue.push_back(&x));
    CHECK(queue.push_back(&y));
    CHECK(queue.pop_back() == &y);
    CHECK(queue.pop_back() == &x);
    CHECK(queue.pop_back() == nullptr);
  }
  SECTION("Put three, Steal two") {
    CHECK(queue.push_back(&x));
    CHECK(queue.push_back(&y));
    CHECK(queue.push_back(&x));
    CHECK(queue.steal_front() == &x);
    CHECK(queue.steal_front() == &y);
    CHECK(queue.steal_front() == nullptr);
    CHECK(queue.pop_back() == &x);
    CHECK(queue.pop_back() == nullptr);
  }
  SECTION("Put 4, Steal 1, Get 3") {
    CHECK(queue.push_back(&x));
    CHECK(queue.push_back(&y));
    CHECK(queue.push_back(&x));
    CHECK(queue.push_back(&y));
    CHECK(queue.steal_front() == &x);
    CHECK(queue.pop_back() == &y);
    CHECK(queue.pop_back() == &x);
    CHECK(queue.pop_back() == &y);
    CHECK(queue.pop_back() == nullptr);
  }
}