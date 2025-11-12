/*
 * Copyright (c) 2023 Maikel Nadolski
 * Copyright (c) 2023 NVIDIA Corporation
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

#include "exec/sequence/marbles.hpp"

#include "exec/sequence/empty_sequence.hpp"
#include "exec/sequence/merge.hpp"
#include "stdexec/__detail/__meta.hpp"
#include <catch2/catch.hpp>

#include <test_common/schedulers.hpp>
#include <test_common/receivers.hpp>
#include <test_common/senders.hpp>
#include <test_common/sequences.hpp>
#include <test_common/type_helpers.hpp>

namespace {

  struct __clock_t {
    using duration = std::chrono::milliseconds;
    using rep = duration::rep;
    using period = duration::period;
    using time_point = std::chrono::time_point<__clock_t>;
    [[maybe_unused]]
    static const bool is_steady = true;

    time_point __now_{};

    [[maybe_unused]]
    time_point now() noexcept {
      return __now_;
    }
  };

  using __marble_t = exec::marble_t<__clock_t>;
  using __marbles_t = std::vector<__marble_t>;

#if STDEXEC_HAS_STD_RANGES()

  TEST_CASE("marbles - parse empty diagram", "[sequence_senders][marbles]") {
    __clock_t __clock{};
    auto marbles = get_marbles_from(__clock, ""_mstr);
    auto expected = __marbles_t{};
    CHECK(0 == marbles.size());
    CHECK(expected == marbles);
  }

  TEST_CASE("marbles - parse never diagram", "[sequence_senders][marbles]") {
    __clock_t __clock{};
    auto marbles = get_marbles_from(__clock, "--"_mstr);
    auto expected = __marbles_t{};
    CHECK(0 == marbles.size());
    CHECK(expected == marbles);
  }

  TEST_CASE("marbles - parse never with values diagram", "[sequence_senders][marbles]") {
    __clock_t __clock{};
    auto marbles = get_marbles_from(__clock, "-a-b-"_mstr);
    auto expected = __marbles_t{
      __marble_t{__clock.now() + 1ms, ex::set_value, 'a'},
      __marble_t{__clock.now() + 3ms, ex::set_value, 'b'}
    };
    CHECK(2 == marbles.size());
    CHECK(expected == marbles);
  }

  TEST_CASE("marbles - parse values diagram", "[sequence_senders][marbles]") {
    __clock_t __clock{};
    auto marbles = get_marbles_from(__clock, "-a-b-|"_mstr);
    auto expected = __marbles_t{
      __marble_t{__clock.now() + 1ms, ex::set_value, 'a'},
      __marble_t{__clock.now() + 3ms, ex::set_value, 'b'},
      __marble_t{__clock.now() + 5ms, sequence_end}
    };
    CHECK(3 == marbles.size());
    CHECK(expected == marbles);
  }

  TEST_CASE("marbles - parse values with skip ms diagram", "[sequence_senders][marbles]") {
    __clock_t __clock{};
    auto marbles = get_marbles_from(__clock, "-a- 20ms b-|"_mstr);
    auto expected = __marbles_t{
      __marble_t{__clock.now() + 1ms, ex::set_value, 'a'},
      __marble_t{__clock.now() + 23ms, ex::set_value, 'b'},
      __marble_t{__clock.now() + 25ms, sequence_end}
    };
    CHECK(3 == marbles.size());
    CHECK(expected == marbles);
  }

  TEST_CASE("marbles - parse values with skip s diagram", "[sequence_senders][marbles]") {
    __clock_t __clock{};
    auto marbles = get_marbles_from(__clock, "-a- 2s b-|"_mstr);
    auto expected = __marbles_t{
      __marble_t{__clock.now() + 1ms, ex::set_value, 'a'},
      __marble_t{__clock.now() + 2003ms, ex::set_value, 'b'},
      __marble_t{__clock.now() + 2005ms, sequence_end}
    };
    CHECK(3 == marbles.size());
    CHECK(expected == marbles);
  }

  TEST_CASE("marbles - parse values with skip m diagram", "[sequence_senders][marbles]") {
    __clock_t __clock{};
    auto marbles = get_marbles_from(__clock, "-a- 2m b-|"_mstr);
    auto expected = __marbles_t{
      __marble_t{__clock.now() + 1ms, ex::set_value, 'a'},
      __marble_t{__clock.now() + 120003ms, ex::set_value, 'b'},
      __marble_t{__clock.now() + 120005ms, sequence_end}
    };
    CHECK(3 == marbles.size());
    CHECK(expected == marbles);
  }

  TEST_CASE("marbles - parse values with skip first diagram", "[sequence_senders][marbles]") {
    __clock_t __clock{};
    auto marbles = get_marbles_from(__clock, "20ms -a-b-|"_mstr);
    auto expected = __marbles_t{
      __marble_t{__clock.now() + 21ms, ex::set_value, 'a'},
      __marble_t{__clock.now() + 23ms, ex::set_value, 'b'},
      __marble_t{__clock.now() + 25ms, sequence_end}
    };
    CHECK(3 == marbles.size());
    CHECK(expected == marbles);
  }

  TEST_CASE("marbles - record marbles of empty_sequence", "[sequence_senders][marbles]") {
    __clock_t __clock{};
    auto actual = record_marbles(__clock, empty_sequence());
    auto expected = get_marbles_from(__clock, "=^|"_mstr);
    CHECK(expected == actual);
  }

  TEST_CASE("marbles - record marbles of range", "[sequence_senders][marbles]") {
    __clock_t __clock{};
    auto actual = record_marbles(__clock, range('0', '3'));
    auto expected = get_marbles_from(__clock, "=^(012|)"_mstr);
    CHECK(expected == actual);
  }

  TEST_CASE("marbles - record marbles of merged ranges", "[sequence_senders][marbles]") {
    __clock_t __clock{};
    auto actual = record_marbles(__clock, merge(range('0', '2'), range('2', '4')));
    auto expected = get_marbles_from(__clock, "=^(0123|)"_mstr);
    CHECK(expected == actual);
  }
#endif // STDEXEC_HAS_STD_RANGES()
} // namespace
