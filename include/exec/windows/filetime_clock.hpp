/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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
#pragma once

#include <chrono>
#include <cstdint>

#include <windows.h>

namespace exec::__win32 {

  class filetime_clock {
   public:
    using rep = std::int64_t;
    using ratio = std::ratio<1, 10'000'000>; // 100ns
    using duration = std::chrono::duration<rep, ratio>;

    static constexpr bool is_steady = false;

    class time_point {
     public:
      using duration = filetime_clock::duration;

      constexpr time_point() noexcept = default;

      constexpr time_point(const time_point &) noexcept = default;

      auto operator=(const time_point &) noexcept -> time_point & = default;

      [[nodiscard]]
      auto get_ticks() const noexcept -> std::uint64_t {
        return ticks_;
      }

      static constexpr auto(max)() noexcept -> time_point {
        time_point tp;
        tp.ticks_ = (std::numeric_limits<std::int64_t>::max)();
        return tp;
      }

      static constexpr auto(min)() noexcept -> time_point {
        return time_point{};
      }

      static auto from_ticks(std::uint64_t ticks) noexcept -> time_point {
        time_point tp;
        tp.ticks_ = ticks;
        return tp;
      }

      template <typename Rep, typename Ratio>
      auto operator+=(const std::chrono::duration<Rep, Ratio> &d) noexcept -> time_point & {
        ticks_ += std::chrono::duration_cast<duration>(d).count();
        return *this;
      }

      template <typename Rep, typename Ratio>
      auto operator-=(const std::chrono::duration<Rep, Ratio> &d) noexcept -> time_point & {
        ticks_ -= std::chrono::duration_cast<duration>(d).count();
        return *this;
      }

      friend auto operator-(time_point a, time_point b) noexcept -> duration {
        return duration{a.ticks_} - duration{b.ticks_};
      }

      template <typename Rep, typename Ratio>
      friend auto
        operator-(time_point t, std::chrono::duration<Rep, Ratio> d) noexcept -> time_point {
        time_point tp = t;
        tp -= d;
        return tp;
      }

      template <typename Rep, typename Ratio>
      friend auto
        operator+(time_point t, std::chrono::duration<Rep, Ratio> d) noexcept -> time_point {
        time_point tp = t;
        tp += d;
        return tp;
      }

      template <typename Rep, typename Ratio>
      friend auto
        operator+(std::chrono::duration<Rep, Ratio> d, time_point t) noexcept -> time_point {
        return t + d;
      }

      friend auto operator==(time_point a, time_point b) noexcept -> bool {
        return a.ticks_ == b.ticks_;
      }

      friend auto operator!=(time_point a, time_point b) noexcept -> bool {
        return a.ticks_ != b.ticks_;
      }

      friend auto operator<(time_point a, time_point b) noexcept -> bool {
        return a.ticks_ < b.ticks_;
      }

      friend auto operator>(time_point a, time_point b) noexcept -> bool {
        return a.ticks_ > b.ticks_;
      }

      friend auto operator<=(time_point a, time_point b) noexcept -> bool {
        return a.ticks_ <= b.ticks_;
      }

      friend auto operator>=(time_point a, time_point b) noexcept -> bool {
        return a.ticks_ >= b.ticks_;
      }

     private:
      // Ticks since Jan 1, 1601 (UTC)
      std::uint64_t ticks_{};
    };

    static auto now() noexcept -> time_point {
      FILETIME filetime;
      ::GetSystemTimeAsFileTime(&filetime);

      ULARGE_INTEGER ticks;
      ticks.HighPart = filetime.dwHighDateTime;
      ticks.LowPart = filetime.dwLowDateTime;

      return time_point::from_ticks(ticks.QuadPart);
    }
  };

} // namespace exec::__win32
