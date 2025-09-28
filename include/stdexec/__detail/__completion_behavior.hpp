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
#pragma once

#include <compare>
#include <type_traits>
#include <initializer_list>

#include "__config.hpp"
#include "__concepts.hpp"
#include "__query.hpp"
#include "__meta.hpp"
#include "__execution_fwd.hpp"
#include "__type_traits.hpp"

namespace stdexec {
  //////////////////////////////////////////////////////////////////////////////////////////
  // get_completion_behavior
  namespace __completion_behavior {
    enum class completion_behavior : int {
      unknown, ///< The completion behavior is unknown.
      asynchronous, ///< The operation's completion will not happen on the calling thread before `start()`
                    ///< returns.
      synchronous, ///< The operation's completion happens-before the return of `start()`.
      inline_completion ///< The operation completes synchronously within `start()` on the same thread that called
                        ///< `start()`.
    };

    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto operator<=>(completion_behavior __a, completion_behavior __b) noexcept
      -> std::strong_ordering {
      return static_cast<int>(__a) <=> static_cast<int>(__b);
    }

    template <completion_behavior _CB>
    using __constant_t = std::integral_constant<completion_behavior, _CB>;

    using __unknown_t = __constant_t<completion_behavior::unknown>;
    using __asynchronous_t = __constant_t<completion_behavior::asynchronous>;
    using __synchronous_t = __constant_t<completion_behavior::synchronous>;
    using __inline_completion_t = __constant_t<completion_behavior::inline_completion>;
  } // namespace __completion_behavior

  struct min_t;

  struct completion_behavior {
   private:
    template <__completion_behavior::completion_behavior _CB>
    using __constant_t = std::integral_constant<__completion_behavior::completion_behavior, _CB>;

    friend struct min_t;

   public:
    struct unknown_t : __completion_behavior::__unknown_t { };
    struct asynchronous_t : __completion_behavior::__asynchronous_t { };
    struct synchronous_t : __completion_behavior::__synchronous_t { };
    struct inline_completion_t : __completion_behavior::__inline_completion_t { };

    static constexpr unknown_t unknown{};
    static constexpr asynchronous_t asynchronous{};
    static constexpr synchronous_t synchronous{};
    static constexpr inline_completion_t inline_completion{};
  };

  //////////////////////////////////////////////////////////////////////////////////////////
  // get_completion_behavior: A sender can define this attribute to describe the sender's
  // completion behavior
  struct get_completion_behavior_t
    : __query<get_completion_behavior_t, completion_behavior::unknown, __q1<__decay_t>> {
    template <class _Attrs, class... _Env>
    STDEXEC_ATTRIBUTE(always_inline, host, device)
    static constexpr void __validate() noexcept {
      static_assert(
        __nothrow_queryable_with<_Attrs, get_completion_behavior_t, _Env...>,
        "The get_completion_behavior query must be noexcept.");
      static_assert(
        convertible_to<
          __query_result_t<_Attrs, get_completion_behavior_t, _Env...>,
          __completion_behavior::completion_behavior
        >,
        "The get_completion_behavior query must return one of the static member variables in "
        "execution::completion_behavior.");
    }

    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    static constexpr auto query(forwarding_query_t) noexcept -> bool {
      return true;
    }
  };

  struct min_t {
    using __completion_behavior_t = __completion_behavior::completion_behavior;

    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    static constexpr auto __minimum(std::initializer_list<__completion_behavior_t> __cbs) noexcept
      -> __completion_behavior_t {
      auto __result = __completion_behavior::completion_behavior::inline_completion;
      for (auto __cb: __cbs) {
        if (__cb < __result) {
          __result = __cb;
        }
      }
      return __result;
    }

    template <__completion_behavior_t... _CBs>
    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto operator()(completion_behavior::__constant_t<_CBs>...) const noexcept {
      constexpr auto __behavior = __minimum({_CBs...});

      if constexpr (__behavior == completion_behavior::unknown) {
        return completion_behavior::unknown;
      } else if constexpr (__behavior == completion_behavior::asynchronous) {
        return completion_behavior::asynchronous;
      } else if constexpr (__behavior == completion_behavior::synchronous) {
        return completion_behavior::synchronous;
      } else if constexpr (__behavior == completion_behavior::inline_completion) {
        return completion_behavior::inline_completion;
      }
      STDEXEC_UNREACHABLE();
    }
  };

  constexpr min_t min{};

  template <class _Attrs, class... _Env>
  concept __completes_inline =
    (__call_result_t<get_completion_behavior_t, const _Attrs&, const _Env&...>{}
     == completion_behavior::inline_completion);

  template <class _Sndr, class... _Env>
  STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
  consteval auto get_completion_behavior() noexcept {
    using __behavior_t =
      __call_result_t<get_completion_behavior_t, env_of_t<_Sndr>, const _Env&...>;
    return __behavior_t{};
  }

} // namespace stdexec
