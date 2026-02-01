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

#include "__execution_fwd.hpp"

// include these after __execution_fwd.hpp
#include "__concepts.hpp"
#include "__config.hpp"
#include "__meta.hpp"
#include "__query.hpp"
#include "__utility.hpp"

#include <compare>
#include <type_traits>

namespace STDEXEC {
  //////////////////////////////////////////////////////////////////////////////////////////
  // get_completion_behavior
  struct min_t;

  struct completion_behavior {
    enum class behavior : int {
      unknown, ///< The completion behavior is unknown.
      asynchronous, ///< The operation's completion will not always happen on the calling thread before `start()`
                    ///< returns.
      asynchronous_affine, ///< Like asynchronous, but completes where it starts.
      inline_completion ///< The operation completes synchronously within `start()` on the same thread that called
                        ///< `start()`.
    };

    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    friend constexpr auto operator<=>(behavior __a, behavior __b) noexcept -> std::strong_ordering {
      return static_cast<int>(__a) <=> static_cast<int>(__b);
    }

   private:
    template <behavior _CB>
    using __constant_t = std::integral_constant<behavior, _CB>;

    using __unknown_t = __constant_t<behavior::unknown>;
    using __asynchronous_t = __constant_t<behavior::asynchronous>;
    using __asynchronous_affine_t = __constant_t<behavior::asynchronous_affine>;
    using __inline_completion_t = __constant_t<behavior::inline_completion>;

    friend struct min_t;

   public:
    struct unknown_t : __unknown_t { };
    struct asynchronous_t : __asynchronous_t { };
    struct asynchronous_affine_t : __asynchronous_affine_t { };
    struct inline_completion_t : __inline_completion_t { };

    static constexpr unknown_t unknown{};
    static constexpr asynchronous_t asynchronous{};
    static constexpr asynchronous_affine_t asynchronous_affine{};
    static constexpr inline_completion_t inline_completion{};

    struct weakest_t {
      template <behavior... _CBs>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto operator()(completion_behavior::__constant_t<_CBs>...) const noexcept {
        constexpr auto __behavior = static_cast<behavior>(
          STDEXEC::__umin({static_cast<std::size_t>(_CBs)...}));

        if constexpr (__behavior == completion_behavior::unknown) {
          return completion_behavior::unknown;
        } else if constexpr (__behavior == completion_behavior::asynchronous) {
          return completion_behavior::asynchronous;
        } else if constexpr (__behavior == completion_behavior::asynchronous_affine) {
          return completion_behavior::asynchronous_affine;
        } else if constexpr (__behavior == completion_behavior::inline_completion) {
          return completion_behavior::inline_completion;
        }
        STDEXEC_UNREACHABLE();
      }
    };

    static constexpr weakest_t weakest{};
  };

  //////////////////////////////////////////////////////////////////////////////////////////
  // get_completion_behavior: A sender can define this attribute to describe the sender's
  // completion behavior
  namespace __queries {
    template <__completion_tag _Tag>
    struct get_completion_behavior_t {
     private:
      template <class _Attrs, class... _Env>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __validate() noexcept {
        using __result_t = __member_query_result_t<_Attrs, get_completion_behavior_t, _Env...>;
        static_assert(
          __nothrow_member_queryable_with<_Attrs, get_completion_behavior_t, _Env...>,
          "The get_completion_behavior query must be noexcept.");
        static_assert(
          __std::convertible_to<__result_t, completion_behavior::behavior>,
          "The get_completion_behavior query must return one of the static member variables in "
          "execution::completion_behavior.");
        return __result_t{};
      }

     public:
      template <class _Sig>
      static inline constexpr get_completion_behavior_t (*signature)(_Sig) = nullptr;

      template <class _Attrs, class... _Env>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto operator()(const _Attrs&, const _Env&...) const noexcept {
        if constexpr (
          __member_queryable_with<const _Attrs&, get_completion_behavior_t<_Tag>, _Env...>) {
          return __validate<_Attrs, _Env...>();
        } else if constexpr (__member_queryable_with<
                               const _Attrs&,
                               get_completion_behavior_t<_Tag>
                             >) {
          return __validate<_Attrs>();
        } else {
          return completion_behavior::unknown;
        }
      }

      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      static constexpr auto query(forwarding_query_t) noexcept -> bool {
        return true;
      }
    };
  } // namespace __queries

  [[deprecated("use STDEXEC::completion_behavior::weakest instead")]]
  inline constexpr const auto& min = completion_behavior::weakest;

  template <class... _CBs>
  using __common_completion_behavior_t = __result_of<completion_behavior::weakest, _CBs...>;

  template <class _Tag, class _Attrs, class... _Env>
  concept __completes_inline =
    (__call_result_t<get_completion_behavior_t<_Tag>, const _Attrs&, const _Env&...>{}
     == completion_behavior::inline_completion);

  template <class _Tag, class _Attrs, class... _Env>
  concept __completes_where_it_starts =
    (__call_result_t<get_completion_behavior_t<_Tag>, const _Attrs&, const _Env&...>{}
     >= completion_behavior::asynchronous_affine);

  template <class _Tag, class _Sndr, class... _Env>
  STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
  constexpr auto get_completion_behavior() noexcept {
    using __behavior_t =
      __call_result_t<get_completion_behavior_t<_Tag>, env_of_t<_Sndr>, const _Env&...>;
    return __behavior_t{};
  }
} // namespace STDEXEC
