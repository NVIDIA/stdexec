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
#include "__query.hpp"
#include "__utility.hpp"

#include <compare>
#include <type_traits>

namespace STDEXEC
{
  //////////////////////////////////////////////////////////////////////////////////////////
  // __get_completion_behavior
  struct __completion_behavior
  {
    enum class __behavior : int
    {
      __unknown,              ///< The completion behavior is unknown.
      __asynchronous,         ///< The operation's completion will not always happen on
                              ///< the calling thread before `start()` returns.
      __asynchronous_affine,  ///< Like asynchronous, but completes where it starts.
      __inline_completion     ///< The operation completes synchronously within `start()`
                              ///< on the same thread that called `start()`.
    };

    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    friend constexpr auto
    operator<=>(__behavior __a, __behavior __b) noexcept -> std::strong_ordering
    {
      return static_cast<int>(__a) <=> static_cast<int>(__b);
    }

   private:
    template <__behavior _CB>
    using __constant_t = std::integral_constant<__behavior, _CB>;

   public:
    struct __unknown_t : __constant_t<__behavior::__unknown>
    { };
    struct __asynchronous_t : __constant_t<__behavior::__asynchronous>
    { };
    struct __asynchronous_affine_t : __constant_t<__behavior::__asynchronous_affine>
    { };
    struct __inline_completion_t : __constant_t<__behavior::__inline_completion>
    { };

    static constexpr __unknown_t             __unknown{};
    static constexpr __asynchronous_t        __asynchronous{};
    static constexpr __asynchronous_affine_t __asynchronous_affine{};
    static constexpr __inline_completion_t   __inline_completion{};

    struct __weakest_t
    {
      template <__behavior... _CBs>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto operator()(__completion_behavior::__constant_t<_CBs>...) const noexcept
      {
        constexpr auto __behavior = static_cast<__completion_behavior::__behavior>(
          STDEXEC::__umin({static_cast<std::size_t>(_CBs)...}));

        if constexpr (__behavior == __completion_behavior::__unknown)
        {
          return __completion_behavior::__unknown;
        }
        else if constexpr (__behavior == __completion_behavior::__asynchronous)
        {
          return __completion_behavior::__asynchronous;
        }
        else if constexpr (__behavior == __completion_behavior::__asynchronous_affine)
        {
          return __completion_behavior::__asynchronous_affine;
        }
        else if constexpr (__behavior == __completion_behavior::__inline_completion)
        {
          return __completion_behavior::__inline_completion;
        }
        STDEXEC_UNREACHABLE();
      }
    };

    static constexpr __weakest_t __weakest{};
  };

  //////////////////////////////////////////////////////////////////////////////////////////
  // __get_completion_behavior: A sender can define this attribute to describe the sender's
  // completion behavior
  template <__completion_tag _Tag>
  struct __get_completion_behavior_t
  {
   private:
    template <class _Attrs, class... _Env>
    STDEXEC_ATTRIBUTE(always_inline, host, device)
    static constexpr auto __validate() noexcept
    {
      using __result_t = __member_query_result_t<_Attrs, __get_completion_behavior_t, _Env...>;
      static_assert(__nothrow_member_queryable_with<_Attrs, __get_completion_behavior_t, _Env...>,
                    "The __get_completion_behavior query must be noexcept.");
      static_assert(__std::convertible_to<__result_t, __completion_behavior::__behavior>,
                    "The __get_completion_behavior query must return one of the static member "
                    "variables in "
                    "execution::__completion_behavior.");
      return __result_t{};
    }

   public:
    template <class _Sig>
    inline static constexpr __get_completion_behavior_t (*signature)(_Sig) = nullptr;

    template <class _Attrs, class... _Env>
    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto operator()(_Attrs const &, _Env const &...) const noexcept
    {
      if constexpr (__member_queryable_with<_Attrs const &,
                                            __get_completion_behavior_t<_Tag>,
                                            _Env...>)
      {
        return __validate<_Attrs, _Env...>();
      }
      else if constexpr (__member_queryable_with<_Attrs const &, __get_completion_behavior_t<_Tag>>)
      {
        return __validate<_Attrs>();
      }
      else
      {
        return __completion_behavior::__unknown;
      }
    }

    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    static constexpr auto query(forwarding_query_t) noexcept -> bool
    {
      return true;
    }
  };

  template <class... _CBs>
  using __common_completion_behavior_t = __result_of<__completion_behavior::__weakest, _CBs...>;

  template <class _Tag, class _Attrs, class... _Env>
  concept __completes_inline =
    (__call_result_t<__get_completion_behavior_t<_Tag>, _Attrs const &, _Env const &...>{}
     == __completion_behavior::__inline_completion);

  template <class _Tag, class _Attrs, class... _Env>
  concept __completes_where_it_starts =
    (__call_result_t<__get_completion_behavior_t<_Tag>, _Attrs const &, _Env const &...>{}
     >= __completion_behavior::__asynchronous_affine);

  template <class _Tag, class _Sndr, class... _Env>
  STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
  constexpr auto __get_completion_behavior() noexcept
  {
    using __behavior_t =
      __call_result_t<__get_completion_behavior_t<_Tag>, env_of_t<_Sndr>, _Env const &...>;
    return __behavior_t{};
  }

#if !defined(STDEXEC_DOXYGEN_INVOKED)

  struct [[deprecated("Use exec::completion_behavior from "
                      "<exec/completion_behavior.hpp> instead")]] completion_behavior
  {
    using unknown_t             = __completion_behavior::__unknown_t;
    using asynchronous_t        = __completion_behavior::__asynchronous_t;
    using asynchronous_affine_t = __completion_behavior::__asynchronous_affine_t;
    using inline_completion_t   = __completion_behavior::__inline_completion_t;
    using weakest_t             = __completion_behavior::__weakest_t;

    static constexpr auto const &unknown             = __completion_behavior::__unknown;
    static constexpr auto const &asynchronous        = __completion_behavior::__asynchronous;
    static constexpr auto const &asynchronous_affine = __completion_behavior::__asynchronous_affine;
    static constexpr auto const &inline_completion   = __completion_behavior::__inline_completion;
    static constexpr auto const &weakest             = __completion_behavior::__weakest;
  };

  template <__completion_tag _Tag>
  using get_completion_behavior_t [[deprecated("Use exec::get_completion_behavior from "
                                               "<exec/completion_behavior.hpp> instead")]]
  = __get_completion_behavior_t<_Tag>;

  // clang-format off
  template <class _Tag, class _Sndr, class... _Env>
  [[deprecated("Use exec::get_completion_behavior from <exec/completion_behavior.hpp> "
               "instead")]]
  STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
  constexpr auto get_completion_behavior() noexcept
  {
    return __get_completion_behavior<_Tag, _Sndr, _Env...>();
  }
  // clang-format on

#endif  // !defined(STDEXEC_DOXYGEN_INVOKED)

}  // namespace STDEXEC
