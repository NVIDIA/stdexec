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

#include <cstdint>
#include <type_traits>

namespace STDEXEC
{
  //////////////////////////////////////////////////////////////////////////////////////////
  // __get_completion_behavior
  struct __completion_behavior
  {
    //private:
    template <__completion_tag _Tag>
    friend struct __get_completion_behavior_t;

    enum __flag : std::uint8_t
    {
      __not_affine_ = 1,
      __async_      = 2,
      __inline_     = 4
    };

    enum class __behavior : std::uint8_t
    {
      // The operation will complete asynchronously, and may complete on a different
      // context than the one that started it.
      __asynchronous = __async_ | __not_affine_,

      // The operation will complete asynchronously, but will complete on the same
      // context that started it.
      __asynchronous_affine = __async_,

      // The operation will complete synchronously (before 'start()' returns) on the same
      // thread that started it.
      __inline_completion = __inline_,

      // The operation's completion behavior is unknown.
      __unknown = __not_affine_ | __async_ | __inline_
    };

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

    // __asynchronous        | __asynchronous_affine   => __async_                            (aka __asynchronous)
    // __asynchronous        | __inline_completion     => __async_ | __inline_ | __not_affine (aka __unknown)
    // __asynchronous_affine | __inline_completion     => __async_ | __inline_
    template <__behavior _Left, __behavior _Right>
    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    friend constexpr auto operator|(__constant_t<_Left>, __constant_t<_Right>) noexcept
    {
      return __constant_t<static_cast<__behavior>(static_cast<std::uint8_t>(_Left)
                                                  | static_cast<std::uint8_t>(_Right))>();
    }

    template <__behavior _Left, __behavior _Right>
    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    friend constexpr bool operator==(__constant_t<_Left>, __constant_t<_Right>) noexcept
    {
      return _Left == _Right;
    }

    template <__behavior _CB>
    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    static constexpr bool __is_affine(__constant_t<_CB>) noexcept
    {
      return !(static_cast<std::uint8_t>(_CB) & __not_affine_);
    }

    template <__behavior _CB>
    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    static constexpr bool __is_always_asynchronous(__constant_t<_CB>) noexcept
    {
      return !(static_cast<std::uint8_t>(_CB) & __inline_);
    }

    template <__behavior _CB>
    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    static constexpr bool __may_be_asynchronous(__constant_t<_CB>) noexcept
    {
      return bool(static_cast<std::uint8_t>(_CB) & __async_);
    }

    struct __common_t
    {
      template <__behavior... _CSs>
        requires(sizeof...(_CSs) > 0)
      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      constexpr auto operator()(__constant_t<_CSs>... __cbs) const noexcept
      {
        return (__cbs | ...);
      }
    };

    static constexpr __common_t __common{};
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
                    "variables in STDEXEC::__completion_behavior.");
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

  template <class _Tag, class _Attrs, class... _Env>
  inline constexpr auto __completion_behavior_of_v =
    __call_result_t<__get_completion_behavior_t<_Tag>, _Attrs const &, _Env const &...>{};

  template <class _Tag, class _Attrs, class... _Env>
  concept __completes_inline = (__completion_behavior_of_v<_Tag, _Attrs, _Env...>
                                == __completion_behavior::__inline_completion);

  template <class _Tag, class _Attrs, class... _Env>
  concept __completes_where_it_starts = __completion_behavior::__is_affine(
    __completion_behavior_of_v<_Tag, _Attrs, _Env...>);

  template <class _Tag, class _Sndr, class... _Env>
  STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
  constexpr auto __get_completion_behavior() noexcept
  {
    return __completion_behavior_of_v<_Tag, env_of_t<_Sndr>, _Env...>;
  }

#if !defined(STDEXEC_DOXYGEN_INVOKED)

  struct [[deprecated("Use exec::completion_behavior from "
                      "<exec/completion_behavior.hpp> instead")]] completion_behavior
  {
    using unknown_t             = __completion_behavior::__unknown_t;
    using asynchronous_t        = __completion_behavior::__asynchronous_t;
    using asynchronous_affine_t = __completion_behavior::__asynchronous_affine_t;
    using inline_completion_t   = __completion_behavior::__inline_completion_t;

    static constexpr auto const &unknown             = __completion_behavior::__unknown;
    static constexpr auto const &asynchronous        = __completion_behavior::__asynchronous;
    static constexpr auto const &asynchronous_affine = __completion_behavior::__asynchronous_affine;
    static constexpr auto const &inline_completion   = __completion_behavior::__inline_completion;
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
