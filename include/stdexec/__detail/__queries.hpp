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
#include "__completion_behavior.hpp"
#include "__concepts.hpp"
#include "__config.hpp"
#include "__query.hpp"

namespace STDEXEC {
  //////////////////////////////////////////////////////////////////////////////////////////////////
  // [exec.queries]
  namespace __queries {
    //////////////////////////////////////////////////////////////////////////////////
    // [exec.get.allocator]
    struct get_allocator_t : __query<get_allocator_t> {
      using __query<get_allocator_t>::operator();

      // defined in __read_env.hpp
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto operator()() const noexcept;

      template <class _Env>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr void __validate() noexcept {
        static_assert(__nothrow_callable<get_allocator_t, const _Env&>);
        static_assert(__allocator_<__call_result_t<get_allocator_t, const _Env&>>);
      }

      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      static consteval auto query(forwarding_query_t) noexcept -> bool {
        return true;
      }
    };

    // NOT TO SPEC:
    struct __is_scheduler_affine_t {
      template <class _Result>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      static consteval auto __ensure_bool_constant() noexcept {
        if constexpr (__is_bool_constant<_Result>) {
          return static_cast<bool>(_Result::value);
        } else {
          static_assert(
            __is_bool_constant<_Result>,
            "The __is_scheduler_affine query must be one of the following forms:\n"
            "  static constexpr bool query(__is_scheduler_affine_t) noexcept;\n"
            "  bool_constant<Bool> query(__is_scheduler_affine_t) const noexcept;\n"
            "  bool_constant<Bool> query(__is_scheduler_affine_t, const Env&) const noexcept;\n");
        }
      }

      template <class _Attrs, class... _Env>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      consteval auto operator()() const noexcept -> bool {
        return __completes_where_it_starts<set_value_t, _Attrs, const _Env&...>;
      }

      template <__queryable_with<__is_scheduler_affine_t> _Attrs, class... _Env>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      consteval auto operator()() const noexcept -> bool {
        if constexpr (__statically_queryable_with<_Attrs, __is_scheduler_affine_t>) {
          return _Attrs::query(__is_scheduler_affine_t());
        } else {
          return __ensure_bool_constant<__query_result_t<_Attrs, __is_scheduler_affine_t>>();
        }
      }

      template <class _Attrs, class _Env>
        requires __queryable_with<_Attrs, __is_scheduler_affine_t, const _Env&>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      consteval auto operator()() const noexcept -> bool {
        using __result_t = __query_result_t<_Attrs, __is_scheduler_affine_t, const _Env&>;
        return __ensure_bool_constant<__result_t>();
      }

      template <class _Attrs, class... _Env>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      consteval auto operator()(const _Attrs&, const _Env&...) const noexcept -> bool {
        return operator()<_Attrs, _Env...>();
      }

      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      static consteval auto query(forwarding_query_t) noexcept -> bool {
        return false;
      }
    };
  } // namespace __queries

  using __queries::get_allocator_t;
  using __queries::__is_scheduler_affine_t;

  inline constexpr get_allocator_t get_allocator{};

  template <class _Sender, class... _Env>
  concept __is_scheduler_affine = requires {
    requires __is_scheduler_affine_t().operator()<env_of_t<_Sender>, _Env...>();
  };

  // The attributes of a sender adaptor that does not introduce asynchrony.
  template <class _Sender>
  struct __sync_attrs {
    [[nodiscard]]
    constexpr auto query(__is_scheduler_affine_t) const noexcept {
      return __mbool<__is_scheduler_affine<_Sender>>();
    }

    template <class _Tag, class... _Env>
    [[nodiscard]]
    constexpr auto query(get_completion_behavior_t<_Tag>, const _Env&...) const noexcept {
      return get_completion_behavior<_Tag, _Sender, _Env...>();
    }

    template <__forwarding_query _Query, class... _Args>
      requires __queryable_with<env_of_t<_Sender>, _Query, _Args...>
    [[nodiscard]]
    constexpr auto query(_Query, _Args&&... __args) const
      noexcept(__nothrow_queryable_with<env_of_t<_Sender>, _Query, _Args...>)
        -> __query_result_t<env_of_t<_Sender>, _Query, _Args...> {
      return __query<_Query>()(get_env(__sndr_), static_cast<_Args&&>(__args)...);
    }

    const _Sender& __sndr_;
  };

  template <class _Sender>
  STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE __sync_attrs(const _Sender&) -> __sync_attrs<_Sender>;
} // namespace STDEXEC
